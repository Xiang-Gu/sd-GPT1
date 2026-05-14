import argparse
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from GPT import GPT
from NextTokenDataset import NextTokenDataset
from Tokenizer import Tokenizer


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_text(path, max_chars=None, offset=0):
    with open(path, "r", encoding="utf-8") as f:
        if offset:
            f.seek(offset)
        if max_chars is None:
            return f.read()
        return f.read(max_chars)


def encode_split(data_root, filename, tokenizer, max_chars=None, offset=0):
    text = read_text(Path(data_root) / filename, max_chars, offset=offset)
    return tokenizer.encode(text)


def load_model(run_dir, checkpoint_name, device):
    checkpoint = torch.load(run_dir / checkpoint_name, map_location="cpu")
    config = checkpoint["config"]
    model = GPT.from_config(config).to(device)
    load_result = model.load_state_dict(checkpoint["model_state_dict"])
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint load mismatch: missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )
    model.eval()
    return model, config, checkpoint


def confidence_stats(model, dataloader, device, top_tokens):
    totals = {
        "tokens": 0,
        "loss_sum": 0.0,
        "correct": 0,
        "correct_prob_sum": 0.0,
        "max_prob_sum": 0.0,
        "entropy_sum": 0.0,
    }
    high_loss_targets = Counter()
    wrong_high_conf_targets = Counter()
    loss_chunks = []
    correct_prob_chunks = []
    max_prob_chunks = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            vocab_size = logits.shape[-1]
            flat_logits = logits.reshape(-1, vocab_size)
            flat_y = y.reshape(-1)

            losses = F.cross_entropy(flat_logits, flat_y, reduction="none")
            probs = torch.softmax(flat_logits, dim=-1)
            correct_probs = probs.gather(1, flat_y[:, None]).squeeze(1)
            max_probs, pred = probs.max(dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
            correct = pred == flat_y
            loss_chunks.append(losses.detach().cpu())
            correct_prob_chunks.append(correct_probs.detach().cpu())
            max_prob_chunks.append(max_probs.detach().cpu())

            num_tokens = flat_y.numel()
            totals["tokens"] += num_tokens
            totals["loss_sum"] += losses.sum().item()
            totals["correct"] += correct.sum().item()
            totals["correct_prob_sum"] += correct_probs.sum().item()
            totals["max_prob_sum"] += max_probs.sum().item()
            totals["entropy_sum"] += entropy.sum().item()

            high_loss_k = min(top_tokens, num_tokens)
            _, high_loss_indices = torch.topk(losses, high_loss_k)
            high_loss_targets.update(flat_y[high_loss_indices].cpu().tolist())

            wrong_mask = ~correct
            wrong_indices = torch.nonzero(wrong_mask, as_tuple=False).squeeze(1)
            if wrong_indices.numel() > 0:
                wrong_conf = max_probs[wrong_indices]
                wrong_k = min(top_tokens, wrong_indices.numel())
                _, local_indices = torch.topk(wrong_conf, wrong_k)
                wrong_high_conf_indices = wrong_indices[local_indices]
                wrong_high_conf_targets.update(flat_y[wrong_high_conf_indices].cpu().tolist())

    tokens = max(totals["tokens"], 1)
    all_losses = torch.cat(loss_chunks) if loss_chunks else torch.empty(0)
    all_correct_probs = (
        torch.cat(correct_prob_chunks) if correct_prob_chunks else torch.empty(0)
    )
    all_max_probs = torch.cat(max_prob_chunks) if max_prob_chunks else torch.empty(0)

    return {
        "tokens": totals["tokens"],
        "loss": totals["loss_sum"] / tokens,
        "accuracy": totals["correct"] / tokens,
        "avg_correct_prob": totals["correct_prob_sum"] / tokens,
        "avg_max_prob": totals["max_prob_sum"] / tokens,
        "avg_entropy": totals["entropy_sum"] / tokens,
        "loss_quantiles": quantiles(all_losses),
        "correct_prob_quantiles": quantiles(all_correct_probs),
        "max_prob_quantiles": quantiles(all_max_probs),
        "high_loss_targets": high_loss_targets,
        "wrong_high_conf_targets": wrong_high_conf_targets,
    }


def quantiles(values):
    if values.numel() == 0:
        return {}
    qs = torch.tensor([0.0, 0.5, 0.9, 0.95, 0.99, 1.0], dtype=torch.float32)
    out = torch.quantile(values.float(), qs)
    return {
        "min": out[0].item(),
        "p50": out[1].item(),
        "p90": out[2].item(),
        "p95": out[3].item(),
        "p99": out[4].item(),
        "max": out[5].item(),
    }


def make_dataloader(ids, L, batch_size, max_batches=None):
    dataset = NextTokenDataset(ids, L)
    if len(dataset) == 0:
        raise ValueError("Not enough tokens to build a dataset.")
    if max_batches is not None:
        max_examples = min(len(dataset), max_batches * batch_size)
        dataset.ids = dataset.ids[: max_examples * L + 1]
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def printable_token(token):
    return repr(token).replace("\n", "\\n")


def print_counter(counter, tokenizer, title, limit):
    print(f"  {title}:")
    for token_id, count in counter.most_common(limit):
        token = tokenizer.itot.get(int(token_id), "<missing>")
        print(f"    {count:>5,}  id={token_id:<4} len={len(token):<3} {printable_token(token)}")


def print_stats(split_name, stats, tokenizer, top_tokens):
    print(f"{split_name}:")
    print(f"  tokens: {stats['tokens']:,}")
    print(f"  loss: {stats['loss']:.4f}")
    print(f"  accuracy: {100 * stats['accuracy']:.2f}%")
    print(f"  avg correct-token probability: {stats['avg_correct_prob']:.6f}")
    print(f"  avg max probability: {stats['avg_max_prob']:.6f}")
    print(f"  avg entropy: {stats['avg_entropy']:.4f}")
    print_quantiles("loss", stats["loss_quantiles"])
    print_quantiles("correct-token probability", stats["correct_prob_quantiles"])
    print_quantiles("max probability", stats["max_prob_quantiles"])
    print_counter(stats["high_loss_targets"], tokenizer, "targets among highest-loss positions", top_tokens)
    print_counter(
        stats["wrong_high_conf_targets"],
        tokenizer,
        "targets among highest-confidence wrong positions",
        top_tokens,
    )


def print_quantiles(name, values):
    print(
        f"  {name}: "
        f"min={values['min']:.4f} "
        f"p50={values['p50']:.4f} "
        f"p90={values['p90']:.4f} "
        f"p95={values['p95']:.4f} "
        f"p99={values['p99']:.4f} "
        f"max={values['max']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate confidence/overfitting diagnostics for a saved model."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        choices=["best.pt", "latest.pt"],
        default="best.pt",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/tinystories"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--max-chars", type=int)
    parser.add_argument("--train-max-chars", type=int)
    parser.add_argument("--val-max-chars", type=int)
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--val-offset", type=int, default=0)
    parser.add_argument("--top-tokens", type=int, default=10)
    args = parser.parse_args()

    device = get_device()
    tokenizer = Tokenizer.load(args.run_dir / "tokenizer.json")
    model, config, checkpoint = load_model(args.run_dir, args.checkpoint, device)
    batch_size = args.batch_size or config["B"]

    train_max_chars = args.train_max_chars
    val_max_chars = args.val_max_chars
    if args.max_chars is not None:
        train_max_chars = train_max_chars or args.max_chars
        val_max_chars = val_max_chars or args.max_chars

    print(f"run: {args.run_dir}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"checkpoint epoch: {int(checkpoint['epoch']) + 1}, step: {checkpoint['step']}")
    print(f"device: {device}")
    print(f"tokenizer: {type(tokenizer).__name__}, vocab size: {tokenizer.vocab_size():,}")
    print(
        f"model: N={config['N']} d_model={config['d_model']} d_ff={config['d_ff']} "
        f"h={config['h']} L={config['L']} B={batch_size}"
    )
    print()

    train_ids = encode_split(
        args.data_root,
        "input_train.txt",
        tokenizer,
        train_max_chars,
        offset=args.train_offset,
    )
    val_ids = encode_split(
        args.data_root,
        "input_val.txt",
        tokenizer,
        val_max_chars,
        offset=args.val_offset,
    )
    train_loader = make_dataloader(train_ids, config["L"], batch_size, args.max_batches)
    val_loader = make_dataloader(val_ids, config["L"], batch_size, args.max_batches)

    train_stats = confidence_stats(model, train_loader, device, args.top_tokens)
    val_stats = confidence_stats(model, val_loader, device, args.top_tokens)
    print_stats("train", train_stats, tokenizer, args.top_tokens)
    print_stats("validation", val_stats, tokenizer, args.top_tokens)


if __name__ == "__main__":
    main()

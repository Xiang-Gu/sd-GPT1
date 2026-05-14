import argparse
import csv
import torch
from pathlib import Path

from Tokenizer import Tokenizer
from GPT import GPT


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_weights(model, model_ckpt, device):
    config = model_ckpt["config"]
    load_result = model.load_state_dict(model_ckpt["model_state_dict"])

    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []
    assert model.embeds.num_embeddings == config["vocab_size"]
    assert model.embeds.embedding_dim == config["d_model"]
    assert len(model.blocks) == config["N"]
    assert model.final_linear.out_features == config["vocab_size"]
    assert model.final_linear.in_features == config["d_model"]

    x = torch.randint(
        0, config["vocab_size"], (1, config["L"]), dtype=torch.long, device=device
    )
    logits = model(x)

    assert logits.shape == (1, config["L"], config["vocab_size"])
    assert torch.isfinite(logits).all()

    next_token = model.sample_next_token_id(x)
    assert next_token.shape == (1,)
    assert next_token.dtype == torch.long
    assert 0 <= next_token.item() < config["vocab_size"]


def read_validation_losses(run_dir):
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return []

    losses = []
    with open(metrics_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["val_loss"]:
                losses.append(
                    {
                        "epoch": int(row["epoch"]),
                        "step": int(row["step"]),
                        "val_loss": float(row["val_loss"]),
                    }
                )
    return losses


def find_best_run(runs_root):
    candidates = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "best.pt").exists():
            continue
        if not (run_dir / "tokenizer.json").exists():
            continue

        losses = read_validation_losses(run_dir)
        if not losses:
            continue

        best = min(losses, key=lambda row: row["val_loss"])
        candidates.append((best["val_loss"], best["step"], best["epoch"], run_dir))

    if not candidates:
        raise RuntimeError(f"No run with metrics.csv, tokenizer.json, and best.pt under {runs_root}.")

    return min(candidates, key=lambda item: item[0])


def load_model_from_run(run_dir, device):
    losses = read_validation_losses(run_dir)
    best = min(losses, key=lambda row: row["val_loss"]) if losses else None
    tokenizer = Tokenizer.load(run_dir / "tokenizer.json")
    model_ckpt = torch.load(run_dir / "best.pt", map_location="cpu")
    config = model_ckpt["config"]
    model = GPT.from_config(config).to(device)
    load_model_weights(model, model_ckpt, device)
    model.eval()

    print(f"Loaded run: {run_dir}")
    if best is not None:
        print(
            f"validation loss {best['val_loss']:.4f} "
            f"at epoch {best['epoch'] + 1}, step {best['step']}"
        )
    print_config(config)
    return tokenizer, model, config


def print_config(config):
    tokenizer_type = config.get("tokenizer_type", "char")
    max_vocab_size = config.get("max_vocab_size", "")
    print(
        f"config: tokenizer={tokenizer_type} max_vocab_size={max_vocab_size} "
        f"N={config['N']} d_model={config['d_model']} d_ff={config['d_ff']} "
        f"h={config['h']} B={config['B']} L={config['L']}"
    )


def load_best_model(runs_root, device):
    best_val_loss, best_step, best_epoch, run_dir = find_best_run(runs_root)
    tokenizer = Tokenizer.load(run_dir / "tokenizer.json")
    model_ckpt = torch.load(run_dir / "best.pt", map_location="cpu")
    config = model_ckpt["config"]
    model = GPT.from_config(config).to(device)
    load_model_weights(model, model_ckpt, device)
    model.eval()

    print(f"Loaded best run: {run_dir}")
    print(
        f"validation loss {best_val_loss:.4f} at epoch {best_epoch + 1}, step {best_step}"
    )
    print_config(config)
    return tokenizer, model, config


def sample_text(tokenizer, model, config, prompt, max_new_tokens, temperature, top_k, device):
    ids = tokenizer.encode(prompt)
    if not ids:
        return ""

    for _ in range(max_new_tokens):
        context = ids[-config["L"] :]
        x = torch.tensor([context], dtype=torch.long, device=device)
        next_token_id = model.sample_next_token_id(x, temperature, top_k).item()
        ids.append(next_token_id)

    return tokenizer.decode(ids)


def main():
    parser = argparse.ArgumentParser(
        description="Sample text from the best checkpoint under runs/."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature. Lower is more conservative; default: 0.8.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Sample only from the top K next-token candidates; default: 50.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=400,
        help="Maximum number of tokens to generate after the prompt; default: 400.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Load best.pt from this specific run directory instead of the best run under runs/.",
    )
    args = parser.parse_args()

    if args.temperature <= 0:
        parser.error("--temperature must be positive.")
    if args.top_k <= 0:
        parser.error("--top-k must be positive.")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive.")

    runs_root = Path("runs")
    device = get_device()
    if args.run_dir is None:
        tokenizer, model, config = load_best_model(runs_root, device)
    else:
        tokenizer, model, config = load_model_from_run(args.run_dir, device)

    while True:
        prompt = input("TinyStories prompt: ")
        if prompt.strip() in {"q", "quit", "exit"}:
            break

        generated = sample_text(
            tokenizer,
            model,
            config,
            prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            device,
        )
        if generated:
            print(generated)


if __name__ == "__main__":
    main()

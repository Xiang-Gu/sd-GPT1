import math
import csv
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from torch import nn, optim
from torch.utils.data import DataLoader
from GPT import GPT
import torch
from NextTokenDataset import NextTokenDataset
from Tokenizer import BPETokenizer, Tokenizer


DEFAULT_DATA_ROOT = "data/tinystories"


def get_device():
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device = {device}")
    return device


def build_config():
    return {
        "max_vocab_size": 3000,
        "tokenizer_type": "char",
        "N": 2,
        "d_model": 128,
        "d_ff": 512,
        "h": 4,
        "B": 64,
        "L": 160,
        "epochs": 5,
        "lr": 5e-4,
        "warmup_steps": 1000,
        "log_every": 100,
        "grad_clip": 1.0,
        "dropout": 0.1,
        "weight_tying": True,
        "early_stopping_patience": 3,
        "early_stopping_min_delta": 0.0,
    }


def experiment_configs():
    base = build_config()
    return {
        "baseline-char": {
            **base,
            "run_name": "baseline_char_N2_D128_L160_B64_E5",
            "tokenizer_type": "char",
            "N": 2,
            "d_model": 128,
            "d_ff": 512,
            "h": 4,
            "B": 64,
            "L": 160,
            "epochs": 5,
        },
        "big-char": {
            **base,
            "run_name": "big_char_N4_D256_L160_B64_E5",
            "tokenizer_type": "char",
            "N": 4,
            "d_model": 256,
            "d_ff": 1024,
            "h": 8,
            "B": 64,
            "L": 160,
            "epochs": 5,
        },
        "bpe-base": {
            **base,
            "run_name": "bpe_vocab1000_N2_D128_L160_B64_E5",
            "tokenizer_type": "bpe",
            "max_vocab_size": 1000,
            "N": 2,
            "d_model": 128,
            "d_ff": 512,
            "h": 4,
            "B": 64,
            "L": 160,
            "epochs": 5,
        },
    }


def create_run_dir(config, runs_root="runs"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(runs_root) / f"{timestamp}_{config['run_name']}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_config(run_dir, config):
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


METRIC_FIELDNAMES = [
    "epoch",
    "step",
    "train_loss",
    "val_loss",
    "lr",
    "grad_norm_before_clip",
    "grad_norm_after_clip",
    "tokens_seen",
    "tokens_per_sec",
    "elapsed_sec",
    "epoch_elapsed_sec",
    "avg_step_sec",
    "epoch_eta_sec",
    "skipped_updates",
]


def append_metric(run_dir, row):
    path = run_dir / "metrics.csv"
    write_header = not path.exists()

    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def build_lr_scheduler(optimizer, total_steps, warmup_steps):
    warmup_steps = min(warmup_steps, max(1, total_steps - 1))

    def lr_multiplier(step):
        if step < warmup_steps:
            return step / warmup_steps

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)


def create_tokenizer(config):
    tokenizer_type = config.get("tokenizer_type", "char")
    if tokenizer_type == "char":
        return Tokenizer(max_vocab_size=config["max_vocab_size"])
    if tokenizer_type == "bpe":
        return BPETokenizer(max_vocab_size=config["max_vocab_size"])
    raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}")


def build_tokenizer_and_encode(data_root, run_dir, config, filename):
    with open(Path(data_root) / filename, "r", encoding="utf-8") as file:
        text = file.read()

    tokenizer = create_tokenizer(config)
    tokenizer.fit(text)

    print(f"Built a tokenizer of vocabulary size {tokenizer.vocab_size()}\n")
    tokenizer.save(run_dir / "tokenizer.json")
    return tokenizer, tokenizer.encode(text)


def encode_file(data_root, filename, tokenizer):
    with open(Path(data_root) / filename, "r", encoding="utf-8") as file:
        text = file.read()
    return tokenizer.encode(text)


def build_dataloaders(data_root, run_dir, config, tokenizer=None):
    if tokenizer is None:
        tokenizer, train_ids = build_tokenizer_and_encode(
            data_root, run_dir, config, "input_train.txt"
        )
    else:
        train_ids = encode_file(data_root, "input_train.txt", tokenizer)

    validation_ids = encode_file(data_root, "input_val.txt", tokenizer)

    train_dataset = NextTokenDataset(train_ids, config["L"])
    validation_dataset = NextTokenDataset(validation_ids, config["L"])

    train_dataloader = DataLoader(train_dataset, batch_size=config["B"], shuffle=True)
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=config["B"], shuffle=True
    )
    x, y = next(iter(train_dataloader))
    assert x.shape == y.shape == (config["B"], config["L"])
    x, y = next(iter(validation_dataloader))
    assert x.shape == y.shape == (config["B"], config["L"])
    return train_dataloader, validation_dataloader, tokenizer.vocab_size()


def move_batch_to_device(batch, device):
    x, y = batch
    return x.to(device), y.to(device)


def should_log(batch_idx, log_every):
    return (batch_idx + 1) % log_every == 0


def compute_loss(model, X, y, loss_fn):
    # pred.shape = (B, L, vocab_size), y.shape = (B, L)
    pred = model(X)
    B, L, vocab_size = pred.shape
    # `loss_fn` is a CrossEntropyLoss which takes as input logits=(B, vocab_size), label=(B,)
    return loss_fn(pred.reshape(B * L, vocab_size), y.reshape(B * L))


def train_step(model, X, y, loss_fn, optimizer, lr_scheduler=None, grad_clip=None):
    optimizer.zero_grad(set_to_none=True)
    loss = compute_loss(model, X, y, loss_fn)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite loss detected before backward: {loss.item()}")

    loss.backward()
    grad_norm_before_clip = total_grad_norm(model.parameters())
    if not torch.isfinite(grad_norm_before_clip):
        bad_grad_name = first_nonfinite_grad_name(model)
        optimizer.zero_grad(set_to_none=True)
        return (
            loss,
            grad_norm_before_clip,
            grad_norm_before_clip,
            False,
            bad_grad_name,
        )

    if grad_clip is not None:
        grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip, error_if_nonfinite=True
        )
        grad_norm_after_clip = torch.minimum(
            grad_norm_before_clip,
            torch.tensor(float(grad_clip), device=grad_norm_before_clip.device),
        )
    else:
        grad_norm_after_clip = grad_norm_before_clip

    optimizer.step()
    if lr_scheduler is not None:
        lr_scheduler.step()

    return loss, grad_norm_before_clip, grad_norm_after_clip, True, ""


def first_nonfinite_grad_name(model):
    for name, param in model.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            return name
    return ""


def total_grad_norm(parameters):
    grads = [p.grad.detach().norm(2) for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.stack(grads), ord=2)


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def progress_timing(train_start_time, epoch_start_time, batch_idx, dataloader, optimizer_steps):
    now = time.perf_counter()
    elapsed_sec = now - train_start_time
    epoch_elapsed_sec = now - epoch_start_time
    completed_batches = batch_idx + 1
    avg_batch_sec = epoch_elapsed_sec / max(completed_batches, 1)
    avg_step_sec = epoch_elapsed_sec / max(optimizer_steps, 1)
    remaining_batches = len(dataloader) - completed_batches
    epoch_eta_sec = avg_batch_sec * remaining_batches
    return elapsed_sec, epoch_elapsed_sec, avg_step_sec, epoch_eta_sec


def log_progress(
    epoch,
    num_epochs,
    batch_idx,
    current_batch_size,
    dataloader,
    global_step,
    avg_loss,
    grad_norm_before_clip,
    grad_norm_after_clip,
    total_elapsed_sec,
    epoch_elapsed_sec,
    avg_step_sec,
    epoch_eta_sec,
):
    seen = batch_idx * dataloader.batch_size + current_batch_size
    seen = min(seen, len(dataloader.dataset))
    total = len(dataloader.dataset)
    percent = 100 * seen / total

    print(
        f"epoch {epoch + 1}/{num_epochs} "
        f"batch {batch_idx + 1}/{len(dataloader)} "
        f"step {global_step} "
        f"loss {avg_loss:.4f} "
        f"grad_norm_before_clip {grad_norm_before_clip:.2f} "
        f"grad_norm_after_clip {grad_norm_after_clip:.2f} "
        f"[{seen}/{total} | {percent:.1f}%] "
        f"elapsed {format_duration(total_elapsed_sec)} "
        f"epoch_elapsed {format_duration(epoch_elapsed_sec)} "
        f"avg_step {avg_step_sec:.3f}s "
        f"epoch_eta {format_duration(epoch_eta_sec)}"
    )


def current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def evaluate_model(device, validation_dataloader, model, loss_fn):
    num_tokens = 0
    num_batches = len(validation_dataloader)

    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for x, y in validation_dataloader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)  # (B, L, vocab_size)
            B, L, vocab_size = pred.shape
            test_loss += loss_fn(
                pred.reshape(B * L, vocab_size), y.reshape(B * L)
            ).item()
            correct += (pred.argmax(dim=-1) == y).sum().item()
            num_tokens += B * L

    test_loss /= num_batches
    accuracy = correct / num_tokens
    print(f"Validation: accuracy {100 * accuracy:.2f}%, loss {test_loss:.4f}\n")
    return test_loss


def save_checkpoint(
    run_dir,
    model,
    optimizer,
    lr_scheduler,
    epoch,
    global_step,
    loss,
    config,
    best_val_loss,
    epochs_without_improvement,
    is_best_model_ck,
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "epoch": epoch,
        "step": global_step,
        "loss": float(loss.detach().cpu()),
        "best_val_loss": best_val_loss,
        "epochs_without_improvement": epochs_without_improvement,
        "config": config,
    }

    torch.save(checkpoint, run_dir / f"epoch_{epoch}.pt")
    torch.save(checkpoint, run_dir / "latest.pt")
    if is_best_model_ck:
        torch.save(checkpoint, run_dir / "best.pt")


def train(
    device,
    run_dir,
    train_dataloader,
    validation_dataloader,
    model,
    loss_fn,
    optimizer,
    lr_scheduler,
    num_epochs,
    config,
    start_epoch=0,
    global_step=0,
    best_val_loss=float("inf"),
    epochs_without_improvement=0,
):
    log_every = config["log_every"]
    grad_clip = config["grad_clip"]
    early_stopping_patience = config["early_stopping_patience"]
    early_stopping_min_delta = config["early_stopping_min_delta"]
    loss = torch.zeros((), device=device)
    train_start_time = time.perf_counter()

    if start_epoch >= num_epochs:
        print(
            f"Nothing to train: checkpoint already reached epoch {start_epoch}, "
            f"target epochs is {num_epochs}."
        )
        return

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = torch.zeros((), device=device)
        latest_grad_norm_before_clip = torch.zeros((), device=device)
        latest_grad_norm_after_clip = torch.zeros((), device=device)
        skipped_updates = 0
        epoch_start_time = time.perf_counter()
        epoch_optimizer_steps = 0

        for batch_idx, batch in enumerate(train_dataloader):
            x, y = move_batch_to_device(batch, device)
            (
                loss,
                grad_norm_before_clip,
                grad_norm_after_clip,
                did_step,
                bad_grad_name,
            ) = train_step(
                model, x, y, loss_fn, optimizer, lr_scheduler, grad_clip=grad_clip
            )
            running_loss += loss.detach()
            latest_grad_norm_before_clip = grad_norm_before_clip.detach()
            latest_grad_norm_after_clip = grad_norm_after_clip.detach()
            if did_step:
                global_step += 1
                epoch_optimizer_steps += 1
            else:
                skipped_updates += 1
                bad_grad_details = f" in {bad_grad_name}" if bad_grad_name else ""
                print(
                    "warning: skipped optimizer update for non-finite gradient"
                    f"{bad_grad_details} at epoch {epoch + 1} "
                    f"batch {batch_idx + 1}; loss {loss.item():.4f}"
                )

            if should_log(batch_idx, log_every):
                avg_loss = float((running_loss / log_every).cpu())
                grad_norm_before_clip_value = float(latest_grad_norm_before_clip.cpu())
                grad_norm_after_clip_value = float(latest_grad_norm_after_clip.cpu())
                (
                    elapsed_sec,
                    epoch_elapsed_sec,
                    avg_step_sec,
                    epoch_eta_sec,
                ) = progress_timing(
                    train_start_time,
                    epoch_start_time,
                    batch_idx,
                    train_dataloader,
                    epoch_optimizer_steps,
                )
                tokens_seen = global_step * config["B"] * config["L"]
                tokens_per_sec = tokens_seen / max(elapsed_sec, 1e-9)
                log_progress(
                    epoch,
                    num_epochs,
                    batch_idx,
                    len(x),
                    train_dataloader,
                    global_step,
                    avg_loss,
                    grad_norm_before_clip_value,
                    grad_norm_after_clip_value,
                    elapsed_sec,
                    epoch_elapsed_sec,
                    avg_step_sec,
                    epoch_eta_sec,
                )
                append_metric(
                    run_dir,
                    {
                        "epoch": epoch,
                        "step": global_step,
                        "train_loss": avg_loss,
                        "val_loss": "",
                        "lr": current_lr(optimizer),
                        "grad_norm_before_clip": grad_norm_before_clip_value,
                        "grad_norm_after_clip": grad_norm_after_clip_value,
                        "tokens_seen": tokens_seen,
                        "tokens_per_sec": tokens_per_sec,
                        "elapsed_sec": elapsed_sec,
                        "epoch_elapsed_sec": epoch_elapsed_sec,
                        "avg_step_sec": avg_step_sec,
                        "epoch_eta_sec": epoch_eta_sec,
                        "skipped_updates": skipped_updates,
                    },
                )
                running_loss.zero_()

        if skipped_updates:
            print(f"epoch {epoch + 1}: skipped {skipped_updates} optimizer update(s)")

        val_loss = evaluate_model(device, validation_dataloader, model, loss_fn)
        epoch_elapsed_sec = time.perf_counter() - epoch_start_time
        elapsed_sec = time.perf_counter() - train_start_time
        avg_step_sec = epoch_elapsed_sec / max(epoch_optimizer_steps, 1)
        tokens_seen = global_step * config["B"] * config["L"]
        tokens_per_sec = tokens_seen / max(elapsed_sec, 1e-9)
        append_metric(
            run_dir,
            {
                "epoch": epoch,
                "step": global_step,
                "train_loss": "",
                "val_loss": val_loss,
                "lr": current_lr(optimizer),
                "grad_norm_before_clip": float(latest_grad_norm_before_clip.cpu()),
                "grad_norm_after_clip": float(latest_grad_norm_after_clip.cpu()),
                "tokens_seen": tokens_seen,
                "tokens_per_sec": tokens_per_sec,
                "elapsed_sec": elapsed_sec,
                "epoch_elapsed_sec": epoch_elapsed_sec,
                "avg_step_sec": avg_step_sec,
                "epoch_eta_sec": 0.0,
                "skipped_updates": skipped_updates,
            },
        )
        improvement = best_val_loss - val_loss
        is_best_model_ck = improvement > early_stopping_min_delta
        if is_best_model_ck:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            run_dir,
            model,
            optimizer,
            lr_scheduler,
            epoch,
            global_step,
            loss,
            config,
            best_val_loss,
            epochs_without_improvement,
            is_best_model_ck,
        )

        if epochs_without_improvement >= early_stopping_patience:
            print(
                "early stopping... "
                f"validation loss has not improved by at least "
                f"{early_stopping_min_delta} for {early_stopping_patience} epochs"
            )
            break


def run_training(config, data_root=DEFAULT_DATA_ROOT):
    run_dir = create_run_dir(config)
    print(f"run_dir = {run_dir}")

    train_dataloader, validation_dataloader, actual_vocab_size = build_dataloaders(
        data_root, run_dir, config
    )
    config["vocab_size"] = actual_vocab_size
    save_config(run_dir, config)

    device = get_device()
    model = GPT(
        config["vocab_size"],
        config["N"],
        config["d_model"],
        config["d_ff"],
        config["h"],
        max_seq_len=config["L"],
        dropout=config["dropout"],
        weight_tying=config["weight_tying"],
    ).to(device)
    config["param_count"] = count_parameters(model)
    save_config(run_dir, config)
    print(f"parameter count = {config['param_count']:,}")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
    total_steps = len(train_dataloader) * config["epochs"]
    lr_scheduler = build_lr_scheduler(optimizer, total_steps, config["warmup_steps"])

    train(
        device,
        run_dir,
        train_dataloader,
        validation_dataloader,
        model,
        loss_fn,
        optimizer,
        lr_scheduler,
        config["epochs"],
        config,
    )

    return run_dir


def load_resume_checkpoint(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    run_dir = checkpoint_path.parent
    return run_dir, checkpoint


def run_resume_training(
    checkpoint_path,
    data_root=DEFAULT_DATA_ROOT,
    epochs=None,
    lr=None,
    grad_clip=None,
    fresh_optimizer=False,
):
    run_dir, checkpoint = load_resume_checkpoint(checkpoint_path)
    config = dict(checkpoint["config"])
    if epochs is not None:
        config["epochs"] = epochs
    if lr is not None:
        config["lr"] = lr
        fresh_optimizer = True
    if grad_clip is not None:
        config["grad_clip"] = grad_clip

    print(f"resuming from {checkpoint_path}")
    print(f"run_dir = {run_dir}")

    tokenizer = Tokenizer.load(run_dir / "tokenizer.json")
    train_dataloader, validation_dataloader, actual_vocab_size = build_dataloaders(
        data_root, run_dir, config, tokenizer=tokenizer
    )
    assert actual_vocab_size == config["vocab_size"]
    save_config(run_dir, config)

    device = get_device()
    model = GPT.from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    config["param_count"] = count_parameters(model)
    save_config(run_dir, config)
    print(f"parameter count = {config['param_count']:,}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])

    total_steps = len(train_dataloader) * config["epochs"]
    lr_scheduler = build_lr_scheduler(optimizer, total_steps, config["warmup_steps"])

    if fresh_optimizer:
        print(
            f"using fresh optimizer and scheduler: lr={config['lr']} "
            f"grad_clip={config['grad_clip']}"
        )
    else:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    start_epoch = int(checkpoint["epoch"]) + 1
    global_step = int(checkpoint["step"])
    best_val_loss = checkpoint["best_val_loss"]
    epochs_without_improvement = checkpoint["epochs_without_improvement"]

    print(
        f"resume state: start_epoch={start_epoch + 1}, "
        f"global_step={global_step}, best_val_loss={best_val_loss:.4f}"
    )

    train(
        device,
        run_dir,
        train_dataloader,
        validation_dataloader,
        model,
        loss_fn,
        optimizer,
        lr_scheduler,
        config["epochs"],
        config,
        start_epoch=start_epoch,
        global_step=global_step,
        best_val_loss=best_val_loss,
        epochs_without_improvement=epochs_without_improvement,
    )

    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--fresh-optimizer", action="store_true")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--experiment",
        choices=sorted(experiment_configs().keys()),
        action="append",
        help="Experiment config to run. Can be passed more than once.",
    )
    parser.add_argument("--list-experiments", action="store_true")
    args = parser.parse_args()

    configs = experiment_configs()

    if args.list_experiments:
        for name, config in configs.items():
            print(
                f"{name}: {config['run_name']} "
                f"tokenizer={config['tokenizer_type']} "
                f"max_vocab={config['max_vocab_size']} "
                f"N={config['N']} d_model={config['d_model']} "
                f"d_ff={config['d_ff']} h={config['h']} "
                f"B={config['B']} L={config['L']} epochs={config['epochs']}"
            )
        return

    if args.resume is not None:
        run_dir = run_resume_training(
            args.resume,
            data_root=args.data_root,
            epochs=args.epochs,
            lr=args.lr,
            grad_clip=args.grad_clip,
            fresh_optimizer=args.fresh_optimizer,
        )
        print(f"Done: {run_dir}")
        return

    selected_experiments = args.experiment or ["baseline-char"]
    for experiment_name in selected_experiments:
        run_dir = run_training(dict(configs[experiment_name]), data_root=args.data_root)
        print(f"Done: {run_dir}")


if __name__ == "__main__":
    main()

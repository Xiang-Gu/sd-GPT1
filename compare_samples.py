import argparse
from pathlib import Path

import torch

from sample import (
    get_device,
    load_model_weights,
    read_validation_losses,
    sample_text,
)
from GPT import GPT
from Tokenizer import Tokenizer


RUNS_ROOT = Path("runs")
OUTPUT_PATH = RUNS_ROOT / "sample_comparison.txt"
DEFAULT_PROMPTS = [
    "Once upon a time, Little Daisy found out that",
    "Tom saw a small blue box under the bed.",
    "Lily wanted to help her friend, but",
]
SAMPLES_PER_PROMPT = 3
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.8
TOP_K = 50


def best_validation_loss(run_dir):
    losses = read_validation_losses(run_dir)
    if not losses:
        return None
    return min(losses, key=lambda row: row["val_loss"])


def candidate_runs(runs_root):
    runs = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "best.pt").exists():
            continue
        if not (run_dir / "tokenizer.json").exists():
            continue

        best = best_validation_loss(run_dir)
        if best is not None:
            runs.append((best["val_loss"], run_dir, best))

    return sorted(runs, key=lambda item: item[0])


def load_run(run_dir, device):
    tokenizer = Tokenizer.load(run_dir / "tokenizer.json")
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu")
    config = checkpoint["config"]
    model = GPT.from_config(config).to(device)
    load_model_weights(model, checkpoint, device)
    model.eval()
    return tokenizer, model, config


def write_samples_for_run(
    f,
    run_dir,
    best,
    tokenizer,
    model,
    config,
    device,
    prompts,
    samples_per_prompt,
    max_new_tokens,
    temperatures,
    top_ks,
):
    f.write("=" * 100 + "\n")
    f.write(f"{run_dir}\n")
    f.write(
        f"best validation loss {best['val_loss']:.4f} "
        f"at epoch {best['epoch'] + 1}, step {best['step']}\n"
    )
    f.write(
        f"tokenizer={config.get('tokenizer_type', 'char')} "
        f"max_vocab_size={config.get('max_vocab_size', '')} "
        f"N={config['N']} d_model={config['d_model']} d_ff={config['d_ff']} "
        f"h={config['h']} B={config['B']} L={config['L']} "
        f"dropout={config['dropout']} weight_tying={config['weight_tying']}\n\n"
    )

    for prompt in prompts:
        f.write("-" * 100 + "\n")
        f.write(f"PROMPT: {prompt!r}\n\n")
        for temperature in temperatures:
            for top_k in top_ks:
                f.write(f"temperature={temperature} top_k={top_k}\n\n")
                for idx in range(samples_per_prompt):
                    text = sample_text(
                        tokenizer,
                        model,
                        config,
                        prompt,
                        max_new_tokens,
                        temperature,
                        top_k,
                        device,
                    )
                    f.write(f"[sample {idx + 1}]\n{text}\n\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--samples-per-prompt", type=int, default=SAMPLES_PER_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, action="append", dest="temperatures")
    parser.add_argument("--top-k", type=int, action="append", dest="top_ks")
    parser.add_argument("--prompt", action="append", dest="prompts")
    args = parser.parse_args()

    prompts = args.prompts or DEFAULT_PROMPTS
    temperatures = args.temperatures or [0.6, TEMPERATURE, 1.0]
    top_ks = args.top_ks or [20, TOP_K]
    runs = candidate_runs(args.runs_root)
    if not runs:
        raise RuntimeError("No runs with best.pt, tokenizer.json, and validation metrics found.")

    device = get_device()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for _, run_dir, best in runs:
            tokenizer, model, config = load_run(run_dir, device)
            write_samples_for_run(
                f,
                run_dir,
                best,
                tokenizer,
                model,
                config,
                device,
                prompts,
                args.samples_per_prompt,
                args.max_new_tokens,
                temperatures,
                top_ks,
            )

    print(f"Wrote {args.output}")
    print(
        f"Compared {len(runs)} run(s), {len(prompts)} prompt(s), "
        f"{len(temperatures)} temperature(s), {len(top_ks)} top-k setting(s)."
    )


if __name__ == "__main__":
    main()

# Transformer

A small PyTorch GPT-style language model training project. The repo includes:

- A decoder-only Transformer model with causal self-attention.
- Character and simple BPE tokenizers.
- Non-overlapping next-token datasets.
- Training presets, checkpointing, resume support, metrics logging, and text sampling.
- Unit tests for the core model, tokenizer, dataset, and training helpers.

## Project Layout

```text
GPT.py                    Transformer/GPT model implementation
Tokenizer.py              Character tokenizer and simple BPE tokenizer
NextTokenDataset.py       Next-token prediction dataset
train.py                  Training entry point
sample.py                 Interactive sampling from trained checkpoints
compare_samples.py        Compare outputs across runs
analyze_runs.py           Summarize run metrics
plot_eval_loss.py         Plot evaluation loss
diagnose_*.py             Debugging/diagnostic scripts
tests/                    Unit tests
data/                     Local datasets, ignored by Git
runs/                     Training outputs/checkpoints, ignored by Git
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies. At minimum this project needs PyTorch. The TinyStories downloader also needs `datasets`.

```bash
pip install torch datasets
```

For NVIDIA GPU training, install the CUDA-enabled PyTorch build that matches your machine from the official PyTorch install selector.

## Data

Training expects a data directory with:

```text
data/tinystories/input_train.txt
data/tinystories/input_val.txt
```

You can generate the TinyStories subset with:

```bash
python data/tinystories/download_dataset.py
```

By default, `train.py` reads from `data/tinystories`. Use `--data-root` to train on another dataset with the same file names:

```bash
python train.py --data-root data/tinysp
```

## Training

List available experiment presets:

```bash
python train.py --list-experiments
```

Run the default baseline character model:

```bash
python train.py
```

Run one or more explicit experiments:

```bash
python train.py --experiment baseline-char
python train.py --experiment baseline-char --experiment big-char --experiment bpe-base
```

Each run writes a timestamped directory under `runs/` containing:

- `config.json`
- `tokenizer.json`
- `metrics.csv`
- `epoch_*.pt`
- `latest.pt`
- `best.pt`

Device selection is automatic:

```text
CUDA -> Apple MPS -> CPU
```

So on a CUDA-enabled desktop with a working NVIDIA PyTorch install, training should automatically use the NVIDIA GPU.

## Resume Training

Resume from a checkpoint:

```bash
python train.py --resume runs/<run_dir>/latest.pt
```

Override selected training settings while resuming:

```bash
python train.py --resume runs/<run_dir>/latest.pt --epochs 10
python train.py --resume runs/<run_dir>/latest.pt --lr 0.0001 --fresh-optimizer
python train.py --resume runs/<run_dir>/latest.pt --grad-clip 0.5
```

## Sampling

Sample interactively from the best run under `runs/`:

```bash
python sample.py
```

Sample from a specific run:

```bash
python sample.py --run-dir runs/<run_dir>
```

Sampling options:

```bash
python sample.py --temperature 0.8 --top-k 50 --max-new-tokens 400
```

At the prompt, type text to continue it. Type `q`, `quit`, or `exit` to stop.

## Tests

Run the unit tests:

```bash
./run_tests.sh
```

Or directly:

```bash
venv/bin/python -m unittest discover -s tests -v
```

## Git Notes

The repo intentionally ignores generated and local-heavy files:

- `venv/`
- `__pycache__/`
- `.vscode/`
- `data/`
- `runs/`
- checkpoint files such as `*.pt`

Keep source code, tests, and small scripts in Git. Keep datasets and trained checkpoints outside Git unless you intentionally publish them elsewhere.

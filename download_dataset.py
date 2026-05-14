from datasets import load_dataset
from pathlib import Path

out = Path("data/tinystories")
out.mkdir(parents=True, exist_ok=True)

# Start small for your laptop. Increase these later.
train_n = 100_000
val_n = 5_000

ds = load_dataset("roneneldan/TinyStories")

with open(out / "input_train.txt", "w", encoding="utf-8") as f:
    for row in ds["train"].select(range(train_n)):
        f.write(row["text"].strip())
        f.write("\n\n")

with open(out / "input_val.txt", "w", encoding="utf-8") as f:
    for row in ds["validation"].select(range(val_n)):
        f.write(row["text"].strip())
        f.write("\n\n")

print("wrote", out / "input_train.txt")
print("wrote", out / "input_val.txt")

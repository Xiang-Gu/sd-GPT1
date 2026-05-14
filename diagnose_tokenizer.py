import argparse
from collections import Counter
from pathlib import Path

from Tokenizer import BPETokenizer, Tokenizer, UNK_TOKEN


def read_text(path, max_chars=None):
    with open(path, "r", encoding="utf-8") as f:
        if max_chars is None:
            return f.read()
        return f.read(max_chars)


def percentile(sorted_values, q):
    if not sorted_values:
        return 0
    idx = round((len(sorted_values) - 1) * q)
    return sorted_values[idx]


def token_stats(name, text, tokenizer):
    tokens = tokenizer._tokenize_text(text)
    ids = tokenizer.encode(text)
    token_counts = Counter(tokens)
    unk_id = tokenizer.ttoi[UNK_TOKEN]
    unk_count = sum(1 for token_id in ids if token_id == unk_id)
    token_lengths = sorted(len(token) for token in tokens)

    return {
        "name": name,
        "text": text,
        "tokens": tokens,
        "ids": ids,
        "token_counts": token_counts,
        "chars": len(text),
        "num_tokens": len(tokens),
        "unique_tokens": len(token_counts),
        "unique_ids": len(set(ids)),
        "unk_count": unk_count,
        "unk_rate": unk_count / max(len(ids), 1),
        "chars_per_token": len(text) / max(len(tokens), 1),
        "avg_token_len": sum(token_lengths) / max(len(token_lengths), 1),
        "p50_token_len": percentile(token_lengths, 0.50),
        "p90_token_len": percentile(token_lengths, 0.90),
        "p99_token_len": percentile(token_lengths, 0.99),
        "max_token_len": token_lengths[-1] if token_lengths else 0,
    }


def print_basic_stats(stats):
    print(f"{stats['name']}:")
    print(f"  chars: {stats['chars']:,}")
    print(f"  tokens: {stats['num_tokens']:,}")
    print(f"  chars/token: {stats['chars_per_token']:.2f}")
    print(f"  unique token strings: {stats['unique_tokens']:,}")
    print(f"  unique token ids: {stats['unique_ids']:,}")
    print(f"  <UNK>: {stats['unk_count']:,} ({100 * stats['unk_rate']:.4f}%)")
    print(
        "  token length: "
        f"avg={stats['avg_token_len']:.2f} "
        f"p50={stats['p50_token_len']} "
        f"p90={stats['p90_token_len']} "
        f"p99={stats['p99_token_len']} "
        f"max={stats['max_token_len']}"
    )


def printable_token(token):
    return repr(token).replace("\n", "\\n")


def print_top_tokens(stats, limit):
    print(f"\nTop {limit} tokens in {stats['name']}:")
    for token, count in stats["token_counts"].most_common(limit):
        print(f"  {count:>9,}  len={len(token):<3}  {printable_token(token)}")


def print_train_val_drift(train_stats, val_stats, limit):
    train_counts = train_stats["token_counts"]
    val_counts = val_stats["token_counts"]
    train_total = max(train_stats["num_tokens"], 1)
    val_total = max(val_stats["num_tokens"], 1)

    val_only = [
        (token, count)
        for token, count in val_counts.items()
        if train_counts.get(token, 0) == 0
    ]
    val_only.sort(key=lambda item: item[1], reverse=True)

    print(f"\nValidation tokens absent from compared train text: {len(val_only):,}")
    for token, count in val_only[:limit]:
        print(f"  {count:>9,}  len={len(token):<3}  {printable_token(token)}")

    drift = []
    for token, val_count in val_counts.items():
        train_rate = train_counts.get(token, 0) / train_total
        val_rate = val_count / val_total
        ratio = val_rate / max(train_rate, 1 / train_total)
        drift.append((ratio, val_rate, train_rate, token, val_count, train_counts.get(token, 0)))
    drift.sort(reverse=True)

    print(f"\nTokens most overrepresented in validation vs compared train text:")
    for ratio, val_rate, train_rate, token, val_count, train_count in drift[:limit]:
        print(
            f"  ratio={ratio:>8.1f} "
            f"val={val_count:>8,} ({100 * val_rate:.4f}%) "
            f"train={train_count:>8,} ({100 * train_rate:.4f}%) "
            f"len={len(token):<3} {printable_token(token)}"
        )


def print_bpe_summary(tokenizer):
    if not isinstance(tokenizer, BPETokenizer):
        return

    merge_lengths = sorted(len("".join(pair)) for pair in tokenizer.merges)
    print("\nBPE merges:")
    print(f"  merges: {len(tokenizer.merges):,}")
    if merge_lengths:
        print(
            "  merged pair length: "
            f"p50={percentile(merge_lengths, 0.50)} "
            f"p90={percentile(merge_lengths, 0.90)} "
            f"p99={percentile(merge_lengths, 0.99)} "
            f"max={merge_lengths[-1]}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compare train/validation tokenization for a saved tokenizer."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/tinystories"))
    parser.add_argument(
        "--max-chars",
        type=int,
        help="Read at most this many characters from both train and validation.",
    )
    parser.add_argument(
        "--train-max-chars",
        type=int,
        help="Read at most this many training characters. Overrides --max-chars for train.",
    )
    parser.add_argument(
        "--val-max-chars",
        type=int,
        help="Read at most this many validation characters. Overrides --max-chars for validation.",
    )
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    tokenizer = Tokenizer.load(args.run_dir / "tokenizer.json")
    train_max_chars = args.train_max_chars
    val_max_chars = args.val_max_chars
    if args.max_chars is not None:
        train_max_chars = train_max_chars or args.max_chars
        val_max_chars = val_max_chars or args.max_chars

    train_text = read_text(args.data_root / "input_train.txt", train_max_chars)
    val_text = read_text(args.data_root / "input_val.txt", val_max_chars)

    print(f"run: {args.run_dir}")
    print(f"tokenizer: {type(tokenizer).__name__}, vocab size: {tokenizer.vocab_size():,}")
    print_bpe_summary(tokenizer)
    print()

    train_stats = token_stats("train", train_text, tokenizer)
    val_stats = token_stats("validation", val_text, tokenizer)
    print_basic_stats(train_stats)
    print_basic_stats(val_stats)
    print_top_tokens(train_stats, args.top)
    print_top_tokens(val_stats, args.top)
    print_train_val_drift(train_stats, val_stats, args.top)


if __name__ == "__main__":
    main()

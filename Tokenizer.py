import json
from collections import Counter
from pathlib import Path
import re

UNK_TOKEN = "<UNK>"
EOS_TOKEN = "<EOS>"
SPECIAL_TOKENS = [UNK_TOKEN, EOS_TOKEN]


class Tokenizer:
    def __init__(self, max_vocab_size=5000, token_to_id=None, id_to_token=None):
        self.max_vocab_size = max_vocab_size
        self.ttoi = token_to_id or {UNK_TOKEN: 0, EOS_TOKEN: 1}
        self.itot = id_to_token or {0: UNK_TOKEN, 1: EOS_TOKEN}
        self.is_fitted = token_to_id is not None and id_to_token is not None

    @staticmethod
    def _tokenize_text(text):
        """
        Simple per-character tokenizer.

        Every character is a token, including spaces and newlines.
        """
        return [ch for ch in text]

    @staticmethod
    def detokenize(tokens):
        return "".join(tokens)

    def fit(self, text):
        self._build_vocab(self._tokenize_text(text))

    def _build_vocab(self, tokens):
        if self.is_fitted:
            raise RuntimeError("Vocabulary has already been built.")

        next_id = len(SPECIAL_TOKENS)
        num_regular_tokens = self.max_vocab_size - next_id

        for token, _ in Counter(tokens).most_common(num_regular_tokens):
            if token not in self.ttoi:
                self.ttoi[token] = next_id
                self.itot[next_id] = token
                next_id += 1

        self.is_fitted = True

    def vocab_size(self):
        return len(self.ttoi)

    def encode(self, text):
        if not self.is_fitted:
            raise RuntimeError("Tokenizer must be fit before calling encode.")
        return self._tokens_to_ids(self._tokenize_text(text))

    def _tokens_to_ids(self, tokens):
        unk_id = self.ttoi[UNK_TOKEN]
        return [self.ttoi.get(token, unk_id) for token in tokens]

    def decode(self, ids):
        return self.detokenize(self._ids_to_tokens(ids))

    def _ids_to_tokens(self, ids):
        return [self.itot.get(int(id), UNK_TOKEN) for id in ids]

    def save(self, path):
        data = {
            "tokenizer_type": "char",
            "max_vocab_size": self.max_vocab_size,
            "token_to_id": self.ttoi,
            "id_to_token": self.itot,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("tokenizer_type", "char") == "bpe":
            return BPETokenizer.from_data(data)

        tokenizer = cls(max_vocab_size=data["max_vocab_size"])
        tokenizer.ttoi = {token: int(id) for token, id in data["token_to_id"].items()}
        tokenizer.itot = {int(id): token for id, token in data["id_to_token"].items()}
        tokenizer.is_fitted = True
        return tokenizer


class BPETokenizer(Tokenizer):
    def __init__(
        self,
        max_vocab_size=5000,
        token_to_id=None,
        id_to_token=None,
        merges=None,
    ):
        super().__init__(max_vocab_size, token_to_id, id_to_token)
        self.merges = [tuple(merge) for merge in merges] if merges is not None else []

    @staticmethod
    def _initial_chunks(text):
        return re.findall(r"\s+|\S+", text)

    def _tokenize_text(self, text):
        if not self.is_fitted:
            raise RuntimeError("Tokenizer must be fit before tokenizing text.")
        tokens = []
        for chunk in self._initial_chunks(text):
            tokens.extend(self._apply_merges(tuple(chunk)))
        return tokens

    def fit(self, text):
        chunks = self._initial_chunks(text)
        if self.is_fitted:
            raise RuntimeError("Vocabulary has already been built.")

        chunk_counts = Counter(tuple(chunk) for chunk in chunks)
        vocab_tokens = set(SPECIAL_TOKENS)
        for chunk in chunk_counts:
            vocab_tokens.update(chunk)

        while len(vocab_tokens) < self.max_vocab_size:
            pair_counts = Counter()
            for symbols, count in chunk_counts.items():
                for pair in zip(symbols, symbols[1:]):
                    pair_counts[pair] += count

            if not pair_counts:
                break

            best_pair = None
            merged_token = None
            for pair, _ in pair_counts.most_common():
                candidate = "".join(pair)
                if candidate not in vocab_tokens:
                    best_pair = pair
                    merged_token = candidate
                    break

            if best_pair is None:
                break

            next_chunk_counts = Counter()
            for symbols, count in chunk_counts.items():
                merged_symbols = self._merge_pair(symbols, best_pair, merged_token)
                next_chunk_counts[merged_symbols] += count
            chunk_counts = next_chunk_counts
            self.merges.append(best_pair)
            vocab_tokens.add(merged_token)

        self.ttoi = {UNK_TOKEN: 0, EOS_TOKEN: 1}
        self.itot = {0: UNK_TOKEN, 1: EOS_TOKEN}
        next_id = len(SPECIAL_TOKENS)

        for token in sorted(vocab_tokens - set(SPECIAL_TOKENS)):
            if next_id >= self.max_vocab_size:
                break
            self.ttoi[token] = next_id
            self.itot[next_id] = token
            next_id += 1

        self.is_fitted = True

    @staticmethod
    def _merge_pair(symbols, pair, merged_token):
        out = []
        idx = 0
        while idx < len(symbols):
            if idx < len(symbols) - 1 and (symbols[idx], symbols[idx + 1]) == pair:
                out.append(merged_token)
                idx += 2
            else:
                out.append(symbols[idx])
                idx += 1
        return tuple(out)

    def _apply_merges(self, symbols):
        pairs = set(zip(symbols, symbols[1:]))
        for pair in self.merges:
            if pair not in pairs:
                continue
            merged_token = "".join(pair)
            symbols = self._merge_pair(symbols, pair, merged_token)
            pairs = set(zip(symbols, symbols[1:]))
        return list(symbols)

    def save(self, path):
        data = {
            "tokenizer_type": "bpe",
            "max_vocab_size": self.max_vocab_size,
            "token_to_id": self.ttoi,
            "id_to_token": self.itot,
            "merges": self.merges,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def from_data(cls, data):
        tokenizer = cls(max_vocab_size=data["max_vocab_size"], merges=data["merges"])
        tokenizer.ttoi = {token: int(id) for token, id in data["token_to_id"].items()}
        tokenizer.itot = {int(id): token for id, token in data["id_to_token"].items()}
        tokenizer.is_fitted = True
        return tokenizer

import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from Tokenizer import BPETokenizer, EOS_TOKEN, UNK_TOKEN, Tokenizer


class TestTokenizer(unittest.TestCase):
    def test_private_char_tokenizer_preserves_characters_spaces_and_newlines(self):
        text = '"I don\'t like cats", she said.\nIs that right?'

        tokens = Tokenizer._tokenize_text(text)

        self.assertEqual(tokens, list(text))
        self.assertIn(" ", tokens)
        self.assertIn("\n", tokens)

    def test_fit_keeps_most_common_characters(self):
        text = "rare-firstabcabdae"
        tokenizer = Tokenizer(max_vocab_size=5)

        tokenizer.fit(text)

        expected_common_tokens = [
            token
            for token, _ in Counter(text).most_common(
                tokenizer.max_vocab_size - len([UNK_TOKEN, EOS_TOKEN])
            )
        ]
        actual_common_tokens = [
            tokenizer.itot[idx]
            for idx in range(len([UNK_TOKEN, EOS_TOKEN]), tokenizer.vocab_size())
        ]

        self.assertEqual(actual_common_tokens, expected_common_tokens)
        self.assertIn("a", tokenizer.ttoi)
        self.assertIn("r", tokenizer.ttoi)

    def test_fit_can_only_be_called_once(self):
        tokenizer = Tokenizer(max_vocab_size=5)
        tokenizer.fit("ab")

        with self.assertRaises(RuntimeError):
            tokenizer.fit("c")

    def test_encode_requires_fit(self):
        tokenizer = Tokenizer(max_vocab_size=5)

        with self.assertRaises(RuntimeError):
            tokenizer.encode("hello")

    def test_encode_unknown_character_uses_unk_id(self):
        tokenizer = Tokenizer(max_vocab_size=5)
        tokenizer.fit("hello")

        ids = tokenizer.encode("hello?")

        self.assertEqual(ids[-1], tokenizer.ttoi[UNK_TOKEN])

    def test_decode_unknown_id_uses_unk_token(self):
        tokenizer = Tokenizer(max_vocab_size=5)
        tokenizer.fit("hello")

        self.assertEqual(tokenizer.decode([tokenizer.ttoi["h"], 999]), "h" + UNK_TOKEN)

    def test_detokenize_joins_tokens(self):
        text = Tokenizer.detokenize(list("I like cats.\n"))

        self.assertEqual(text, "I like cats.\n")

    def test_save_and_load_preserves_vocab(self):
        tokenizer = Tokenizer(max_vocab_size=10)
        tokenizer.fit("hello world hello")

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tokenizer.json"
            tokenizer.save(path)
            loaded = Tokenizer.load(path)

        self.assertEqual(loaded.max_vocab_size, tokenizer.max_vocab_size)
        self.assertEqual(loaded.ttoi, tokenizer.ttoi)
        self.assertEqual(loaded.itot, tokenizer.itot)
        self.assertEqual(loaded.encode("hello?"), tokenizer.encode("hello?"))


class TestBPETokenizer(unittest.TestCase):
    def test_bpe_tokenizer_round_trips_text(self):
        text = "hello hello helper\nhello helper"
        tokenizer = BPETokenizer(max_vocab_size=20)

        tokenizer.fit(text)
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)

        self.assertEqual(decoded, text)
        self.assertGreater(tokenizer.vocab_size(), len(set(text)))
        self.assertTrue(tokenizer.merges)

    def test_bpe_encode_requires_fit(self):
        tokenizer = BPETokenizer(max_vocab_size=20)

        with self.assertRaises(RuntimeError):
            tokenizer.encode("hello")

    def test_bpe_save_and_load_preserves_merges(self):
        text = "Daisy saw Daisy play."
        tokenizer = BPETokenizer(max_vocab_size=16)
        tokenizer.fit(text)

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tokenizer.json"
            tokenizer.save(path)
            loaded = Tokenizer.load(path)

        self.assertIsInstance(loaded, BPETokenizer)
        self.assertEqual(loaded.merges, tokenizer.merges)
        self.assertEqual(loaded.encode(text), tokenizer.encode(text))
        self.assertEqual(loaded.decode(loaded.encode(text)), text)


if __name__ == "__main__":
    unittest.main()

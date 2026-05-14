import unittest
import io
from contextlib import redirect_stdout
from tempfile import TemporaryDirectory
from pathlib import Path

import torch
from torch import nn, optim

from train import build_tokenizer_and_encode, train_step


class BadGradModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, tokens):
        B, L = tokens.shape
        logits = torch.zeros((B, L, 3), device=tokens.device)
        logits[:, :, 0] = self.logit
        return logits


class TestTrainStep(unittest.TestCase):
    def test_train_step_skips_nonfinite_gradient(self):
        model = BadGradModel()
        model.logit.register_hook(lambda grad: torch.full_like(grad, float("nan")))
        optimizer = optim.SGD(model.parameters(), lr=1.0)
        initial_value = model.logit.detach().clone()

        loss, grad_norm_before, grad_norm_after, did_step, bad_grad_name = train_step(
            model,
            torch.zeros((2, 4), dtype=torch.long),
            torch.zeros((2, 4), dtype=torch.long),
            nn.CrossEntropyLoss(),
            optimizer,
            grad_clip=1.0,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(torch.isfinite(grad_norm_before))
        self.assertFalse(torch.isfinite(grad_norm_after))
        self.assertFalse(did_step)
        self.assertEqual(bad_grad_name, "logit")
        torch.testing.assert_close(model.logit, initial_value)


class TestTokenizationDuringTraining(unittest.TestCase):
    def test_bpe_training_ids_use_fitted_tokenization(self):
        text = "hello hello helper\nhello helper hello\n"
        config = {"tokenizer_type": "bpe", "max_vocab_size": 20}

        with TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "data"
            run_dir = Path(tmpdir) / "run"
            data_root.mkdir()
            run_dir.mkdir()
            (data_root / "input_train.txt").write_text(text, encoding="utf-8")

            with redirect_stdout(io.StringIO()):
                tokenizer, train_ids = build_tokenizer_and_encode(
                    data_root, run_dir, config, "input_train.txt"
                )

        expected_ids = tokenizer.encode(text)
        self.assertEqual(train_ids, expected_ids)


if __name__ == "__main__":
    unittest.main()

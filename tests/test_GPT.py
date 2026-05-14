import unittest

import torch

from GPT import (
    AttnFFNBlock,
    FFN,
    GPT,
    LayerNorm,
    MHA,
    PositionalEmbeddings,
)

class TestPositionalEmbeddings(unittest.TestCase):
    def test_positional_embedding_shape_and_first_position(self):
        x = torch.zeros((2, 3, 8))
        pe = PositionalEmbeddings.forward(x)

        self.assertEqual(pe.shape, (3, 8))
        torch.testing.assert_close(
            pe[0],
            torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=x.dtype),
        )

    def test_positional_embedding_supports_odd_d_model(self):
        x = torch.zeros((2, 4, 5))
        pe = PositionalEmbeddings.forward(x)

        self.assertEqual(pe.shape, (4, 5))
        self.assertTrue(torch.isfinite(pe).all())
        self.assertEqual(pe.device, x.device)
        self.assertEqual(pe.dtype, x.dtype)


class TestLayerNorm(unittest.TestCase):
    def test_layer_norm_normalizes_last_axis(self):
        x = torch.randn((3, 4, 8)) * 2.5 + 14
        out = LayerNorm(d_model=8)(x)

        torch.testing.assert_close(
            out.mean(dim=-1), torch.zeros((3, 4)), atol=1e-5, rtol=0
        )
        torch.testing.assert_close(
            out.var(dim=-1, unbiased=False),
            torch.ones((3, 4)),
            atol=1e-4,
            rtol=0,
        )


class TestMHA(unittest.TestCase):
    def test_forward_shape(self):
        x = torch.randn((2, 4, 8))
        out = MHA(d_model=8, h=2)(x)

        self.assertEqual(out.shape, (2, 4, 8))
        self.assertTrue(torch.isfinite(out).all())

    def test_causal_attention_does_not_use_future_positions(self):
        mha = MHA(d_model=1, h=1, dropout=0.0)

        Q = torch.ones((1, 1, 4, 1))
        K = torch.ones((1, 1, 4, 1))
        V = torch.tensor([[[[10.0], [20.0], [30.0], [40.0]]]])

        out = mha.attn(Q, K, V)

        expected = torch.tensor([[[[10.0], [15.0], [20.0], [25.0]]]])
        torch.testing.assert_close(out, expected)

    def test_attention_backward_has_finite_gradients(self):
        mha = MHA(d_model=8, h=2, dropout=0.0)
        x = torch.randn((2, 4, 8), requires_grad=True)

        out = mha(x)
        out.square().mean().backward()

        self.assertTrue(torch.isfinite(x.grad).all())
        for param in mha.parameters():
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())

    def test_split_and_combine_heads_round_trip(self):
        mha = MHA(d_model=16, h=4)
        x = torch.randn((2, 5, 16))

        Q, K, V = mha.split_heads(x, x, x)
        out = mha.combine_heads(Q)

        self.assertEqual(Q.shape, (2, 4, 5, 4))
        torch.testing.assert_close(out, x)


class TestFFN(unittest.TestCase):
    def test_ffn_shape(self):
        x = torch.randn((2, 4, 8))
        out = FFN(d_model=8, d_ff=32)(x)

        self.assertEqual(out.shape, (2, 4, 8))
        self.assertTrue(torch.isfinite(out).all())


class TestBlockAndGPT(unittest.TestCase):
    def test_block_shape(self):
        x = torch.randn((2, 4, 8))
        out = AttnFFNBlock(d_model=8, h=2, d_ff=32)(x)

        self.assertEqual(out.shape, (2, 4, 8))
        self.assertTrue(torch.isfinite(out).all())

    def test_gpt_forward_returns_logits(self):
        tokens = torch.tensor([[0, 1, 2, 5], [5, 2, 1, 0]], dtype=torch.long)
        model = GPT(vocab_size=100, N=1, d_model=8, d_ff=32, h=2)

        logits = model(tokens)

        self.assertEqual(logits.shape, (2, 4, 100))
        self.assertTrue(torch.isfinite(logits).all())

    def test_gpt_sample_next_token_returns_token_ids(self):
        tokens = torch.tensor([[0, 1, 2, 5], [5, 2, 1, 0]], dtype=torch.long)
        model = GPT(vocab_size=100, N=1, d_model=8, d_ff=32, h=2)

        next_tokens = model.sample_next_token_id(tokens)

        self.assertEqual(next_tokens.shape, (2,))
        self.assertEqual(next_tokens.dtype, torch.long)
        self.assertTrue(((0 <= next_tokens) & (next_tokens < 100)).all())

    def test_gpt_sample_next_token_supports_top_k_larger_than_vocab(self):
        tokens = torch.tensor([[0, 1, 2, 5]], dtype=torch.long)
        model = GPT(vocab_size=10, N=1, d_model=8, d_ff=32, h=2)

        next_tokens = model.sample_next_token_id(tokens, top_k=50)

        self.assertEqual(next_tokens.shape, (1,))
        self.assertTrue(0 <= next_tokens.item() < 10)

    def test_gpt_sample_next_token_rejects_non_positive_temperature(self):
        tokens = torch.tensor([[0, 1, 2, 5]], dtype=torch.long)
        model = GPT(vocab_size=10, N=1, d_model=8, d_ff=32, h=2)

        with self.assertRaises(ValueError):
            model.sample_next_token_id(tokens, temperature=0)


if __name__ == "__main__":
    unittest.main()

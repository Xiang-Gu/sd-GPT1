import unittest

import torch
from torch.utils.data import DataLoader

from NextTokenDataset import NextTokenDataset


class TestNextTokenDataset(unittest.TestCase):
    def test_length(self):
        dataset = NextTokenDataset(list(range(10)), L=4)

        self.assertEqual(len(dataset), 2)

    def test_getitem_returns_shifted_input_and_target(self):
        dataset = NextTokenDataset(list(range(10)), L=4)

        x, y = dataset[0]

        self.assertTrue(torch.equal(x, torch.tensor([0, 1, 2, 3], dtype=torch.long)))
        self.assertTrue(torch.equal(y, torch.tensor([1, 2, 3, 4], dtype=torch.long)))

    def test_getitem_uses_non_overlapping_window(self):
        dataset = NextTokenDataset(list(range(10)), L=4)

        x, y = dataset[1]

        self.assertTrue(torch.equal(x, torch.tensor([4, 5, 6, 7], dtype=torch.long)))
        self.assertTrue(torch.equal(y, torch.tensor([5, 6, 7, 8], dtype=torch.long)))

    def test_getitem_returns_long_tensors(self):
        dataset = NextTokenDataset(list(range(10)), L=4)

        x, y = dataset[0]

        self.assertEqual(x.dtype, torch.long)
        self.assertEqual(y.dtype, torch.long)

    def test_dataloader_batches_to_expected_shape(self):
        dataset = NextTokenDataset(list(range(40)), L=4)
        loader = DataLoader(dataset, batch_size=3, shuffle=False)

        x, y = next(iter(loader))

        self.assertEqual(x.shape, (3, 4))
        self.assertEqual(y.shape, (3, 4))
        self.assertEqual(x.dtype, torch.long)
        self.assertEqual(y.dtype, torch.long)


if __name__ == "__main__":
    unittest.main()

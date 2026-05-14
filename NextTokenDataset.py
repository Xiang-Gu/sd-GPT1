from torch.utils.data import Dataset
import torch


class NextTokenDataset(Dataset):
    def __init__(self, ids, L):
        """
        Non-overlapping next-token dataset.

        For ids=[0, 2, 1, 4, 3, 5, 6, 7, 8] and L=4:
        x=[0, 2, 1, 4], y=[2, 1, 4, 3]
        x=[3, 5, 6, 7], y=[5, 6, 7, 8]
        """
        self.ids = torch.as_tensor(ids, dtype=torch.long)
        self.L = L

    def __getitem__(self, index):
        start = index * self.L
        x = self.ids[start : start + self.L]
        y = self.ids[start + 1 : start + self.L + 1]
        return x, y

    def __len__(self):
        return max(0, (len(self.ids) - 1) // self.L)


if __name__ == "__main__":
    N = 10
    L = 1
    ids = list(range(N))
    ds = NextTokenDataset(ids, L)
    assert len(ds) == N - L

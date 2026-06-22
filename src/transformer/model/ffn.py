import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float, expansion: int = 4) -> None:
        super().__init__()
        hidden = d_model * expansion
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

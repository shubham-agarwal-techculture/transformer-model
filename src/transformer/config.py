from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    block_size: int = 256
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")


@dataclass
class TrainConfig:
    lr: float = 3e-4
    batch_size: int = 32
    max_steps: int = 1000
    eval_interval: int = 100
    device: str = "cpu"

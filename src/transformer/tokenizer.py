from __future__ import annotations

from typing import Iterable


class CharTokenizer:
    """Character-level tokenizer with a fixed vocabulary built from a corpus."""

    def __init__(self, chars: Iterable[str] | None = None) -> None:
        self._char_to_id: dict[str, int] = {}
        self._id_to_char: dict[int, str] = {}
        if chars is not None:
            self.build_vocab("".join(chars) if not isinstance(chars, str) else chars)

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        tokenizer = cls()
        tokenizer.build_vocab(text)
        return tokenizer

    def build_vocab(self, text: str) -> None:
        unique_chars = sorted(set(text))
        self._char_to_id = {ch: i for i, ch in enumerate(unique_chars)}
        self._id_to_char = {i: ch for ch, i in self._char_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self._char_to_id)

    def encode(self, text: str) -> list[int]:
        unknown = self._char_to_id.get("?", None)
        return [self._char_to_id.get(ch, unknown) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._id_to_char[i] for i in ids)

    def to_dict(self) -> dict:
        return {
            "char_to_id": self._char_to_id,
            "id_to_char": {str(k): v for k, v in self._id_to_char.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> CharTokenizer:
        tokenizer = cls()
        tokenizer._char_to_id = dict(data["char_to_id"])
        tokenizer._id_to_char = {int(k): v for k, v in data["id_to_char"].items()}
        return tokenizer

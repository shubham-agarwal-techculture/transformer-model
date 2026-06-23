from __future__ import annotations

from collections import Counter

from tqdm import tqdm

UNK_TOKEN = "<unk>"


def _count_pairs(tokens: list[str]) -> Counter[tuple[str, str]]:
    pairs: Counter[tuple[str, str]] = Counter()
    for i in range(len(tokens) - 1):
        pairs[(tokens[i], tokens[i + 1])] += 1
    return pairs


def _apply_merge(tokens: list[str], pair: tuple[str, str], merged: str) -> list[str]:
    if not tokens:
        return tokens
    first, second = pair
    merged_tokens: list[str] = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == first and tokens[i + 1] == second:
            merged_tokens.append(merged)
            i += 2
        else:
            merged_tokens.append(tokens[i])
            i += 1
    return merged_tokens


def _best_pair(pairs: Counter[tuple[str, str]]) -> tuple[str, str] | None:
    if not pairs:
        return None
    best_count = max(pairs.values())
    candidates = [pair for pair, count in pairs.items() if count == best_count]
    return min(candidates)


class BPETokenizer:
    """Byte Pair Encoding tokenizer trained from a text corpus."""

    def __init__(
        self,
        token_to_id: dict[str, int],
        merges: list[tuple[str, str]],
        target_vocab_size: int,
    ) -> None:
        self._token_to_id = token_to_id
        self._id_to_token = {idx: token for token, idx in token_to_id.items()}
        self._merges = merges
        self._target_vocab_size = target_vocab_size
        self._unk_id = token_to_id[UNK_TOKEN]

    @classmethod
    def from_text(
        cls,
        text: str,
        vocab_size: int = 4096,
        *,
        show_progress: bool = True,
    ) -> BPETokenizer:
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")

        token_to_id: dict[str, int] = {UNK_TOKEN: 0}
        for ch in sorted(set(text)):
            token_to_id[ch] = len(token_to_id)

        tokens = list(text)
        merges: list[tuple[str, str]] = []
        merge_iter = range(len(token_to_id), vocab_size)
        if show_progress:
            merge_iter = tqdm(merge_iter, desc="Training BPE", unit="merge")

        for _ in merge_iter:
            pairs = _count_pairs(tokens)
            pair = _best_pair(pairs)
            if pair is None:
                break

            merged = pair[0] + pair[1]
            merges.append(pair)
            token_to_id[merged] = len(token_to_id)
            tokens = _apply_merge(tokens, pair, merged)

        return cls(token_to_id, merges, target_vocab_size=vocab_size)

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)

    @property
    def merge_count(self) -> int:
        return len(self._merges)

    @property
    def target_vocab_size(self) -> int:
        return self._target_vocab_size

    def _tokenize(self, text: str) -> list[str]:
        tokens = [ch if ch in self._token_to_id else UNK_TOKEN for ch in text]
        for pair in self._merges:
            tokens = _apply_merge(tokens, pair, pair[0] + pair[1])
        return tokens

    def encode(self, text: str) -> list[int]:
        return [self._token_to_id[token] for token in self._tokenize(text)]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._id_to_token[i] for i in ids)

    def to_dict(self) -> dict:
        return {
            "type": "bpe",
            "target_vocab_size": self._target_vocab_size,
            "merges": [list(pair) for pair in self._merges],
            "token_to_id": self._token_to_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BPETokenizer:
        merges = [tuple(pair) for pair in data["merges"]]
        token_to_id = dict(data["token_to_id"])
        return cls(
            token_to_id=token_to_id,
            merges=merges,
            target_vocab_size=data.get("target_vocab_size", len(token_to_id)),
        )


def load_tokenizer(data: dict) -> BPETokenizer:
    if "char_to_id" in data:
        raise ValueError(
            "Checkpoint uses the legacy character tokenizer. "
            "Retrain with BPE: python scripts/train_book.py --data <text>"
        )
    if data.get("type") != "bpe" and "merges" not in data:
        raise ValueError("Unsupported tokenizer checkpoint format")
    return BPETokenizer.from_dict(data)

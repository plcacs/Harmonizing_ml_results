from __future__ import annotations
import pyperf
import collections
import re
from pathlib import Path
from typing import List, Dict, Tuple

class SimpleBytePairEncoding:
    def __init__(self, *, pat_str: str, mergeable_ranks: Dict[bytes, int]) -> None:
        self.pat_str: str = pat_str
        self.mergeable_ranks: Dict[bytes, int] = mergeable_ranks
        self._decoder: Dict[int, bytes] = {token: token_bytes for (token_bytes, token) in mergeable_ranks.items()}
        self._pat = re.compile(pat_str)

    def encode(self, text: str) -> List[int]:
        words: List[str] = self._pat.findall(text)
        tokens: List[int] = []
        for word in words:
            word_bytes: bytes = word.encode('utf-8')
            word_tokens: List[int] = bpe_encode(self.mergeable_ranks, word_bytes)
            tokens.extend(word_tokens)
        return tokens

    def decode_bytes(self, tokens: List[int]) -> bytes:
        return b''.join(self._decoder[token] for token in tokens)

    def decode(self, tokens: List[int]) -> str:
        return self.decode_bytes(tokens).decode('utf-8', errors='replace')

    @staticmethod
    def train(training_data: str, vocab_size: int, pat_str: str) -> SimpleBytePairEncoding:
        mergeable_ranks: Dict[bytes, int] = bpe_train(data=training_data, vocab_size=vocab_size, pat_str=pat_str)
        return SimpleBytePairEncoding(pat_str=pat_str, mergeable_ranks=mergeable_ranks)

def bpe_encode(mergeable_ranks: Dict[bytes, int], input: bytes) -> List[int]:
    parts: List[bytes] = [bytes([b]) for b in input]
    while True:
        min_idx: int | None = None
        min_rank: int | None = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank: int | None = mergeable_ranks.get(pair[0] + pair[1])
            if (rank is not None) and ((min_rank is None) or (rank < min_rank)):
                min_idx = i
                min_rank = rank
        if min_rank is None:
            break
        assert min_idx is not None
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    tokens: List[int] = [mergeable_ranks[part] for part in parts]
    return tokens

def bpe_train(data: str, vocab_size: int, pat_str: str) -> Dict[bytes, int]:
    if vocab_size < (2 ** 8):
        raise ValueError('vocab_size must be at least 256, so we can encode all bytes')
    ranks: Dict[bytes, int] = {}
    for i in range(2 ** 8):
        ranks[bytes([i])] = i
    words: List[List[bytes]] = [[bytes([b]) for b in word.encode('utf-8')] for word in re.findall(pat_str, data)]
    while len(ranks) < vocab_size:
        stats: collections.Counter[Tuple[bytes, bytes]] = collections.Counter()
        for piece in words:
            for pair in zip(piece[:-1], piece[1:]):
                stats[pair] += 1
        most_common_pair: Tuple[bytes, bytes] = max(stats, key=lambda x: stats[x])
        token_bytes: bytes = most_common_pair[0] + most_common_pair[1]
        token: int = len(ranks)
        ranks[token_bytes] = token
        new_words: List[List[bytes]] = []
        for word in words:
            new_word: List[bytes] = []
            i: int = 0
            while i < len(word) - 1:
                if (word[i], word[i + 1]) == most_common_pair:
                    new_word.append(token_bytes)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            if i == len(word) - 1:
                new_word.append(word[i])
            new_words.append(new_word)
        words = new_words
    return ranks

def train(data: str) -> None:
    pattern: str = "'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?\\d+| ?[^\\sa-zA-Z\\d]+|\\s+(?!\\S)|\\s+"
    enc: SimpleBytePairEncoding = SimpleBytePairEncoding.train(data, vocab_size=1024, pat_str=pattern)
    tokens: List[int] = enc.encode('hello world')
    assert enc.decode(tokens) == 'hello world'
    enc.encode(data)

def bench_bpe_tokeniser(loops: int) -> float:
    DATA: Path = (Path(__file__).parent / 'data') / 'frankenstein_intro.txt'
    with open(DATA, 'r', encoding='utf8') as f:
        data: str = f.read()
    range_it = range(loops)
    t0: float = pyperf.perf_counter()
    for _ in range_it:
        train(data)
    return pyperf.perf_counter() - t0

if __name__ == '__main__':
    runner = pyperf.Runner()
    runner.metadata['description'] = 'Benchmark a BPE tokeniser'
    runner.bench_time_func('bpe_tokeniser', bench_bpe_tokeniser)
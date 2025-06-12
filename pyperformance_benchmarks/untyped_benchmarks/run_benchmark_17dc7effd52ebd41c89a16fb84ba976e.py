
'\nBenchmark a BPE tokeniser.\n\nAuthor: Shantanu Jain\n\nBased on code from tiktoken.\nhttps://github.com/openai/tiktoken\n'
from __future__ import annotations
import pyperf
import collections
import re
from pathlib import Path

class SimpleBytePairEncoding():

    def __init__(self, *, pat_str, mergeable_ranks):
        self.pat_str = pat_str
        self.mergeable_ranks = mergeable_ranks
        self._decoder = {token: token_bytes for (token_bytes, token) in mergeable_ranks.items()}
        self._pat = re.compile(pat_str)

    def encode(self, text):
        words = self._pat.findall(text)
        tokens = []
        for word in words:
            word_bytes = word.encode('utf-8')
            word_tokens = bpe_encode(self.mergeable_ranks, word_bytes)
            tokens.extend(word_tokens)
        return tokens

    def decode_bytes(self, tokens):
        return b''.join((self._decoder[token] for token in tokens))

    def decode(self, tokens):
        return self.decode_bytes(tokens).decode('utf-8', errors='replace')

    @staticmethod
    def train(training_data, vocab_size, pat_str):
        mergeable_ranks = bpe_train(data=training_data, vocab_size=vocab_size, pat_str=pat_str)
        return SimpleBytePairEncoding(pat_str=pat_str, mergeable_ranks=mergeable_ranks)

def bpe_encode(mergeable_ranks, input):
    parts = [bytes([b]) for b in input]
    while True:
        min_idx = None
        min_rank = None
        for (i, pair) in enumerate(zip(parts[:(- 1)], parts[1:])):
            rank = mergeable_ranks.get((pair[0] + pair[1]))
            if ((rank is not None) and ((min_rank is None) or (rank < min_rank))):
                min_idx = i
                min_rank = rank
        if (min_rank is None):
            break
        assert (min_idx is not None)
        parts = ((parts[:min_idx] + [(parts[min_idx] + parts[(min_idx + 1)])]) + parts[(min_idx + 2):])
    tokens = [mergeable_ranks[part] for part in parts]
    return tokens

def bpe_train(data, vocab_size, pat_str):
    if (vocab_size < (2 ** 8)):
        raise ValueError('vocab_size must be at least 256, so we can encode all bytes')
    ranks = {}
    for i in range((2 ** 8)):
        ranks[bytes([i])] = i
    words = [[bytes([b]) for b in word.encode('utf-8')] for word in re.findall(pat_str, data)]
    while (len(ranks) < vocab_size):
        stats = collections.Counter()
        for piece in words:
            for pair in zip(piece[:(- 1)], piece[1:]):
                stats[pair] += 1
        most_common_pair = max(stats, key=(lambda x: stats[x]))
        token_bytes = (most_common_pair[0] + most_common_pair[1])
        token = len(ranks)
        ranks[token_bytes] = token
        new_words = []
        for word in words:
            new_word = []
            i = 0
            while (i < (len(word) - 1)):
                if ((word[i], word[(i + 1)]) == most_common_pair):
                    new_word.append(token_bytes)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            if (i == (len(word) - 1)):
                new_word.append(word[i])
            new_words.append(new_word)
        words = new_words
    return ranks

def train(data):
    pattern = "'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?\\d+| ?[^\\sa-zA-Z\\d]+|\\s+(?!\\S)|\\s+"
    enc = SimpleBytePairEncoding.train(data, vocab_size=1024, pat_str=pattern)
    tokens = enc.encode('hello world')
    assert (enc.decode(tokens) == 'hello world')
    enc.encode(data)

def bench_bpe_tokeniser(loops):
    DATA = ((Path(__file__).parent / 'data') / 'frankenstein_intro.txt')
    with open(DATA, 'r', encoding='utf8') as f:
        data = f.read()
    range_it = range(loops)
    t0 = pyperf.perf_counter()
    for _ in range_it:
        train(data)
    return (pyperf.perf_counter() - t0)
if (__name__ == '__main__'):
    runner = pyperf.Runner()
    runner.metadata['description'] = 'Benchmark a BPE tokeniser'
    runner.bench_time_func('bpe_tokeniser', bench_bpe_tokeniser)


import pickle
from typing import Iterable, Iterator
import regex as re

import numpy as np
from tqdm import tqdm

from functools import lru_cache

CACHE_SIZE = 65536

class Tokenizer(object):
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None, log_progress = False):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else []# in case of overlapping special tokens, longer one takes precedence
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Create merge lookup dict for O(1) access
        self.merge_dict = {(first, second): i for i, (first, second) in enumerate(self.merges)}
        
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        self.log_progress = log_progress

    @classmethod    
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None, log_progress = False):
        with open(vocab_filepath, 'rb') as vocab_file:
            vocab = pickle.load(vocab_file)
        
        with open(merges_filepath, 'rb') as merges_file:
            merges = pickle.load(merges_file)
        
        return cls(vocab, merges, special_tokens, log_progress)
    
    def apply_merge(self, token: tuple[bytes, ...]) -> list[int]:
        token = list(token)
        
        # Keep applying merges until no more changes
        while True:
            new_token = []
            i = 0
            min_merge_idx = len(self.merges)
            best_pos = -1
            
            # Find the earliest merge (by index) that can be applied
            for j in range(len(token) - 1):
                pair = (token[j], token[j+1])
                if pair in self.merge_dict:
                    merge_idx = self.merge_dict[pair]
                    if merge_idx < min_merge_idx:
                        min_merge_idx = merge_idx
                        best_pos = j
            
            # If no merge found, we're done
            if best_pos == -1:
                break
            
            # Apply the earliest merge
            while i < len(token):
                if i == best_pos:
                    new_token.append(token[i] + token[i+1])
                    i += 2
                else:
                    new_token.append(token[i])
                    i += 1
            
            token = new_token
        
        return [self.reverse_vocab[t] for t in token]
    
    @lru_cache(maxsize=CACHE_SIZE)
    def encode_chunk(self, text: str) -> list[int]:
        tokens = re.findall(self.PAT, text)
        split_tokens = [tuple(bytes([i]) for i in token.encode('utf-8')) for token in tokens]
        ret = []
        for split in split_tokens:
            ret += self.apply_merge(split)
        return ret

    def encode(self, text: str) -> list[int]:
        splits = re.split(f"({'|'.join(map(re.escape, self.special_tokens))})", text) if self.special_tokens else [text]# use capturing group to retain special tokens
        ret = []
        if self.log_progress:
            splits = tqdm(splits)
        for split in splits:
            if split in self.special_tokens:
                ret.append(self.reverse_vocab[split.encode('utf-8')])
            else:
                ret += self.encode_chunk(split)
        return ret
             
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        return (token for text in iterable for token in self.encode(text))

    def decode(self, ids: list[int]) -> str:
        ret = b''
        for token in ids:
            ret += self.vocab[token]
        return ret.decode('utf-8', errors='replace')

if __name__ == "__main__":
    vocab_path = "data/tiny_stories/vocab.pkl"
    merge_path = "data/tiny_stories/merge.pkl"
    special_tokens = ['<|endoftext|>']

    tokenizer = Tokenizer.from_files(vocab_path, merge_path, special_tokens=special_tokens, log_progress=True)

    with open("data/tiny_stories/TinyStoriesV2-GPT4-train.txt", "r") as f:
        text = f.read()
        ids = tokenizer.encode(text)
        arr = np.array(ids)
    with open("data/tiny_stories/tokenized_train.npy", "wb") as f:
        np.save(f, arr)
    # print(tokenizer.encode("the cat ate"))  # Output: 'hello world'
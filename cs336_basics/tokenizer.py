
import pickle
from typing import Iterable, Iterator
import regex as re

class Tokenizer(object):
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else []# in case of overlapping special tokens, longer one takes precedence
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    @classmethod    
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, 'rb') as vocab_file:
            vocab = pickle.load(vocab_file)
        
        with open(merges_filepath, 'rb') as merges_file:
            merges = pickle.load(merges_file)
        
        return cls(vocab, merges, special_tokens)
    
    def apply_merge(self, token: tuple[bytes, ...]) -> list[int]:
        for first, second in self.merges:
            new_token = []
            i = 0
            while i < len(token):
                if i < len(token) - 1 and (token[i], token[i+1]) == (first, second):
                    new_token.append(first + second)
                    i += 2
                else:
                    new_token.append(token[i])
                    i += 1
            token = new_token
        ret = []
        for t in token:
            ret.append(self.reverse_vocab[t])
        return ret
    
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
    vocab_path = "data/vocab_owt.pkl"
    merge_path = "data/merge_owt.pkl"
    special_tokens = ['<|endoftext|>']

    tokenizer = Tokenizer.from_files(vocab_path, merge_path, special_tokens=special_tokens)
    print(tokenizer.encode("the cat ate"))  # Output: 'hello world'
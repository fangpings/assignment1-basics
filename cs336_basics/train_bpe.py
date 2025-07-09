import os

from .pretokenization_example import find_chunk_boundaries
import multiprocessing
import regex as re
from collections import defaultdict

import tqdm
import pickle

NUM_PROCESSES = multiprocessing.cpu_count()
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pretokenization_process_chunk(chunk: str, special_tokens: list[str], pos: int = 0) -> dict[tuple[bytes, ...], int]:
    splits = re.split("|".join(map(re.escape, special_tokens)), chunk)
    ret = defaultdict(int)
    cache = {}
    for split in tqdm.tqdm(splits, position=pos):
        tokens = re.findall(PAT, split)
        for token in tokens:
            if token in cache:
                split_token = cache[token]
            else:
                split_token = tuple(bytes([i]) for i in token.encode('utf-8'))
                cache[token] = split_token
            ret[split_token] += 1
    return ret

def pretokenization(input_path: str | os.PathLike, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, NUM_PROCESSES, "<|endoftext|>".encode("utf-8"))
    
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)
    
    pool = multiprocessing.Pool(processes=NUM_PROCESSES)
    inputs = [(chunk, special_tokens, i) for i, chunk in enumerate(chunks)]
    pretokenizations = pool.starmap(pretokenization_process_chunk, inputs)

    ret = defaultdict(int)
    for p in pretokenizations:
        for k in p:
            ret[k] += p[k]
    return ret

def merge(
    pretoken_frequency: dict[tuple[bytes, ...], int], 
    byte_pair_frequency: dict[tuple[bytes, bytes], int], 
    byte_pair_pretoken_map: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]
) -> tuple[bytes, tuple[bytes, bytes]]:
    # first sort by byte_pair_frequency and get the most frequent one  
    sorted_frequency = sorted(byte_pair_frequency.items(), key=lambda x: (x[1], x[0]))
    merge_byte_pair = sorted_frequency[-1][0]
    # no need to remove it from byte_pair_frequency, since we will subtract all frequency of pretoken that has this byte pair
    # so frequency will always go to 0

    # for all the pretokens that have the byte pair to be merged
    pretokens = byte_pair_pretoken_map[merge_byte_pair].copy()

    for pretoken in pretokens:
        # pop it from pretoken_frequency, since we need to update the pretoken with new byte pair
        frequency = pretoken_frequency.pop(pretoken)
        # since the byte pair changed, we need to first subtract the old byte pair from frequency mapping
        for byte_pair in get_pretoken_byte_pair(pretoken):
            byte_pair_frequency[byte_pair] -= frequency
            if pretoken in byte_pair_pretoken_map[byte_pair]:
                byte_pair_pretoken_map[byte_pair].remove(pretoken)
        
        # now update the pretoken with new byte pair
        new_pretoken = get_new_pretoken(pretoken, merge_byte_pair)
        # first add new pretoken to pretoken_frequency
        pretoken_frequency[new_pretoken] = frequency
        # then update byte_pair_frequency and byte_pair_pretoken_map
        for byte_pair in get_pretoken_byte_pair(new_pretoken):
            byte_pair_frequency[byte_pair] += frequency
            byte_pair_pretoken_map[byte_pair].add(new_pretoken)
    
    # finally return the new vocab and the byte pair merge
    return merge_byte_pair[0] + merge_byte_pair[1], merge_byte_pair
        
def get_new_pretoken(pretoken: tuple[bytes, ...], merge: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    new_pretoken = []
    i = 0
    while i < len(pretoken):
        if i < len(pretoken) - 1 and (pretoken[i], pretoken[i+1]) == merge:
            new_pretoken.append(merge[0] + merge[1])
            i += 2
        else:
            new_pretoken.append(pretoken[i])
            i += 1
    return tuple(new_pretoken)


def get_pretoken_byte_pair(pretoken: tuple[bytes, ...]) -> list[tuple[bytes, bytes]]:
    return [(first, second) for first, second in zip(pretoken[:-1], pretoken[1:])]

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    print("running pre-tokenization...")
    pretoken_frequency = pretokenization(input_path, special_tokens=special_tokens)
    print("running merge...")
    byte_pair_frequency = defaultdict(int)
    byte_pair_pretoken_map = defaultdict(set)
    for pretoken in pretoken_frequency:
        if len(pretoken) < 2:
            continue
        for byte_pair in get_pretoken_byte_pair(pretoken):
            byte_pair_frequency[byte_pair] += pretoken_frequency[pretoken]
            byte_pair_pretoken_map[byte_pair].add(pretoken)

    vocab = [s.encode("utf-8") for s in special_tokens] + [bytes([x]) for x in range(256)]
    merges = []

    for _ in tqdm.tqdm(range(vocab_size - len(vocab))):
        new_vocab, new_merge = merge(pretoken_frequency, byte_pair_frequency, byte_pair_pretoken_map)
        vocab.append(new_vocab)
        merges.append(new_merge)
    
    vocab = dict(enumerate(vocab))
    return vocab, merges
    

if __name__ == "__main__":
    input_path = "data/TinyStoriesV2-GPT4-train.txt"

    # pretokens = pretokenization(input_path, special_tokens=['<|endoftext|>'])
    # pickle.dump(pretokens, open("pretokenized_data.pkl", "wb"))

    vocab, merge = run_train_bpe(input_path, vocab_size=1000, special_tokens=['<|endoftext|>'])
    print(vocab)
    print(merge)

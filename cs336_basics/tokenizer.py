import time
from typing import Iterable, Iterator

import regex as re
from .utils import *
import concurrent.futures as futures
import multiprocessing as mp

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        self.concurrency = mp.cpu_count()

        # initialize
        for token in self.special_tokens:
            if token not in self.vocab:
                self.vocab[len(self.vocab)] = token.encode("utf-8")
        self.vocab_reverse = {v: k for k, v in self.vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        pass

    def encode(self, text: str):
        tokens_list = self._pre_tokenize(text)
        with futures.ProcessPoolExecutor(max_workers=self.concurrency) as executor:
            future_to_token = {executor.submit(self._merge, tokens): (tokens, ) for tokens in tokens_list}




    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        pass

    def _pre_tokenize(self, text: str):
        """
        Pre-tokenize the input text into byte sequences.
        """
        tokens = []
        split_pattern = "|".join(self.special_tokens)
        chunks = re.split(split_pattern, text)
        for chunk in chunks:
            for match in re.finditer(PAT, chunk):
                tokens.append(tuple(bytes([b]) for b in match.group().encode("UTF-8")))
        return tokens

    def _merge(self, tokens):
        for merge in self.merges:
            while len(tokens) > 1 :
                for i in range(len(tokens) - 1):
                    if (tokens[i], tokens[i + 1]) == merge:
                        tokens = (tokens[:i],) + (tokens[i] + tokens[i + 1],) + (tokens[i + 2:],)
                        break
        res = [self.vocab_reverse[token] for token in tokens]
        return res

class BPETokenizerTrainer:
    def __init__(self,
                 special_tokens: list[str] = None,
                 vocab: dict[int, bytes]= None,
                 desired_num_chunks: int = 128,
                 concurrency: int = None):
        """
        Initialize the BPE Tokenizer with special tokens.
        """
        self.vocab = vocab
        self.special_tokens = special_tokens if special_tokens else []
        self.desired_num_chunks = desired_num_chunks
        self.concurrency = concurrency or mp.cpu_count()
        self.merges = []
        self.pair_freqs = {}
        self.token_dit = {}
        self.frequent_table = []
        self.initialize()

    def initialize(self):
        self.vocab = {i : bytes([i]) for i in range(256)}
        for token in self.special_tokens:
            self.vocab[len(self.vocab)] = token.encode("utf-8")

    def process_chunk(self, file, start, end):
        freqs : dict[tuple[bytes], int] = {}
        with open(file, "rb") as f:
            f.seek(start)
            chunk_data = f.read(end - start)
            decoded_chunk = chunk_data.decode('utf-8', errors="ignore")
            # Removing special tokens before pre-tokenization
            split_pattern = "|".join(self.special_tokens)
            chunks = re.split(split_pattern, decoded_chunk)
            for chunk in chunks:
                for match in re.finditer(PAT, chunk):
                    match_bytes = tuple(bytes([b]) for b in match.group().encode("UTF-8"))
                    freqs[match_bytes] = freqs.get(match_bytes, 0) + 1
        return freqs

    def pretokenization(self,file):
        """
        Initialize pre-tokenization from the given text.
        """
        total_counts = {}
        with open(file, "rb") as f:
            boundaries = find_chunk_boundaries(f, self.desired_num_chunks, split_special_token=b"<|endoftext|>")
            chunks = list(zip(boundaries[:-1], boundaries[1:]))
            # use concurrent futures to parallelize the processing
            with futures.ProcessPoolExecutor(max_workers=self.concurrency) as executor:
                future_to_chunk = {executor.submit(self.process_chunk, file, start, end): (start, end)
                                   for start, end in chunks}
                for future in futures.as_completed(future_to_chunk):
                    chunk_counts = future.result()
                    # Merge results into the main token_counts dictionary
                    for token, count in chunk_counts.items():
                        if token in total_counts:
                            total_counts[token] += count
                        else:
                            total_counts[token] = count
        for key, value in total_counts.items():
            self.frequent_table.append((key, value))


    def _update_pair_freqs_after_merge(self,a,b):
        new_token = (a,) + (b,)
        self.pair_freqs.pop(new_token, None)  # Remove the merged pair
        idx_dic = self.token_dit.pop(new_token, {})

        for idx, times in idx_dic.items():
            if times <= 0:
                continue

            tokens = self.frequent_table[idx][0]
            counts = self.frequent_table[idx][1]
            # update tokens with the new token
            new_tokens = ()
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i],) + (tokens[i + 1],) == new_token:
                    new_tokens += (tokens[i] + tokens[i + 1],)
                    # upgrade related pairs
                    if i - 1 >= 0:
                        p = (tokens[i-1],) + (tokens[i],)
                        if p != new_token:
                            self.token_dit[p][idx] -= 1
                            self.pair_freqs[p] -= counts
                    if i + 2 < len(tokens):
                        p = (tokens[i+1],) + (tokens[i + 2],)
                        if p != new_token:
                            self.token_dit[p][idx] -= 1
                            self.pair_freqs[p] -= counts
                    i += 2  # Skip the next token since it's merged
                else:
                    new_tokens += (tokens[i],)
                    i += 1
            tokens = new_tokens
            self.frequent_table[idx] = (tokens, counts)
            for i in range(len(tokens)-1):
                if tokens[i] == a+b or tokens[i+1] == a+b:
                    pair = (tokens[i],) + (tokens[i+1],)
                    self._update_pair_freqs(pair, idx, counts)


    def _update_pair_freqs(self, pair, idx, counts):
        if pair not in self.token_dit:
            self.token_dit[pair] = {}
        self.token_dit[pair][idx] = self.token_dit[pair].get(idx, 0) + 1
        if pair not in self.pair_freqs:
            self.pair_freqs[pair] = counts
        else:
            self.pair_freqs[pair] += counts

    def bpe_merge(self, vocab_size: int):
        # initialize pair freqs
        for idx, (tokens,counts) in enumerate(self.frequent_table):
            for i in range(len(tokens) - 1):
                pair = (tokens[i],) + (tokens[i + 1],)
                self._update_pair_freqs(pair, idx, counts)


        while len(self.vocab) < vocab_size:
            #  take the lexicographically greater pair
            most_frequent_pair = max(self.pair_freqs.items(), key=lambda x: (x[1], x[0]))[0]
            if not most_frequent_pair:
                break

            a, b = most_frequent_pair
            self.merges.append((a, b))
            new_token = a + b
            self.vocab[len(self.vocab)] = new_token
            self._update_pair_freqs_after_merge(a,b)

    def train(self,input_path,vocab_size):
        """
        Train the BPE tokenizer on the input text file.
        """
        print("Pretokenization started...")
        start_time_1 = time.time()
        self.pretokenization(input_path)
        print(f'Pretokenization completed in {time.time() - start_time_1:.2f} seconds.')
        print("BPE merging started...")
        start_time_2 = time.time()
        self.bpe_merge(vocab_size)
        print(f'BPE merging completed in {time.time() - start_time_2:.2f} seconds.')
        # empty temp data
        self.frequent_table = []
        self.pair_freqs = {}
        self.token_dit = {}
        print(f'Training completed in {time.time() - start_time_1:.2f} seconds.')

    def get_vocab(self):
        """
        Returns the current vocabulary.
        """
        return self.vocab

    def get_merges(self):
        """
        Returns the list of merges performed.
        """
        return self.merges


if __name__ == "__main__":
    pass

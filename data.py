"""
data.py — Streaming token data loader for arbitrary text corpora.

Designed for 25B-token training runs from raw text files (one document
per line, or arbitrary text — we just stream). Tokenizer is the GPT-2
BPE via `tiktoken` if available, else a simple byte fallback.

Key properties:
  - Constant memory (streaming, never loads full corpus)
  - Async prefetch on a worker thread
  - Batches are formed by concatenating tokens with EOS boundaries
  - Each batch is [B, T] long tensor + same shifted for next-token target
  - Literal `<|endoftext|>` in corpus text does not crash encoding
"""

from __future__ import annotations

import io
import os
import threading
import queue
from pathlib import Path
from typing import Iterator, Iterable

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def build_tokenizer(name: str = "gpt2"):
    """
    Build a tokenizer. Tries tiktoken first (fast BPE), falls back to
    a byte-level tokenizer (vocab 256) so the codebase is runnable
    without any external dependency.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding(name)
        return enc, enc.n_vocab
    except Exception:
        class ByteTok:
            n_vocab = 256
            def encode(self, s: str) -> list[int]:
                return list(s.encode("utf-8"))
            def decode(self, ids: list[int]) -> str:
                return bytes(ids).decode("utf-8", errors="replace")
        return ByteTok(), 256


def encode_text(tokenizer, text: str) -> list[int]:
    """
    Encode text without crashing on literal special tokens in the corpus.

    tiktoken's default ``encode`` raises ValueError when it sees
    ``<|endoftext|>``. For streaming corpora that may contain that string
    as ordinary text, prefer ``encode_ordinary`` (treat as normal BPE),
    then ``encode(..., allowed_special=...)``, then plain ``encode``.
    """
    encode_ordinary = getattr(tokenizer, "encode_ordinary", None)
    if callable(encode_ordinary):
        return encode_ordinary(text)
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise TypeError(f"tokenizer has no encode: {type(tokenizer)!r}")
    try:
        return encode(text, allowed_special={"<|endoftext|>"})
    except TypeError:
        # ByteTok / callables that only accept the string
        return encode(text)


# ---------------------------------------------------------------------------
# Streaming corpus
# ---------------------------------------------------------------------------

class StreamingCorpus:
    """
    Lazily stream tokens from a directory of text files. Tokens are yielded
    in document order with an EOS token (vocab_size - 1) inserted between
    documents.
    """

    def __init__(
        self,
        path: str | os.PathLike,
        tokenizer,
        eos_id: int,
        chunk_size: int = 1 << 20,   # 1 MB read chunks
    ):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.eos_id = eos_id
        self.chunk_size = chunk_size

    def __iter__(self) -> Iterator[int]:
        if self.path.is_file():
            files = [self.path]
        else:
            files = sorted(p for p in self.path.rglob("*") if p.is_file())
        for f in files:
            # Stream by chunk so multi-GB corpora do not explode RAM.
            # Previous impl concatenated the entire file then tokenized once.
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                while True:
                    chunk = fh.read(self.chunk_size)
                    if not chunk:
                        break
                    ids = encode_text(self.tokenizer, chunk)
                    for i in ids:
                        yield i
                yield self.eos_id


# ---------------------------------------------------------------------------
# Batcher
# ---------------------------------------------------------------------------

class TokenBatcher:
    """
    Consume a token iterator and produce [B, T] long tensors for training.
    Each batch is a contiguous slice of the token stream — no padding, no
    document-boundary alignment (the EOS tokens teach the model boundaries).
    """

    def __init__(
        self,
        token_stream: Iterable[int],
        batch_size: int,
        seq_len: int,
        device: torch.device | str = "cpu",
        prefetch: int = 4,
    ):
        self.stream = iter(token_stream)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = torch.device(device)
        self.prefetch = prefetch

        self._queue: queue.Queue = queue.Queue(maxsize=prefetch)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        B, T = self.batch_size, self.seq_len
        need = B * T + 1
        # Preallocate CPU long buffer; avoid per-token Python list growth +
        # torch.tensor(list) which is a major CPU I/O bottleneck at scale.
        buf = torch.empty(need * 4, dtype=torch.long)
        n_buf = 0
        try:
            while not self._stop.is_set():
                while n_buf < need:
                    try:
                        tok = next(self.stream)
                    except StopIteration:
                        self._stop.set()
                        break
                    if n_buf >= buf.numel():
                        buf = torch.cat([buf, torch.empty(buf.numel(), dtype=torch.long)])
                    buf[n_buf] = tok
                    n_buf += 1
                if self._stop.is_set() and n_buf < need:
                    break
                flat = buf[:need].clone()
                if n_buf > need:
                    buf[: n_buf - need] = buf[need:n_buf]
                n_buf = max(0, n_buf - need)
                x = flat[:-1].view(B, T).to(self.device, non_blocking=True)
                y = flat[1:].view(B, T).to(self.device, non_blocking=True)
                self._queue.put((x, y))
        except Exception as e:
            self._queue.put(e)

    def __iter__(self):
        while not (self._stop.is_set() and self._queue.empty()):
            item = self._queue.get()
            if isinstance(item, Exception):
                raise item
            if item is None:
                break
            yield item

    def close(self):
        self._stop.set()
        try:
            while not self._queue.empty():
                self._queue.get_nowait()
        except Exception:
            pass

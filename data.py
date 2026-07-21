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
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                buf = io.StringIO()
                while True:
                    chunk = fh.read(self.chunk_size)
                    if not chunk:
                        break
                    buf.write(chunk)
                text = buf.getvalue()
                if not text:
                    continue
                ids = self.tokenizer.encode(text)
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
        buf: list[int] = []
        try:
            while not self._stop.is_set():
                need = B * T + 1
                while len(buf) < need:
                    try:
                        buf.append(next(self.stream))
                    except StopIteration:
                        self._stop.set()
                        break
                if self._stop.is_set() and len(buf) < need:
                    break
                arr = buf[:need]
                del buf[:need]
                x = torch.tensor(arr[:-1], dtype=torch.long, device=self.device).view(B, T)
                y = torch.tensor(arr[1:], dtype=torch.long, device=self.device).view(B, T)
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

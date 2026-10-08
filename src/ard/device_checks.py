"""Fail-closed tensor checks that also hold inside a captured CUDA graph (plan 0105).

``require(condition, message, error)`` is the host check ``if not bool(condition):
raise error(message)`` that the scientific code has always made. While the
current CUDA stream is capturing a graph (``training.cuda_graph``) a host read
is impossible, so the same predicate is enqueued as a ``torch._assert_async``
device assert instead: it runs at every replay, and a violation kills the CUDA
context before the poisoned update can reach a checkpoint (the mechanism of
``ard.engine.trainer._assert_finite_training_loss``). Outside a capture the
behaviour -- predicate, exception type and message -- is exactly the former one.
"""

from __future__ import annotations

import torch

__all__ = ["capturing", "require"]


def capturing(tensor: torch.Tensor) -> bool:
    """True iff ``tensor`` lives on CUDA and the current stream is capturing a graph."""
    return tensor.is_cuda and torch.cuda.is_current_stream_capturing()


def require(condition: torch.Tensor, message: str, error: type[Exception] = ValueError) -> None:
    """Raise ``error(message)`` unless the scalar bool tensor ``condition`` holds (device assert under capture)."""
    if capturing(condition):
        torch._assert_async(condition, message)
        return
    if not bool(condition):
        raise error(message)

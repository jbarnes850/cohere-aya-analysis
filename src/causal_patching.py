"""Causal patching primitives for residual-stream interventions.

Three primitives shared between the bidirectional-patching experiment and the
E3 causal-validation experiment:

- ``first_divergence``: index of the first differing token between two id sequences.
- ``teacher_forced_continuation_logprob``: sum log-prob of a continuation under
  an unmodified forward pass.
- ``teacher_forced_continuation_logprob_with_patch``: same, but with the
  last-prompt-position residual at a chosen layer replaced by a donor vector.
"""
from __future__ import annotations

from typing import Any, List, Optional

import torch
import torch.nn.functional as F


def first_divergence(a_ids: List[int], b_ids: List[int]) -> Optional[int]:
    """Index of the first differing token between two id sequences.

    Returns ``None`` if the two sequences are identical and the same length.
    If they share a prefix but differ in length, returns the shorter length.
    """
    for i, (ai, bi) in enumerate(zip(a_ids, b_ids)):
        if ai != bi:
            return i
    if len(a_ids) != len(b_ids):
        return min(len(a_ids), len(b_ids))
    return None


@torch.no_grad()
def teacher_forced_continuation_logprob(
    model: Any,
    input_ids_prompt: torch.Tensor,
    continuation_ids: List[int],
) -> float:
    """Sum of log-probs of ``continuation_ids`` under ``model``, teacher-forced."""
    if not continuation_ids:
        return 0.0
    cont = torch.tensor([continuation_ids], device=input_ids_prompt.device)
    full = torch.cat([input_ids_prompt, cont], dim=1)
    out = model(full, use_cache=False)
    logits = out.logits[0]  # (seq, vocab)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    n_prompt = input_ids_prompt.shape[1]
    total = 0.0
    for k, tok in enumerate(continuation_ids):
        total += float(log_probs[n_prompt - 1 + k, tok].cpu())
    return total


@torch.no_grad()
def teacher_forced_continuation_logprob_with_patch(
    model: Any,
    input_ids_prompt: torch.Tensor,
    continuation_ids: List[int],
    layer_idx: int,
    donor_last_pos_at_prompt_end: torch.Tensor,
) -> float:
    """Logprob of ``continuation_ids`` under a patched forward.

    Patches the residual stream at ``layer_idx`` and at the prompt's last
    position (``n_prompt - 1``), replacing it with
    ``donor_last_pos_at_prompt_end``. Continuation positions are not patched,
    so the donor signal propagates only through the patched location.
    """
    n_prompt = input_ids_prompt.shape[1]
    cont = torch.tensor([continuation_ids], device=input_ids_prompt.device)
    full = torch.cat([input_ids_prompt, cont], dim=1)

    def hook(_module, _input, output):
        if isinstance(output, tuple):
            hidden = output[0].clone()
            hidden[0, n_prompt - 1, :] = donor_last_pos_at_prompt_end.to(hidden.dtype).to(hidden.device)
            return (hidden,) + output[1:]
        hidden = output.clone()
        hidden[0, n_prompt - 1, :] = donor_last_pos_at_prompt_end.to(hidden.dtype).to(hidden.device)
        return hidden

    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    try:
        out = model(full, use_cache=False)
        logits = out.logits[0]
        log_probs = F.log_softmax(logits.float(), dim=-1)
        total = 0.0
        for k, tok in enumerate(continuation_ids):
            total += float(log_probs[n_prompt - 1 + k, tok].cpu())
        return total
    finally:
        handle.remove()

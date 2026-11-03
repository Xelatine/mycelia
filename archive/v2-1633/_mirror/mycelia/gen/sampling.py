"""French generation with coherence guardrails and the exact neural path.

Guarantees:

* no out-of-vocabulary token can be emitted (logits are clamped to the
  tokenizer vocabulary);
* temperature and nucleus (top-p) sampling are configurable;
* a trigram repeated three times is penalised, so the brain cannot loop on the
  same sentence forever;
* the path (visited nodes and traversed links) is recorded for each token, which
  is exactly what the cockpit animates as traffic.

The whole context is re-diffused for every new token, like a recurrent brain:
the signal enters at ``[e]`` and may take a different route each time.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from mycelia.data.tokenizer import BOS, EOS


def _apply_repetition(logits: torch.Tensor, ids: Sequence[int], penalty: float,
                      ngram_penalty: float = 4.0, n: int = 3) -> torch.Tensor:
    if not ids:
        return logits
    if penalty and penalty != 1.0:
        idx = torch.tensor(sorted(set(int(i) for i in ids)), device=logits.device)
        vals = logits[idx]
        logits = logits.clone()
        logits[idx] = torch.where(vals > 0, vals / penalty, vals * penalty)
    if len(ids) >= n - 1:
        counts: Dict[tuple, int] = {}
        for i in range(len(ids) - n + 1):
            gram = tuple(int(t) for t in ids[i:i + n])
            counts[gram] = counts.get(gram, 0) + 1
        prefix = tuple(int(t) for t in ids[-(n - 1):])
        banned = [gram[-1] for gram, count in counts.items()
                  if count >= 3 and gram[: n - 1] == prefix]
        if banned:
            idx = torch.tensor(banned, device=logits.device, dtype=torch.long)
            logits = logits.clone()
            logits[idx] -= ngram_penalty
    return logits


def _top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    remove = cumulative > top_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
    out = torch.full_like(logits, -float("inf"))
    out.scatter_(0, sorted_idx, sorted_logits)
    return out


@torch.no_grad()
def generate(model, tokenizer, prompt: str = "Il était une fois",
             max_new_tokens: int = 48, temperature: float = 0.85,
             top_p: float = 0.9, repetition_penalty: float = 1.15,
             device: str = "cpu", track: bool = True, max_ctx: int = 64,
             seed: Optional[int] = None) -> Dict[str, Any]:
    was_training = model.training
    model.eval()
    if seed is not None:
        torch.manual_seed(int(seed))

    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [BOS]
    ids: List[int] = list(prompt_ids[-max_ctx:])
    generated: List[int] = []
    meta: List[Dict[str, Any]] = []
    guard_fired = False
    vocab = int(model.vocab_size)
    prev_decoded = tokenizer.decode(ids)

    for _ in range(int(max_new_tokens)):
        ctx = ids[-max_ctx:]
        token_t = torch.tensor([ctx], dtype=torch.long, device=device)
        out = model(token_t, tau=0.15, track=track, update_usage=False,
                    gumbel=False, step=getattr(model, "step", 0))
        logits = out["logits"][0, -1].float()
        if logits.numel() != vocab:
            logits = logits[:vocab]

        if temperature <= 1e-4:
            token = int(torch.argmax(logits).item())
        else:
            logits = logits / float(temperature)
            logits = _apply_repetition(logits, generated, float(repetition_penalty))
            logits = _top_p(logits, float(top_p))
            logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
            probs = torch.softmax(logits, dim=-1)
            if not torch.isfinite(probs).all() or float(probs.sum()) <= 0:
                token = int(torch.argmax(logits).item())
                guard_fired = True
            else:
                token = int(torch.multinomial(probs, 1).item())
        token = max(0, min(vocab - 1, token))
        if token == EOS:
            break

        generated.append(token)
        ids.append(token)
        decoded = tokenizer.decode(ids)
        if decoded.startswith(prev_decoded):
            piece = decoded[len(prev_decoded):]
        else:
            piece = tokenizer.decode([token])
        prev_decoded = decoded

        trace = out.get("trace") or {"nodes": [], "edges": []}
        nodes = trace.get("nodes", [])[:96]
        edges = trace.get("edges", [])[:96]
        meta.append({
            "token": piece,
            "token_id": int(token),
            "nodes": nodes,
            "edges": [[int(a), int(b)] for a, b in edges],
            "focus_node": int(nodes[-1]) if nodes else None,
            "energy": round(float(out.get("energy", 0.0)), 6),
        })

    generated_text = tokenizer.decode(generated)
    full_text = tokenizer.decode(ids)
    if not full_text.startswith(prompt):
        full_text = prompt + generated_text
    model.train(was_training)
    return {
        "prompt": prompt,
        "text": full_text,
        "generated_text": generated_text,
        "tokens": meta,
        "guard_fired": guard_fired,
        "nodes_per_token": round(sum(len(t["nodes"]) for t in meta) / max(1, len(meta)), 2),
    }

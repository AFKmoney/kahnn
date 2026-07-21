"""
smoke.py — 30-second end-to-end smoke test.

Builds a tiny KAHNN, runs forward + backward + OnlineLearner step on a
random batch, generates a few tokens, and asserts shapes/finite values.

Run:  python -m kuro_brain.smoke
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import KAHNN, KAHNNConfig
from .continuous_learning import OnlineLearner
from .config import SMOKE


def main():
    cfg = SMOKE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device={device}, config: D={cfg.dim} L={cfg.n_layers}")

    model = KAHNN(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke] params: {n_params/1e6:.2f}M")

    B, T = 4, 32
    x = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    y = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    # Forward
    logits = model(x)
    assert logits.shape == (B, T, cfg.vocab_size), f"bad shape {logits.shape}"
    assert torch.isfinite(logits).all(), "non-finite logits"
    print(f"[smoke] forward OK, logits={tuple(logits.shape)}")

    # Loss + backward
    loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
    loss.backward()
    assert torch.isfinite(loss).all(), "non-finite loss"
    print(f"[smoke] loss={loss.item():.4f}")

    # Optimizer + online learner
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    learner = OnlineLearner(model, cfg, base_optimizer=opt)
    learner.step(x, y, loss.detach())
    opt.zero_grad(set_to_none=True)
    print(f"[smoke] online step OK, state={learner.state()}")

    # Generate
    prompt = torch.randint(0, cfg.vocab_size, (1, 8), device=device)
    out = model.generate(prompt, n_new=16, temperature=1.0, top_k=10)
    assert out.shape == (1, 8 + 16), f"bad gen shape {out.shape}"
    print(f"[smoke] generate OK, out_shape={tuple(out.shape)}")

    # Continuous mode
    learner.continuous_mode = True
    for _ in range(3):
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
        loss.backward()
        learner.step(x, y, loss.detach())
    print(f"[smoke] continuous mode OK, state={learner.state()}")

    print("[smoke] ALL PASSED")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""teach.py — Lifelong teach / probe / forget for Kahnn continuous mode.

See docs/LIFELONG_LEARNING.md. Knowledge sticks until explicit forget.

  python teach.py teach --text "Paris is the capital of France." --config smoke --output ./runs/teach
  python teach.py probe --text "Paris is the capital" --resume ./runs/teach/ckpt_teach.pt
  python teach.py forget --text "Paris is the capital of France." --resume ./runs/teach/ckpt_teach.pt
  python teach.py smoke --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from kuro_brain.model import KAHNN
from kuro_brain.config import CONFIGS, describe_config
from kuro_brain.continuous_learning import OnlineLearner
from kuro_brain.device import pick_device


def _build_model(args, device):
    cfg = CONFIGS[args.config]
    if args.config != "smoke":
        try:
            import tiktoken
            cfg.vocab_size = tiktoken.get_encoding("gpt2").n_vocab
        except Exception:
            pass
    model = KAHNN(cfg).to(device)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"[resume] {args.resume} step={ckpt.get('step')}")
    return model, cfg


def _encode_text(text: str, cfg, device, max_len: int = 128) -> torch.Tensor:
    if cfg.vocab_size <= 256:
        ids = [min(ord(c), cfg.vocab_size - 1) for c in text[:max_len]] or [1]
        return torch.tensor([ids], dtype=torch.long, device=device)
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode(text)[:max_len] or [enc.eot_token]
    return torch.tensor([ids], dtype=torch.long, device=device)


def _save(model, cfg, out_dir: Path, step: int, extra=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ckpt_teach.pt"
    payload = {
        "model": model.state_dict(),
        "step": step,
        "config": cfg.__dict__,
        "lifelong": True,
    }
    if extra:
        payload["teach_meta"] = extra
    torch.save(payload, path)
    print(f"[saved] {path}")
    return path


def cmd_teach(args):
    info = pick_device(args.device)
    model, cfg = _build_model(args, info.device)
    text = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
    if not text or not str(text).strip():
        raise SystemExit("teach needs --text or --file")
    token_ids = _encode_text(text.strip(), cfg, info.device, max_len=args.max_len)
    learner = OnlineLearner(
        model, cfg, continuous_mode=True, soft_decay=args.soft_decay
    )
    print(f"[device] {info.kind}  {describe_config(cfg)}")
    print(f"[teach] tokens={token_ids.numel()} passes={args.passes} soft_decay={args.soft_decay}")
    result = learner.teach(token_ids, n_passes=args.passes, force_consolidate=True)
    print(json.dumps({
        "loss_mean": result["loss_mean"],
        "losses": result["losses"],
        "occupied": result["consolidate"]["occupied"],
        "probe_best_memory": result["probe"]["best_memory"],
        "probe": result["probe"]["per_layer"],
    }, indent=2))
    if args.output:
        _save(model, cfg, Path(args.output), learner._step, {
            "text_preview": text.strip()[:200],
            "probe": result["probe"],
        })


def cmd_probe(args):
    info = pick_device(args.device)
    model, cfg = _build_model(args, info.device)
    text = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
    token_ids = _encode_text(text.strip(), cfg, info.device, max_len=args.max_len)
    probe = OnlineLearner(model, cfg, continuous_mode=True).probe(token_ids)
    print(json.dumps(probe, indent=2))


def cmd_forget(args):
    info = pick_device(args.device)
    model, cfg = _build_model(args, info.device)
    learner = OnlineLearner(model, cfg, continuous_mode=True)
    if args.all:
        out = learner.forget_all()
        print(json.dumps({"forgot_all": True, **out}, indent=2))
    elif args.slot is not None:
        if args.layer is None:
            raise SystemExit("--slot requires --layer")
        out = learner.forget_slot(args.layer, args.slot)
        print(json.dumps(out, indent=2))
    else:
        text = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
        if not text:
            raise SystemExit("forget needs --text/--file, or --slot/--all")
        token_ids = _encode_text(text.strip(), cfg, info.device, max_len=args.max_len)
        out = learner.forget(query_ids=token_ids, min_coherence=args.min_coherence)
        print(json.dumps(out, indent=2))
    if args.output:
        _save(model, cfg, Path(args.output), getattr(learner, "_step", 0), {"forgot": True})


def cmd_smoke(args):
    info = pick_device(args.device)
    cfg = CONFIGS["smoke"]
    model = KAHNN(cfg).to(info.device)
    learner = OnlineLearner(model, cfg, continuous_mode=True, soft_decay=False)
    torch.manual_seed(0)
    fact = torch.randint(0, cfg.vocab_size, (1, 32), device=info.device)
    other = torch.randint(0, cfg.vocab_size, (1, 32), device=info.device)
    before = learner.probe(fact)
    learner.teach(fact, n_passes=2, force_consolidate=True)
    after_teach = learner.probe(fact)
    after_other = learner.probe(other)
    usage_before = float(model.memory.usage.sum().item())
    for _ in range(50):
        logits = model(fact)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), fact.reshape(-1))
        loss.backward()
        learner.step(fact, fact, loss.detach())
        model.zero_grad(set_to_none=True)
    usage_after = float(model.memory.usage.sum().item())
    forgot = learner.forget(query_ids=fact, min_coherence=0.2)
    after_forget = learner.probe(fact)
    report = {
        "before_best": before["best_memory"],
        "after_teach_best": after_teach["best_memory"],
        "after_teach_occupied": after_teach["occupied"],
        "other_best": after_other["best_memory"],
        "usage_sum_before_steps": usage_before,
        "usage_sum_after_50_steps": usage_after,
        "soft_decay": learner.soft_decay,
        "forgot_slots": len(forgot["memory_slots_cleared"]),
        "after_forget_best": after_forget["best_memory"],
        "after_forget_occupied": after_forget["occupied"],
        "ok_teach_improved": after_teach["best_memory"] >= before["best_memory"],
        "ok_no_global_decay": usage_after >= usage_before - 1e-6,
    }
    print(json.dumps(report, indent=2))
    if not (report["ok_teach_improved"] and report["ok_no_global_decay"]):
        raise SystemExit("smoke failed")
    print("[smoke] PASS — teach consolidates; no auto global decay in lifelong mode")


def main():
    p = argparse.ArgumentParser(description="Kahnn lifelong teach / forget / probe")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--config", default="smoke", choices=list(CONFIGS))
        sp.add_argument("--device", default="auto")
        sp.add_argument("--resume", default=None)
        sp.add_argument("--output", default=None)
        sp.add_argument("--max-len", type=int, default=128)
        sp.add_argument("--text", default=None)
        sp.add_argument("--file", default=None)

    sp = sub.add_parser("teach")
    add_common(sp)
    sp.add_argument("--passes", type=int, default=3)
    sp.add_argument("--soft-decay", action="store_true")

    sp = sub.add_parser("probe")
    add_common(sp)

    sp = sub.add_parser("forget")
    add_common(sp)
    sp.add_argument("--min-coherence", type=float, default=0.45)
    sp.add_argument("--layer", type=int, default=None)
    sp.add_argument("--slot", type=int, default=None)
    sp.add_argument("--all", action="store_true")

    sp = sub.add_parser("smoke")
    sp.add_argument("--device", default="cpu")
    sp.add_argument("--config", default="smoke", choices=list(CONFIGS))

    args = p.parse_args()
    {"teach": cmd_teach, "probe": cmd_probe, "forget": cmd_forget, "smoke": cmd_smoke}[args.cmd](args)


if __name__ == "__main__":
    main()

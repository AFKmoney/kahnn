# CPU run log / journal d’expériences — 2026-09-08 → 2026-09-09 (FINAL nano)

Mesures **réelles** uniquement (box Cursor / AFKmoney Kahnn). Aucun chiffre inventé.
Fuseau utilisateur : America/Toronto (UTC−4). Timestamps ci-dessous en **PT**.

## Matériel & stack

| Item | Valeur |
|------|--------|
| Device | **CPU only** (`torch.cuda.is_available() == False`) |
| Threads | `torch.set_num_threads(8)` — `nproc` = 8 |
| PyTorch | `2.14.0+cpu` |
| CUDA | non |
| Repo | `AFKmoney/kahnn` @ `main` (après merge PR [#1](https://github.com/AFKmoney/kahnn/pull/1)) |
| Entrée | `train_universal.py`, configs `smoke` / `nano` / `commodity` |
| Vocab | GPT-2 BPE (`tiktoken`), V≈50257 |

## 1. Ce qui a shippé (PR #1, mergée 2026-09-08 ~16:07 PT)

Titre : *feat: universal CPU trainer + lifelong continuous learning (no auto-forget)*  
URL : https://github.com/AFKmoney/kahnn/pull/1

Livré :

- `train_universal.py` — device auto (`cpu` / `cuda` / `mps`), tokens Chinchilla, progressive depth, mode continuous, `torch.compile` avec fallback eager, **`--soft-decay` opt-in (défaut OFF)**
- `kuro_brain/device.py` — sélection device + heuristiques de batch
- Configs **`nano` (~1.8M)** et **`commodity` (~13M)** pour budgets $0–$50
- Hotpath CPU : decode matmul (plus de OOM `[N,V,D]`), engram pull O(E·D), plasticité Hebbienne off pendant Adam, streaming corpus par chunks
- Lifelong : plus de `memory.decay(0.9995)` inconditionnel ; forget explicite via `teach.py` / `OnlineLearner.forget*`
- Docs : `docs/UNIVERSAL_TRAINING.md`, `docs/LIFELONG_LEARNING.md`

## 2. Benches smoke / débit (avant le long run)

Mesures `train_universal.py --smoke-steps` (eager, 8 threads), documentées aussi dans PR #1 :

| Config | Shape (micro×seq) | tps observé | Notes |
|--------|-------------------|-------------|--------|
| `smoke` (~0.3M) | 4×32 | **~870–1040** tok/s | bande observée sur plusieurs runs courts ; PR #1 citait ~870–910 |
| `nano` (~1.8M) | 2×64 (bench early) | **~560** tok/s | bench smoke court ; **pas** le même shape que le pretrain live |
| `commodity` (~13M) | 1×64 | **~145–170** tok/s | bench antérieur, toujours la référence documentée |

Notes :

- Le decode `flat @ codebook.T` (V×D) domine dès que vocab=50k ; l’ancien broadcast OOM (~52 Go) est corrigé.
- `torch.compile` exige un compilateur C++ ; sinon fallback eager (cas de cette box).

## 3. `teach.py smoke` (lifelong)

Sur CPU, même journée :

| Check | Résultat |
|-------|----------|
| Probe avant teach | **~0** (`best_memory`) |
| Probe après teach | **~0.96** |
| 50 steps continuous, `soft_decay=False` | somme des usages **stable** (pas de decay global auto) |
| `forget` | slots matching vidés ; occupied → 0 |
| soft_decay unitaire | slots faibles ages, slots forts tenus |

Commande : `python teach.py smoke --device cpu` → PASS.

## 4. Corpus de pretrain

| Champ | Valeur |
|-------|--------|
| Path local | `/workspace/kahnn/data/corpus.txt` |
| Taille | **~109 MiB** (114072042 octets) |
| Tokens estimés | **~46M** GPT-2 (ratio ~0.406 tok/char, échantillon) |
| Provenance | Voir **`data/DATA.md`** (Gutenberg PD, TinyStories-valid, Wikipedia extracts, sources Kahnn, extraits CPython 3.12) |
| Commit | **Ne pas** committer le `.txt` 109 MB dans git ; documenter + rebuild |

Construction (résumé) : mix prose+code (~38 MB / ~15.4M tok) concaténé **3×** pour couvrir nano Chinchilla (~37M) en un pass `StreamingCorpus`.

## 5. Live nano pretrain (`runs/nano_base`)

### Commande

```bash
cd /workspace/kahnn
python train_universal.py \
  --data /workspace/kahnn/data/corpus.txt \
  --output /workspace/kahnn/runs/nano_base \
  --config nano --device auto \
  --log-every 10 --checkpoint-every 500
```

### Config runtime (log)

```
[device] cpu (CPU) -- torch.set_num_threads(8)
[config] KAHNN config: D=2048, L=3, params≈1.8M
[batch] micro=4 seq=256 accum=2 tokens/step=2,048
[tokens] target=36,986,940 (~20.0 tok/param)
[model] trainable params: 1.86M
[optim] AdamW lr=0.0003
[progressive-depth] 1/3, grow_every=6020
[train] start -- total_steps~18,060 warmup=361
```

### Démarrage

- **Started** : 2026-09-08 ~16:11 PT (20:11 UTC)
- **PID** : écrit dans `runs/nano_base/train.pid` (ex. `2874796` — ne pas tuer pour la doc)
- Logs : `runs/nano_base/train_universal.log`, miroir `console.out`
- Notes run : `runs/nano_base/RUN.md`

### TPS early (extrait réel du log)

| step | tokens | loss | tps | layers | ETA |
|------|--------|------|-----|--------|-----|
| 10 | 20480 | 10.8254 | **1070.8** | 1/3 | 9.6h |
| 20 | 40960 | 10.8245 | **1169.7** | 1/3 | 8.8h |
| 40 | 81920 | 10.8239 | **1221.0** | 1/3 | 8.4h |
| 50 | 102400 | 10.8223 | **1232.3** | 1/3 | 8.3h |
| 80 | 163840 | 10.7707 | **1239.0** | 1/3 | 8.3h |
| 100 | 204800 | 10.7170 | **1180.1** | 1/3 | 8.7h |
| 200 | 409600 | 10.5377 | **1209.4** | 1/3 | 8.4h |
| 350 | 716800 | 10.3808 | **1218.1** | 1/3 | 8.3h |

**Bande early live nano :** ~**1070–1230** tok/s (shape `4×256`, accum 2 — différent du smoke `2×64` ~560 tps).

ETA wall-clock annoncée par le trainer : **~8.3–9.6 h** pour ~37M tokens à ce débit (sujet à ralentissement quand progressive-depth passe à 2/3 puis 3/3).

### Crash tiktoken EOT (~step 2000) + fix PR #4

Vers **step ~2000** (`ckpt_2000.pt`, ~4.1M tokens), le run a planté :

```text
ValueError: Encountered text corresponding to disallowed special token '<|endoftext|>'.
```

Cause : le corpus contient le marqueur littéral GPT-2 `<|endoftext|>` ; `tiktoken.encode` le refuse par défaut.

**Fix** (mergé dans PR [#4](https://github.com/AFKmoney/kahnn/pull/4) — `ee53dbd`) :
`data.encode_text` / `encode_ordinary` (+ harden `teach.py` / `generate.py`). Resume :

```bash
python train_universal.py \
  --data /workspace/kahnn/data/corpus.txt \
  --output /workspace/kahnn/runs/nano_base \
  --config nano --device auto \
  --resume /workspace/kahnn/runs/nano_base/ckpt_2000.pt
```

Log : `[resume] step=2000 tokens=4,096,000`.

### FINAL nano (2026-09-09) — `ckpt_final.pt`

Mesure réelle du log `runs/nano_base/console.out` :

| Champ | Valeur |
|-------|--------|
| Fichier | `runs/nano_base/ckpt_final.pt` (~856 MiB, **gitignored**) |
| Tokens | **36,987,904** (`[done]`) |
| tps final | **≈1361.7** tok/s (layers **3/3**) |
| Last loss | **≈10.50** (`step=18060` loss=`10.5002`) |
| Wall resume | **~7.5 h** de train après resume `ckpt_2000` (post fix tiktoken) |
| Calendrier total | **~10 h** incluant le crash EOT + downtime + resume |

Extrait fin de run :

```text
step=18050 tokens=36,966,400 loss=10.4908 tps=1361.7 layers=3/3 eta=0.0h
step=18060 tokens=36,986,880 loss=10.5002 tps=1361.7 layers=3/3 eta=0.0h
[done] tokens=36,987,904 tps=1361.7 saved=.../runs/nano_base/ckpt_final.pt
```

**Poids non versionnés** — voir [`NANO_FINAL.md`](./NANO_FINAL.md) (chemins locaux + upload Drive/HF).

### Teach sur `ckpt_final` (2026-09-09)

```bash
python teach.py teach --text "La capitale du Canada est Ottawa." \
  --config nano --resume ./runs/nano_base/ckpt_final.pt \
  --output ./runs/teach

python teach.py probe --text "capitale du Canada" \
  --resume ./runs/teach/ckpt_teach.pt --config nano
```

| Métrique | Valeur observée |
|----------|-----------------|
| `probe_best_memory` (après teach) | **~1.00** (`0.9999985…`) |
| `occupied` | **570** |
| CE loss pendant teach | **~10.8** (`loss_mean≈10.828`) |
| Probe partiel `"capitale du Canada"` | **~0.43** (`best_memory≈0.4256`) |
| Checkpoint teach | `runs/teach/ckpt_teach.pt` (gitignored) |

**Honnêteté :** consolidation mémoire OK ; **génération encore pauvre** (CE ~10.8). Nano Chinchilla CPU = base de départ, pas un LLM fluide. Détail : [`NANO_FINAL.md`](./NANO_FINAL.md).

### Après le run

```bash
# Resume si interrompu
python train_universal.py \
  --data /workspace/kahnn/data/corpus.txt \
  --output /workspace/kahnn/runs/nano_base \
  --config nano --device auto \
  --resume /workspace/kahnn/runs/nano_base/ckpt_N.pt

# Lifelong teach
python teach.py teach --text "Fait métier…" \
  --resume /workspace/kahnn/runs/nano_base/ckpt_final.pt \
  --config nano --output ./runs/teach

python teach.py probe --text "…" --resume ./runs/teach/ckpt_teach.pt --config nano
python teach.py forget --text "…" --resume ./runs/teach/ckpt_teach.pt --config nano --output ./runs/teach
```

## 6. Limites honnêtes (rappel)

1. **B1 sur CPU** : irréaliste (20B tokens / ~1B params).
2. **Mémoire « sticky »** = engrammes consolidés, pas magie des poids fondation.
3. Il faut d’abord une **base texte+code** (Adam) ; continuous n’invente pas la langue.
4. **Slots finis** → eviction `argmin(usage)` quand plein ; pas d’oubli auto en lifelong défaut.
5. Les ×8.14 FP8/MoD/PGSU sont pour Ada/Hopper / multi-GPU — **pas** le chemin CPU.

## 7. Liens

- Produit / quickstart CPU : [`UNIVERSAL_TRAINING.md`](./UNIVERSAL_TRAINING.md)
- Lifelong teach/forget : [`LIFELONG_LEARNING.md`](./LIFELONG_LEARNING.md)
- Récap FINAL nano + poids locaux : [`NANO_FINAL.md`](./NANO_FINAL.md)
- Corpus : [`../data/DATA.md`](../data/DATA.md)
- Cluster / B1 (secondaire) : [`B1_TRAINING.md`](./B1_TRAINING.md), [`RTX4000_TRAINING.md`](./RTX4000_TRAINING.md)

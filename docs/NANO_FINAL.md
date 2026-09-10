# Nano FINAL — résultats + poids locaux (2026-09-09)

Mesures **réelles** sur la box Cursor (CPU-only). Aucun chiffre inventé.
Journal détaillé : [`CPU_RUN_LOG.md`](./CPU_RUN_LOG.md). Lifelong : [`LIFELONG_LEARNING.md`](./LIFELONG_LEARNING.md).

## 1. Prétrain `nano` terminé

| Champ | Valeur |
|-------|--------|
| Config | `nano` (~1.8M params, D=2048, L=3) |
| Device | CPU, 8 threads |
| Checkpoint | **`runs/nano_base/ckpt_final.pt`** |
| Tokens | **36,987,904** |
| tps | **≈1361.7** |
| Last loss | **≈10.50** |
| Resume | depuis `ckpt_2000` après fix tiktoken (PR [#4](https://github.com/AFKmoney/kahnn/pull/4)) |
| Wall | ~**7.5 h** resume ; calendrier ~**10 h** incl. crash `<|endoftext|>` |

Crash cause : marqueur littéral `<|endoftext|>` dans le corpus → `tiktoken.encode` ValueError.
Fix : `encode_ordinary` / `data.encode_text` (PR #4).

## 2. Teach / probe sur `ckpt_final` (2026-09-09)

Commandes :

```bash
python teach.py teach --text "La capitale du Canada est Ottawa." \
  --config nano --resume ./runs/nano_base/ckpt_final.pt \
  --output ./runs/teach

python teach.py probe --text "capitale du Canada" \
  --resume ./runs/teach/ckpt_teach.pt --config nano
```

| Métrique | Valeur |
|----------|--------|
| `probe_best_memory` (teach) | **~1.00** |
| `occupied` | **570** |
| CE loss (teach) | **~10.8** |
| Probe partiel `"capitale du Canada"` | **~0.43** |
| Sortie | **`runs/teach/ckpt_teach.pt`** |

Génération encore pauvre malgré un rappel mémoire fort — attendu à cette échelle / loss.

## 3. Où sont les poids (pas dans git)

`.gitignore` exclut déjà `*.pt`, `runs/`, `data/corpus.txt`. **Ne pas** committer les checkpoints (~800+ MiB chacun).

| Artefact | Path local (box) | Taille ordre |
|----------|------------------|--------------|
| Prétrain final | `runs/nano_base/ckpt_final.pt` | ~856 MiB |
| Après teach | `runs/teach/ckpt_teach.pt` | ~841 MiB |
| Intermédiaires | `runs/nano_base/ckpt_*.pt` | ~846–856 MiB |
| Corpus | `data/corpus.txt` | ~109 MiB |

### Upload (HF préféré ; hors ce PR)

Les `.pt` sont destinés à **Hugging Face** (upload séparé — ce PR est docs-only, pas de token requis ici).

```bash
# Exemple (nécessite HF_TOKEN / huggingface-cli login)
huggingface-cli upload <user>/kahnn-nano-cpu \
  runs/nano_base/ckpt_final.pt \
  --repo-type model
# idem pour runs/teach/ckpt_teach.pt
```

**Google Drive** (alternative Colab) : `USE_DRIVE=True` dans `notebooks/kahnn_nano_colab.ipynb`, ou copie manuelle vers `MyDrive/kahnn/runs/…`.

**GitHub Releases / git** : ne pas committer ni uploader ~800 MiB × N ckpts (déjà gitignored).

Sans HF/Drive : les poids restent sur la box aux chemins ci-dessus.

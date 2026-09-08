# Formation universelle Kahnn — CPU / petit GPU (coût minimal)

Objectif : entraîner un modèle Kahnn **de qualité utile** sans H100 ni
cluster 5×4090. Entrée recommandée : `train_universal.py`.

> **Pourquoi CPU-first ?** Le but produit n’est pas « multi-H100 ». C’est
> un cerveau Kahnn qui apprend **en continu** sur machine perso / box
> modestes, puis consolide des faits via engrammes (`teach.py`) sans
> full retrain. Voir aussi `docs/LIFELONG_LEARNING.md` et le journal
> daté `docs/CPU_RUN_LOG.md` (2026-09-08).

## 1. Principe

La qualité n’égale pas « lancer B1 (1B params) ». Pour Kahnn :

- Les embeddings token sont **fixes** (pas de grosse matrice apprise).
- Le mélange séquence est **linéaire en T** (Kuramoto), pas O(T²).
- Après un prétrain court, le mode **continuous learning** (plasticité
  locale + engrammes) continue d’apprendre sans rejouer tout le corpus.

Donc : **petit modèle + tokens Chinchilla + continuous learning** bat
souvent un gros modèle sous-entraîné sur un budget de quelques dollars.

## 2. Configs coût-optimal

| Config | Params | Tokens Chinchilla (20×) | Cible matériel |
|---|---|---|---|
| `smoke` | ~0.3M | ~6M | sanity CPU |
| `nano` | ~1.8M | ~37M | laptop CPU / box 8 threads |
| `commodity` | ~13M | ~264M | CPU fort / GPU 8–12 Go |
| `tiny` | ~6.6M | ~132M | CPU / GPU entry |
| `medium` | ~52M | ~1.05B | GPU 12–24 Go |

Évitez `b1` / `xl` sur CPU ou mono-GPU grand public si le but est le
rapport qualité/prix — le FLOP budget explose (voir `docs/B1_TRAINING.md`).

## 3. Lancement

GPU gratuit : voir aussi [`docs/COLAB_GPU.md`](COLAB_GPU.md) + [`notebooks/kahnn_nano_colab.ipynb`](../notebooks/kahnn_nano_colab.ipynb) (T4/L4 ; mesurer le tps sur place).

```bash
pip install -r requirements.txt
# corpus texte (fichier ou dossier) — voir data/DATA.md

# Auto device (cuda → mps → cpu), preset commodity
python train_universal.py \
  --data /path/to/corpus \
  --output ./runs/commodity_auto \
  --config commodity

# Laptop / box CPU, nano, Chinchilla ~37M tokens
python train_universal.py \
  --data ./data/corpus.txt \
  --output ./runs/nano_base \
  --config nano --device auto \
  --log-every 10 --checkpoint-every 500

# Smoke de débit
python train_universal.py \
  --data /path/to/corpus \
  --output ./runs/nano_cpu \
  --config nano --device cpu \
  --max-tokens 2000000 --smoke-steps 30

# Continuer sans Adam (lifelong engrammes ; pas de decay auto)
python train_universal.py \
  --data /path/to/new_domain.txt \
  --output ./runs/commodity_cont \
  --config commodity --continuous \
  --resume ./runs/commodity_auto/ckpt_final.pt
```

Flags utiles :
- `--progressive-depth` (défaut) : démarre avec ~1/3 des couches.
- `--compile` : tente `torch.compile` (fallback silencieux).
- `--mod` : Mixture-of-Depths (surtout utile GPU).
- `--activation-checkpointing` : petit GPU VRAM limitée.
- `--enable-plasticity` : plasticité locale **pendant** Adam (plus lent ;
  réservé au mode continuous en général).
- `--soft-decay` : avec `--continuous`, age seulement les slots faibles
  (défaut OFF = mémoire à vie jusqu’à `teach.py forget`).

## 4. Speedups déjà dans le code (pertinents CPU / petit GPU)

1. **Decode matmul** (`encoder.py`) — l’ancien `similarity` broadcast
   allouait ~[N,V,D] (dizaines de Go pour vocab GPT-2). Remplacé par
   `flat @ codebook.T`.
2. **Engram pull O(E·D)** (`kuramoto.py`) — sans tenseur `[...,E,D]`.
3. **Plasticité locale off en prétrain Adam** (`continuous_learning.py`)
   — évite un second forward Kuramoto par step ; réactivée en continuous.
4. **Streaming corpus par chunks** (`data.py`) — plus de chargement
   entier du fichier en RAM.
5. **Batcher buffer tensor** (`data.py`) — moins d’overhead Python.
6. **Progressive depth + MoD + checkpointing** — déjà dans le modèle.

## 5. Pareto qualité / coût (ordre de grandeur honnête)

Budgets **compute** approximatifs (électricité / location Vast·RunPod),
pas le salaire ingénieur. Les débits CPU ci-dessous sont ancrés sur des
mesures box 8 threads (2026-09-08) ; GPU = estimation prudente MFU.

| Budget | Setup réaliste | Config | Tokens | Qualité attendue |
|---|---|---|---|---|
| **~$0** | PC perso, nuits CPU | `nano` | 20–40M | Proto / domaine étroit ; perplexité élevée hors domaine |
| **~$5–20** | 1× RTX 3060/4060 quelques heures **ou** CPU plusieurs jours | `commodity` | ~200–300M | Modèle « utile » sur corpus ciblé + continuous fine-tune |
| **~$50–100** | 1× 4090 ~1–2 j **ou** A10S | `tiny`→`medium` | 0.1–1B | Bon niveau domaine ; pas GPT-3, mais stable |
| **~$200** | 1× 4090 plusieurs jours | `medium` plein Chinchilla | ~1B | Meilleur rapport Kahnn sans cluster |
| **≫ $200** | multi-GPU | `b1` | 20B | Voir docs B1/RTX — **hors** chemin universel |

**Ce qui n’est pas réaliste :** B1 20B tokens sur CPU seul, ou « haute
qualité type fondation » à $0. Kahnn réduit le coût vs Transformer, mais
ne supprime pas le besoin de tokens.

## 6. Stratégies qualité ≠ FLOPs bruts

1. **Curriculum de contexte** : `seq=128→256→512` (relancer avec
   `--resume` et `--seq-len` croissant).
2. **Prétrain court + continuous** : Adam sur corpus général, puis
   `--continuous` / `teach.py` sur données métier (pas de replay buffer).
3. **Progressive depth** : déjà activé dans `train_universal.py`.
4. **Données propres** > tokens bruités : 50M tokens filtrés battent
   souvent 300M web dump pour une tâche métier.
5. **Distillation** (recommandation, non implémentée) : soft-labels
   d’un medium → nano pour compresser.

## 7. Mesures CPU locales (box 8 threads, PyTorch 2.14+cpu, eager)

Mesures réelles 2026-09-08 — détail daté dans **`docs/CPU_RUN_LOG.md`**.

### Smoke / benches courts

| Config | micro×seq | tps observé | ETA Chinchilla (20× params) |
|---|---|---|---|
| `smoke` (~0.3M) | 4×32 | **~870–1040** tok/s | n/a (sanity) |
| `nano` (~1.8M) | 2×64 | ~560 tok/s | bench smoke seulement |
| `commodity` (~13M) | 1×64 | **~145–170** tok/s | ~264M tokens ≈ **18–21 jours** CPU |

### Live nano pretrain (`runs/nano_base`, micro=4 seq=256 accum=2)

| Observation | Valeur |
|---|---|
| Target | **36,986,940** tokens (~20 tok/param, 1.86M trainable) |
| TPS early | **~1070–1230** tok/s (steps 10–350) |
| ETA trainer | **~8.3–9.6 h** wall (progressive-depth 1/3 ; ralentira en 2/3–3/3) |
| Loss early | ~10.83 → ~10.38 (steps 10→350) |
| Device log | `cpu` · `torch.set_num_threads(8)` |

Notes :
- Le decode matmul (`V×D`) domine dès que `vocab=50257` ; l’ancien broadcast OOM (~52 Go).
- `torch.compile` nécessite un compilateur C++ (`g++`/`clang`) ; sinon fallback eager (testé).
- Sur un GPU 8–12 Go, attendre ×20–100 vs ces tps CPU (ordre de grandeur, non mesuré ici).
- **Ne pas** confondre le smoke `2×64` (~560 tps) avec le live `4×256` (~1.1–1.2k tps).

## 8. Anti-patterns

- Lancer `launch_b1_rtx4090.sh` « pour voir » sur 1 GPU 8 Go.
- Activer `--enable-plasticity` + Adam sur CPU (double forward).
- Vocab GPT-2 + ancien decode (OOM) — corrigé, gardez le patch.
- Croire les ×8.14 des upgrades FP8/MoD/PGSU sur CPU : **FP8 = Ada/Hopper
  seulement** ; sur CPU seul progressive-depth + decode/engram fixes comptent.
- Committer `data/corpus.txt` (~109 MB) dans git — documenter + rebuild (`data/DATA.md`).

## 9. Vérifs rapides

```bash
python -m kuro_brain.smoke
python mini_train.py
python train_universal.py --data /tmp/kuro_test/tiny.txt \
  --output /tmp/kahnn_uni --config smoke --device cpu --smoke-steps 20
python teach.py smoke --device cpu
```

## 10. Après le prétrain : apprentissage à vie

Une fois la base langage+code obtenue (`nano` / `commodity` + corpus
texte+code), basculez en **continuous lifelong** :

- pas de decay global auto des engrammes (défaut) ;
- oubli **uniquement** via `teach.py forget` / `learner.forget(...)` ;
- `--soft-decay` optionnel (slots faibles seulement).

Voir **`docs/LIFELONG_LEARNING.md`** pour le curriculum, l’API teach/forget,
et les limites honnêtes du « se souvenir pour toujours ».

```bash
python teach.py teach --text "Fait métier…" --resume ./runs/.../ckpt_final.pt \
  --config nano --output ./runs/teach
python teach.py smoke --device cpu
```

## 11. Corpus (box)

Path : `data/corpus.txt` (~109 MiB, ~46M tokens GPT-2 estimés).  
Sources & rebuild : **`data/DATA.md`**. Le fichier binaire n’est **pas**
dans le dépôt ; reconstruire localement avant un long run.

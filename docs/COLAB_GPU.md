# Kahnn sur Google Colab (GPU free / Pro) — nano

Guide court pour Philippe-Antoine : entraîner **nano** sur Colab T4/L4
avec `train_universal.py`, sans toucher au train CPU local.

Notebook : [`notebooks/kahnn_nano_colab.ipynb`](../notebooks/kahnn_nano_colab.ipynb)

> **Attente honnête :** sur la box CPU (8 threads, 2026-09-08) nano tourne
> vers **~1.07–1.23k tok/s**. Un Colab **T4** devrait être **beaucoup plus
> rapide**, mais **on ne publie pas ici de chiffre GPU inventé** — lis le
> `tps=` du smoke (`--smoke-steps 20`) sur *ta* session Colab.

## 1. Ouvrir le notebook dans Colab

1. Va sur le repo : https://github.com/AFKmoney/kahnn
2. Ouvre `notebooks/kahnn_nano_colab.ipynb`
3. Clique **Open in Colab** (si le badge / bouton GitHub → Colab est dispo)
   **ou** dans Colab : *File → Open notebook → GitHub* →
   `AFKmoney/kahnn` → `notebooks/kahnn_nano_colab.ipynb`
4. **Runtime → Change runtime type → GPU**
   - Free : souvent **T4** (~15–16 Go)
   - Pro / Pro+ : parfois **L4**, A100, etc. (quota variable)

Lien direct (branche `main`) :

```text
https://colab.research.google.com/github/AFKmoney/kahnn/blob/main/notebooks/kahnn_nano_colab.ipynb
```

## 2. Free vs Pro (limites pratiques)

| | Free | Pro (ordre de grandeur) |
|--|------|-------------------------|
| GPU | T4 souvent | T4 / L4 / parfois mieux |
| Durée session | Coupures idle + plafond journalier | Sessions plus longues, moins de files |
| Disque `/content` | Éphémère | Idem — **Drive recommandé** pour les gros ckpt |
| Quota | Peut passer en CPU si saturé | Meilleur accès GPU |

Conseil : le notebook a **`USE_DRIVE=False` par défaut** pour éviter le
blocage d’auth interactive. Passe à `True` quand tu veux persister les
ckpt (~**846 MiB** chacun, mesure box).

## 3. Transférer un checkpoint depuis la box CPU

Sur la box (ex. run `runs/nano_base/`) tu as déjà des steps comme
`ckpt_1500.pt`, `ckpt_2000.pt`, …

1. Copie le fichier vers ton Google Drive, ex. :
   `MyDrive/kahnn/runs/from_cpu/ckpt_1500.pt`
   (Drive web, `rclone`, ou sync machine → Drive).
2. Dans le notebook, cellule **Resume** : mets
   `RESUME = "/content/drive/MyDrive/kahnn/runs/from_cpu/ckpt_1500.pt"`.
3. Lance le train avec `--device cuda` — le trainer reprend step + poids.

Même procédure dans l’autre sens : après Colab, copie
`MyDrive/kahnn/runs/nano_colab/ckpt_*.pt` vers la box si tu veux
`teach.py` / `generate.py` en local.

**Ne stoppe pas** un train CPU en cours juste pour tester Colab : copie un
ckpt déjà écrit (`ckpt_1500` ou plus tard) et continue sur GPU à côté.

## 4. Corpus

`data/corpus.txt` (~109 MiB) **n’est pas** dans git (voir `data/DATA.md`).

Options Colab (dans l’ordre du notebook) :

1. Copier depuis Drive si présent (`MyDrive/kahnn/data/corpus.txt`)
2. **Rebuild minimal** Gutenberg + sources repo (pas besoin de Drive)
3. Upload interactif **désactivé par défaut** (`DO_UPLOAD=False`) car
   `files.upload()` bloque la cellule

Pour le mix fidèle (TinyStories, Wikipedia, CPython, …) préfère le fichier
déjà construit sur la box.

## 5. Flags Colab sensés (nano / T4)

```bash
python train_universal.py \
  --data ./data/corpus.txt \
  --output ./runs/nano_colab \
  --config nano --device cuda \
  --micro-batch 8 --seq-len 256 --grad-accum 2 \
  --log-every 10 --checkpoint-every 500
```

- **Progressive depth** : déjà ON par défaut
- Si **OOM** : `--micro-batch 4`, `--mod`, `--activation-checkpointing`
- Smoke débit : ajoute `--smoke-steps 20` et lis `loss=` / `tps=`
- Optionnel : `bitsandbytes` (install notebook) — nice-to-have, pas requis
  pour nano

## 6. Après le prétrain

```bash
python teach.py teach \
  --text "La capitale du Canada est Ottawa." \
  --resume ./runs/nano_colab/ckpt_final.pt \
  --config nano --device cuda \
  --output ./runs/teach_colab
```

Détails lifelong : `docs/LIFELONG_LEARNING.md`.

## 7. Troubleshooting (Colab « ne marche pas »)

| Symptôme | Cause fréquente | Fix |
|----------|-----------------|-----|
| Cellule 0 stop / `cuda_available=False` | Pas de GPU / quota free épuisé | Runtime → GPU ; attendre ; autre compte ; Colab Pro |
| `drive.mount` bloque / auth interactive | Colab demande un clic OAuth | Garder `USE_DRIVE=False` ; train sur `/content` ; télécharge les ckpt à la main |
| `git clone` 404 / auth | Branche fausse ou fork privé | Repo **public** `AFKmoney/kahnn` ; `BRANCH="main"` ; fork privé → URL avec token ou zip |
| Cellule upload qui ne finit jamais | `files.upload()` attend un fichier | `DO_UPLOAD=False` + cellule rebuild Gutenberg |
| `%pip` / magics fragiles | Environnement shell vs Python | Notebook utilise `subprocess` + `sys.executable -m pip` |
| OOM T4 | Batch trop gros | `--micro-batch 4`, `--mod`, `--activation-checkpointing` |
| `ValueError: disallowed special token` (EOT) | Corpus contient le marqueur EOT GPT-2 | Corrigé dans `data.encode_text` (`encode_ordinary`) — tire le dernier `main` |
| Session morte en cours de train | Idle / déconnexion free | Remonter Drive ou re-uploader le dernier `ckpt_*.pt`, `RESUME=...` |

## 8. Ce que Colab ne remplace pas

- Ce n’est **pas** un cluster B1 / multi-4090
- Free Colab peut te **remettre en CPU** sans prévenir — vérifie `nvidia-smi`
  au début de chaque session
- Les chiffres GPU **changent** selon T4 vs L4, charge du datacenter, et
  depth progressive (1/3 → full) — mesure toujours localement

Voir aussi : `docs/UNIVERSAL_TRAINING.md`, `docs/CPU_RUN_LOG.md`.

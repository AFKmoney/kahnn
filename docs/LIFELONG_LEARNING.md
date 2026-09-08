# Apprentissage à vie (lifelong) — Kahnn

Objectif utilisateur : **après une base langage + code**, Kahnn continue
d’apprendre **sans full retrain** et **sans oubli catastrophique**. Une
connaissance consolidée reste jusqu’à ce que l’utilisateur demande
explicitement de l’oublier.

## 1. Ce que ça fait (et ce que ça ne fait pas)

| Oui | Non |
|---|---|
| Consolider des faits / docs courts dans les **engrammes** | Remplacer un vrai prétrain langage+code |
| Plasticité locale (couplages U/V) en mode continuous | Garantir une génération fluide sans base corpus |
| Mémoire stable (pas de decay global auto) | Capacité infinie (slots finis → eviction des *moins utilisés*) |
| `forget` / `forget_slot` / `forget_all` explicites | Oublier tout seul « pour faire de la place » en lifelong |

**Honnêteté :** « comprend le langage et le code » exige encore un
**corpus de base** (texte + code) via `train_universal.py` (Adam,
configs `nano` / `commodity`, tokens ~Chinchilla). Le mode continuous
**ajoute** de la connaissance durable ; il ne crée pas l’alphabet ni
la syntaxe à partir de zéro.

## 2. Curriculum recommandé

```
1) Prétrain base (Adam)
   train_universal.py --config nano|commodity --data ./mix_text_code/
   → ckpt_final.pt

2) Basculer continuous (lifelong)
   train_universal.py --continuous --resume ckpt_final.pt --data ./domaine/
   OU teach.py teach --text "..." --resume ckpt_final.pt

3) Oublier seulement sur demande
   teach.py forget --text "..." --resume ...
```

Mélange texte+code suggéré pour l’étape 1 (exemples, pas un dump magique) :

- texte : Wikipedia / livres / docs produit (propre > massif)
- code : fichiers `.py` / `.ts` / README du dépôt, The Stack subset, etc.
- ratio indicatif : **50–70 % texte, 30–50 % code** pour un assistant code-aware

## 3. Mémoire stable (changement clé)

Avant : en `continuous_mode`, chaque step appelait
`memory.decay(rate=0.9995)` → les compteurs d’usage fondaient → les
slots consolidés redevenaient évincables (**contredit** « se souvenir
pour toujours »).

Maintenant :

- **Défaut lifelong :** aucun decay automatique.
- **`--soft-decay` (opt-in) :** n’age que les slots *faibles*
  (`usage < seuil`), pour libérer de la capacité sans toucher aux
  souvenirs forts.
- L’éviction à l’écriture reste `argmin(usage)` : les slots souvent
  renforcés restent protégés.

## 4. API forget / teach

### Python

```python
from kuro_brain.continuous_learning import OnlineLearner

learner = OnlineLearner(model, cfg, continuous_mode=True)  # soft_decay=False

# Enseigner un fait (passe forward + plasticité + force consolidate)
out = learner.teach(token_ids, n_passes=3)
print(out["probe"]["best_memory"], out["consolidate"]["occupied"])

# Vérifier le rappel (cohérence de phase vs engrammes)
probe = learner.probe(token_ids)

# Oublier explicitement
learner.forget(query_ids=token_ids, min_coherence=0.45)
learner.forget_slot(layer=0, slot=3)
learner.forget_all()
```

### CLI (`teach.py`)

```bash
python teach.py teach --text "La capitale du Canada est Ottawa." \
  --config commodity --resume ./runs/commodity/ckpt_final.pt \
  --output ./runs/teach

python teach.py probe --text "capitale du Canada" \
  --resume ./runs/teach/ckpt_teach.pt --config commodity

python teach.py forget --text "La capitale du Canada est Ottawa." \
  --resume ./runs/teach/ckpt_teach.pt --config commodity --output ./runs/teach

python teach.py smoke --device cpu   # régression synthétique
```

`train_universal.py --continuous` accepte aussi `--soft-decay`.

## 5. Limites honnêtes du « no forgetting »

1. **Engrammes ≠ poids fondation.** La base Adam (embeddings décodage,
   ω, K, MLP…) peut encore dériver légèrement via plasticité locale ;
   seuls les *attracteurs consolidés* sont explicitement « sticky ».
2. **Capacité finie.** `cross_layer_memory_capacity` et
   `n_ensembles_per_layer` bornent le nombre de souvenirs. Sans
   soft-decay, les slots à plus faible usage sont écrasés en dernier
   recours quand tout est plein.
3. **Rappel ≠ génération parfaite.** `probe` mesure la cohérence de
   phase ; un modèle sans base langagière peut « stocker » un fait sans
   le reformuler correctement.
4. **Seuil `min_coherence`.** Un forget trop strict rate la cible ; trop
   lâche efface des voisins. Ajuster selon les probes.
5. **Pas de replay.** C’est voulu — donc pas de filet si vous
   `forget_all` par erreur : rechargez un checkpoint.

## 6. Vérifs

```bash
python -m kuro_brain.smoke
python teach.py smoke --device cpu
python train_universal.py --data /tmp/tiny.txt --output /tmp/u \
  --config smoke --device cpu --continuous --smoke-steps 5
```

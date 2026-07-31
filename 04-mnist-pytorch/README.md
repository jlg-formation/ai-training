# TP 4 — Reconnaissance de chiffres MNIST (première fois en PyTorch)

## Objectif

Refaire le réseau du **TP 3** (une couche cachée + ReLU), mais cette fois **en
PyTorch** au lieu de NumPy « à la main ». La nouveauté du TP n'est **pas**
l'architecture : c'est **l'outil**.

Aux TP 1 à 3, on a écrit soi-même la passe avant, la perte et surtout la
**rétropropagation** (les gradients dérivés un à un). C'était le but : comprendre
la mécanique. Maintenant qu'on l'a comprise, on la confie à un **framework**. La
rétropropagation qu'on a souffert à dériver au TP 3 devient **une seule ligne** :
`loss.backward()`.

Le problème est le prolongement direct du XOR :

```
TP 3 :   2 entrées   -> couche cachée (ReLU) ->  1 sortie   (2 classes)
TP 4 :  784 entrées  -> couche cachée (ReLU) -> 10 sorties  (10 classes)
```

Une image MNIST fait **28 × 28 = 784** pixels en niveaux de gris ; il y a **10**
chiffres (0 à 9). C'est le **même MLP** qu'au TP 3, juste plus large.

## De NumPy à PyTorch : ce qui change vraiment

| TP 1 → 3 (NumPy, à la main)          | TP 4 (PyTorch)                          |
| ------------------------------------ | --------------------------------------- |
| `avant()` écrit à la main            | `forward()` d'un `nn.Module`            |
| `retropropagation()` dérivée à la main | `loss.backward()` (**autograd**)      |
| `W -= lr * grad`                     | `optimizer.step()`                      |
| tout le dataset d'un coup            | des **mini-batchs** via un `DataLoader` |
| sigmoïde + BCE (2 classes)           | softmax + cross-entropy (10 classes)    |
| CPU implicite                        | `.to(device)` (CPU par défaut, GPU si dispo) |

L'idée à retenir : **PyTorch ne fait rien de magique**. Il automatise exactement
ce qu'on a codé à la main jusqu'ici. `nn.Linear`, c'est le `X @ W + b` du TP 3 ;
`SGD`, c'est la descente de gradient ; `backward()`, c'est la rétropropagation.

## Les notions nouvelles

- **Tenseur** : le cousin PyTorch du `ndarray` NumPy, mais qui sait calculer ses
  gradients (autograd) et vivre sur GPU.
- **`nn.Module`** : une classe qui regroupe les couches (`__init__`) et
  l'enchaînement (`forward`). PyTorch crée et suit les poids tout seul.
- **Autograd** : en enregistrant les opérations de la passe avant, PyTorch sait
  calculer **tous** les gradients d'un coup avec `loss.backward()`.
- **Optimiseur** (`torch.optim.SGD`) : encapsule la mise à jour des poids.
- **`Dataset` + `DataLoader`** : chargent les données et les découpent en
  **mini-batchs** (avec mélange à chaque epoch).
- **Batch / epoch** : une **epoch** = un passage complet sur les données ; on la
  parcourt **batch par batch** (`BATCH_SIZE` images à la fois).
- **`device`** : `"cpu"` ou `"cuda"`. Le modèle et les données doivent être sur
  le **même** device (`.to(device)`).

### softmax + cross-entropy : la sortie à 10 classes

Au TP 2, une sortie binaire passait par la **sigmoïde** (une probabilité) et la
perte **BCE**. Avec **10** classes, on généralise :

- la **softmax** transforme les 10 scores bruts (logits) en 10 probabilités qui
  somment à 1 ;
- l'**entropie croisée** (cross-entropy) mesure l'écart avec la bonne classe.

`nn.CrossEntropyLoss` fait les **deux d'un coup**. La dérivation (et le lien avec
la sigmoïde + BCE) est dans [MATH.md](MATH.md).

> **⚠️ Piège classique.** `nn.CrossEntropyLoss` attend les **logits bruts** : elle
> applique la softmax **elle-même**. La couche de sortie ne doit donc **pas**
> avoir d'activation (pas de softmax dans `forward`). En mettre une reviendrait à
> l'appliquer deux fois.

## Installation de PyTorch

PyTorch a été ajouté aux dépendances **partagées** de la racine (jamais dans le
dossier du TP). Si ce n'est pas déjà fait :

```powershell
# à la racine du dépôt
uv add torch torchvision
```

Par défaut, `uv` installe le **build CPU** de PyTorch : il fonctionne partout et
suffit largement pour ce petit MLP (quelques secondes par epoch). Le TP 9 sera
consacré à la comparaison CPU / GPU / consommation.

> **Tu as un GPU NVIDIA ?** Le code le détecte automatiquement
> (`torch.cuda.is_available()`). Pour l'exploiter, il faut installer le build
> **CUDA** de PyTorch (voir [pytorch.org](https://pytorch.org)) — pas nécessaire
> pour réussir ce TP.

## Les données MNIST

`torchvision` **télécharge** MNIST la **première fois** (~11 Mo) dans le dossier
`data/` (ignoré par git), puis le relit depuis le disque. Le split
**train (60 000) / test (10 000)** est **officiel** : plus besoin de le découper
soi-même comme aux TP 2 et 3.

## Lancer le TP (avec uv)

Depuis ce dossier :

```powershell
uv run main.py
```

Ou, depuis n'importe où dans le projet :

```powershell
mise run tp4
```

## Ce que tu dois observer

- La **loss diminue** sur l'entraînement **et** sur le test.
- L'**accuracy test atteint ~97 %** en seulement quelques epochs (ce simple MLP
  est déjà très bon ; le CNN du TP 6 fera encore mieux).
- Le **nombre de paramètres** s'affiche (~100 000) : c'est la **première fois**
  qu'on compte les paramètres d'un modèle.
- Le **temps par epoch** est affiché : de quoi comparer CPU et GPU (TP 9).
- Dans `resultat.png` : des chiffres avec leur **prédiction** (vert = correct,
  rouge = erreur), la **courbe de loss**, et une **matrice de confusion** 10 × 10
  qui montre les confusions typiques (ex. **4 ↔ 9**, **3 ↔ 5**).

## À expérimenter

- **`BATCH_SIZE`** : passe de `64` à `8` puis `512`. Un petit batch rend la loss
  plus « bruitée » mais met à jour les poids plus souvent ; un gros batch est
  plus lisse et plus rapide par epoch. C'est la notion clé du TP.
- **`EPOCHS`** : plus d'epochs améliorent-elles encore l'accuracy, ou plafonne-t-elle ?
- **`H`** : moins de neurones cachés (`32`) ou plus (`256`) — quel effet sur
  l'accuracy et sur le nombre de paramètres ?
- **`LEARNING_RATE`** : trop grand, la loss explose ; trop petit, ça apprend
  lentement. Cherche le bon compromis.
- Remplace `SGD` par `torch.optim.Adam` (avec `lr=1e-3`) : converge-t-il plus vite ?

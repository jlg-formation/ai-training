# TP 6 — Un vrai CNN sur MNIST (la convolution) + reconnaissance dans le navigateur

## Objectif

Au **TP 4**, on reconnaissait déjà les chiffres MNIST, mais avec un **MLP** : on
_aplatissait_ chaque image 28×28 en un vecteur de 784 nombres. Ce faisant, on
jetait la **structure spatiale** de l'image (quel pixel est voisin de quel
autre).

**La notion nouvelle de ce TP : la convolution** (`nn.Conv2d`). Au lieu d'un
poids par pixel, on apprend de petits **filtres** 3×3 qu'on fait _glisser_ sur
toute l'image. Un même filtre détecte le même motif (un bord, un coin, une
boucle) **partout** dans l'image : c'est le **partage de poids**. En empilant ces
couches, le réseau détecte des motifs de plus en plus complexes.

En bonus applicatif, on **exporte le modèle entraîné au format ONNX** et on
construit un petit **site web** (sans serveur) qui charge ce modèle et reconnaît
un chiffre **dessiné à la souris**, directement dans le navigateur.

## Ce qui change par rapport au TP 4

| TP 4 (MLP)                              | TP 6 (CNN)                                   |
| --------------------------------------- | -------------------------------------------- |
| image → **aplatie** en 784              | image gardée en 2D (1, 28, 28)               |
| `nn.Linear` uniquement                  | `nn.Conv2d` + `MaxPool2d` puis `nn.Linear`   |
| un poids par pixel                      | des **filtres** partagés sur toute l'image   |
| pas de régularisation                   | **BatchNorm** + **Dropout**                  |
| optimiseur SGD                          | optimiseur **Adam**                          |
| pas de sauvegarde                       | **export ONNX** pour le navigateur           |

Le reste (perte cross-entropy sur les logits, boucle d'entraînement, `.to(device)`,
mini-batchs) est **identique au TP 4** : on ne change que le modèle.

## Topologie du réseau

```
entrée (1, 28, 28)
  ── Bloc 1 ──────────────────────────────────
  Conv2d(1 → 32, 3×3, padding 1)  → (32, 28, 28)
  BatchNorm + ReLU
  Conv2d(32 → 64, 3×3, padding 1) → (64, 28, 28)
  BatchNorm + ReLU
  MaxPool 2×2                     → (64, 14, 14)
  Dropout(0.25)
  ── Bloc 2 ──────────────────────────────────
  Conv2d(64 → 128, 3×3, padding 1)→ (128, 14, 14)
  BatchNorm + ReLU
  MaxPool 2×2                     → (128, 7, 7)
  Dropout(0.25)
  ── Classifieur ─────────────────────────────
  Flatten                        → 6272
  Linear(6272 → 128) + ReLU + Dropout(0.5)
  Linear(128 → 10)               → 10 logits
```

`padding=1` avec un noyau 3×3 **conserve** la taille (28×28 reste 28×28) : ce
sont les `MaxPool` qui réduisent la résolution.

## Lancer l'entraînement (Python)

Depuis la racine du dépôt :

```
mise run tp6
```

ou depuis ce dossier : `uv run main.py`.

Le script télécharge MNIST dans le `data/` **partagé à la racine** (déjà présent
si tu as fait le TP 4 ou 7), entraîne le CNN (GPU si disponible,
sinon CPU), affiche la précision, puis :

- enregistre les courbes et la matrice de confusion dans **`resultat.png`** ;
- exporte le modèle en **`modele.onnx`** ;
- copie ce modèle dans **`front/public/modele.onnx`** pour le site.

## Lancer le site de reconnaissance (front)

Le front utilise **Bun**, **Vite** et **TypeScript**, avec `onnxruntime-web`
pour l'inférence dans le navigateur.

```
mise run tp6-site
```

ou manuellement depuis ce dossier :

```
cd front
bun install
bun run dev
```

Ouvre l'URL affichée, **dessine un chiffre à la souris**, et le modèle prédit le
chiffre avec la probabilité de chaque classe (0 à 9). Le bouton _Effacer_ remet
le canvas à zéro.

> Le site a besoin de `front/public/modele.onnx`, donc lance d'abord
> l'entraînement (`mise run tp6`) au moins une fois.

### Prétraitement « façon MNIST »

Un CNN entraîné sur MNIST n'a vu que des images très particulières : chiffre
**blanc sur fond noir**, mis à l'échelle dans une boîte 20×20 puis **centré par
son centre de masse** dans une image 28×28. Le front applique **exactement** la
même recette au chiffre dessiné (recadrage, mise à l'échelle, centrage,
normalisation `(x - 0.1307) / 0.3081`). C'est indispensable : sans ce
prétraitement, le modèle se trompe, même s'il est excellent sur MNIST.

## Ce que tu dois observer

- La précision de test dépasse **99 %** — nettement mieux que le MLP du TP 4,
  pour un nombre de paramètres comparable : la convolution _exploite_ la
  structure de l'image au lieu de la jeter.
- Sur `resultat.png`, la **matrice de confusion** est très diagonale ; les rares
  confusions restantes sont « logiques » (4↔9, 3↔5…).
- Dans le navigateur, un chiffre bien centré et bien tracé est reconnu avec une
  probabilité proche de 100 %.

## À expérimenter

- **Enlève la `BatchNorm`** (ou le `Dropout`) et observe l'effet sur la vitesse
  d'apprentissage et la précision.
- **Réduis à un seul bloc de convolution** : combien perd-on en précision ?
- **Change `EPOCHS`, `LEARNING_RATE`, `BATCH_SIZE`** en haut de `main.py`.
- Dans le front, **dessine mal exprès** (chiffre décentré, trait très fin) pour
  voir l'importance du prétraitement et des probabilités affichées.

# Les maths du GAN : un jeu à deux joueurs (minimax)

Ce document explique la **seule** vraie nouveauté mathématique du TP 7 :
l'entraînement **adversaire**. La brique de base, elle, est **déjà connue** — la
perte **BCE** (entropie croisée binaire) du TP 2. Ce qui est neuf, c'est qu'on
l'utilise pour faire **jouer deux réseaux l'un contre l'autre**.

> Les conventions de notation (vecteurs **lignes** en gras minuscule, matrices en
> gras MAJUSCULE, scalaires en maigre) sont décrites une fois pour toutes dans le
> [README.md § Conventions mathématiques](../README.md#conventions-mathématiques).

---

## 1) Les deux réseaux et leurs notations

- Le **générateur** $G$ prend un **vecteur ligne de bruit**
  $\mathbf{z}$ (taille $1 \times 64$), tiré au hasard, et renvoie une **image**
  $G(\mathbf{z})$ (784 pixels, remis en 28×28). Ses poids sont notés
  $\boldsymbol{\theta}_G$.
- Le **discriminateur** $D$ prend une **image** $\mathbf{x}$ (784 pixels) et
  renvoie un **scalaire** $D(\mathbf{x}) \in [0, 1]$ : la **probabilité estimée
  que l'image soit vraie**. Ses poids sont notés $\boldsymbol{\theta}_D$.

$D$ est donc un **classifieur binaire** exactement comme au TP 2 : il sort un
**logit** $s$, et la sigmoïde le transforme en probabilité

$$
D(\mathbf{x}) = \sigma(s) = \frac{1}{1 + e^{-s}} \in [0, 1].
$$

Dans le code, la dernière couche de `Discriminateur` sort le **logit brut** $s$,
et c'est `BCEWithLogitsLoss` qui applique la sigmoïde (comme
`CrossEntropyLoss` applique le softmax aux TP 4/6).

---

## 2) Rappel : la perte BCE (entropie croisée binaire)

Pour **une** image de logit $s$ (donc de probabilité prédite $p = \sigma(s)$) et
de vraie étiquette $y \in \{0, 1\}$, la BCE vaut

$$
\ell(p, y) = -\big[\, y \log p + (1 - y)\log(1 - p) \,\big].
$$

- Si $y = 1$ (« vrai »), il reste $-\log p$ : la perte est faible quand $p$ est
  proche de 1.
- Si $y = 0$ (« faux »), il reste $-\log(1 - p)$ : la perte est faible quand $p$
  est proche de 0.

C'est **exactement** la perte du TP 2. Tout le TP 7 se construit avec cette
brique, appliquée tantôt à des vraies images, tantôt à des fausses.

---

## 3) Le jeu à deux joueurs (la fonction de valeur)

On note $p_{\text{data}}$ la distribution des **vraies** images MNIST et
$p_{\mathbf{z}}$ celle du **bruit**. Le GAN se résume à **une seule** fonction,
que $D$ veut **maximiser** et $G$ veut **minimiser** :

$$
\min_{G}\ \max_{D}\ V(D, G) =
\mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}\big[\log D(\mathbf{x})\big]
\;+\;
\mathbb{E}_{\mathbf{z} \sim p_{\mathbf{z}}}\big[\log\big(1 - D(G(\mathbf{z}))\big)\big].
$$

Lisons les deux termes :

- $\log D(\mathbf{x})$ : $D$ veut dire **1** sur les **vraies** images (les
  reconnaître comme vraies).
- $\log\big(1 - D(G(\mathbf{z}))\big)$ : $D$ veut dire **0** sur les **fausses**
  (les démasquer). $G$, lui, veut le **contraire** : que $D(G(\mathbf{z}))$ soit
  proche de **1**.

C'est un **jeu à somme nulle** : ce qui est bon pour $D$ est mauvais pour $G$, et
réciproquement. D'où le nom **adversaire**.

---

## 4) Ce que chaque réseau minimise, concrètement

En pratique on **n'optimise pas** $V(D, G)$ directement : on découpe en **deux
pertes BCE**, une par réseau, et on fait **deux pas de gradient** par batch.

### Pas du discriminateur

$D$ veut classer les vraies en 1 et les fausses en 0. Sa perte est la **somme de
deux BCE** (moyennées sur le batch de $N$ images) :

$$
\mathcal{L}_D =
\underbrace{-\frac{1}{N}\sum_{i} \log D(\mathbf{x}_i)}_{\text{vraies, cible } 1}
\;+\;
\underbrace{-\frac{1}{N}\sum_{i} \log\big(1 - D(G(\mathbf{z}_i))\big)}_{\text{fausses, cible } 0}.
$$

Minimiser $\mathcal{L}_D$ revient exactement à **maximiser** $V$ par rapport à
$D$. Dans le code :

```python
perte_sur_vraies = perte(discriminateur(images_reelles), cible_vrai)  # cible 1
perte_sur_fausses = perte(
    discriminateur(images_fausses.detach()), cible_faux
)  # cible 0
perte_d = perte_sur_vraies + perte_sur_fausses
```

Le `.detach()` **coupe le lien** vers $G$ : ce pas ne met à jour que
$\boldsymbol{\theta}_D$, jamais $\boldsymbol{\theta}_G$.

### Pas du générateur (version « non saturante »)

En théorie, $G$ devrait **minimiser** $\log\big(1 - D(G(\mathbf{z}))\big)$. Mais
au début $D$ démasque facilement les faux ($D(G(\mathbf{z})) \approx 0$), et là
ce terme a un **gradient minuscule** : $G$ n'apprend presque rien (problème de
**saturation**).

L'astuce de Goodfellow : au lieu de **minimiser** $\log(1 - D(G(\mathbf{z})))$,
on **maximise** $\log D(G(\mathbf{z}))$. Cela revient à réutiliser la **même
BCE**, mais en donnant aux fausses images la cible **1** (« fais-les passer pour
vraies ») :

$$
\mathcal{L}_G = -\frac{1}{N}\sum_{i} \log D\big(G(\mathbf{z}_i)\big)
\qquad\text{(cible } 1\text{).}
$$

Dans le code :

```python
perte_g = perte(discriminateur(images_fausses), cible_vrai)  # cible 1, SANS detach
```

Cette fois **pas de `detach`** : le gradient traverse $D$ (dont les poids sont
figés pendant ce pas) et remonte jusque dans $\boldsymbol{\theta}_G$.

---

## 5) L'équilibre : pourquoi les pertes ne tombent pas à zéro

Aux TP précédents, une perte qui descend vers 0 = « le modèle a gagné ». Ici,
**les deux réseaux tirent en sens opposé**, donc aucune perte ne s'annule
durablement. L'**équilibre** recherché (l'optimum théorique du jeu) est atteint
quand $G$ reproduit parfaitement la distribution des vraies images,
$p_G = p_{\text{data}}$. À ce point, $D$ ne peut plus faire mieux que **tirer à
pile ou face** :

$$
D(\mathbf{x}) = \tfrac{1}{2} \quad\text{partout,}
$$

soit une perte BCE par terme de $-\log \tfrac{1}{2} = \log 2 \approx 0{,}69$.
C'est pourquoi, sur `resultat.png`, une valeur de perte qui **oscille autour de
$0{,}69$** (et non qui plonge vers 0) est le **signe d'un GAN sain**, et non d'un
échec.

---

## 6) Résumé

| | discriminateur $D$ | générateur $G$ |
| --- | --- | --- |
| entrée | image $\mathbf{x}$ (784) | bruit $\mathbf{z}$ (64) |
| sortie | logit → $D(\mathbf{x}) \in [0,1]$ | image $G(\mathbf{z})$ (784) |
| cible des **vraies** | 1 | — |
| cible des **fausses** | 0 | **1** (les faire passer pour vraies) |
| met à jour | $\boldsymbol{\theta}_D$ (fausses `detach`ées) | $\boldsymbol{\theta}_G$ (à travers $D$ figé) |
| perte | $\mathcal{L}_D$ (somme de 2 BCE) | $\mathcal{L}_G$ (1 BCE, cible 1) |

Une seule brique nouvelle à retenir : **deux BCE, deux optimiseurs, un pas
chacun par batch** — le reste est le TP 2 rejoué à deux.

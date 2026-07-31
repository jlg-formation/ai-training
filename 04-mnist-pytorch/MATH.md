# Les maths de la sortie à 10 classes (softmax + entropie croisée)

Ce document explique la **seule** vraie nouveauté mathématique du TP 4 : passer
d'une sortie **binaire** (2 classes, TP 2) à une sortie **multi-classe** (10
chiffres). La bonne nouvelle : softmax + entropie croisée est la
**généralisation** directe de sigmoïde + BCE, et le gradient se simplifie encore
une fois en

```python
erreur = p - y
```

exactement comme aux TP 2 et 3. La **rétropropagation** à travers la couche
cachée, elle, n'est plus à dériver : c'est **autograd** (`loss.backward()`) qui
s'en charge.

> Les conventions de notation (vecteurs **lignes** en gras minuscule, matrices en
> gras MAJUSCULE, scalaires en maigre) sont décrites une fois pour toutes dans le
> [README.md § Conventions mathématiques](../README.md#conventions-mathématiques).

---

## 1) La sortie du modèle : $K = 10$ scores bruts

Le réseau se termine par **$K = 10$ neurones** de sortie (un par chiffre). Pour
une image, ils produisent un **vecteur ligne de logits**
$\mathbf{z} = (z_1, \dots, z_K)$ (taille $1 \times K$), où chaque $z_k$ est un
**score brut** (un réel non borné) en faveur de la classe $k$. C'est le pendant
du **logit unique** $z$ du TP 2, mais il y en a maintenant un **par classe**.

Dans le code, ces logits sortent directement de la dernière couche `nn.Linear`,
**sans activation** :

```python
return self.sortie(x)      # logits bruts (N, 10)
```

---

## 2) La softmax : des logits aux probabilités

La **softmax** transforme les $K$ logits en $K$ **probabilités**
$\mathbf{p} = (p_1, \dots, p_K)$ (taille $1 \times K$) qui sont positives et
**somment à 1** :

$$
p_k = \operatorname{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\displaystyle\sum_{j=1}^{K} e^{z_j}}
\qquad k = 1, \dots, K
$$

- Le **numérateur** $e^{z_k}$ rend tout positif et amplifie les grands scores.
- Le **dénominateur** (la somme sur **toutes** les classes) normalise pour que
  $\sum_k p_k = 1$.

$p_k$ se lit « probabilité que l'image soit le chiffre $k$ ». On prédit la classe
au **plus gros** $p_k$ (donc au plus gros $z_k$) — c'est le `argmax` du code.

---

## 3) La vraie étiquette : un indice ou un vecteur « one-hot »

La vraie classe est un **indice** $c \in \{1, \dots, K\}$ (le chiffre écrit sur
l'image). On la représente aussi par un **vecteur ligne one-hot**
$\mathbf{y} = (y_1, \dots, y_K)$ (taille $1 \times K$), qui vaut $1$ à la bonne
classe et $0$ partout ailleurs :

$$
y_k = \begin{cases} 1 & \text{si } k = c \\ 0 & \text{sinon} \end{cases}
$$

Dans le code, PyTorch prend directement l'**indice** $c$ (les cibles sont des
entiers de 0 à 9) ; le one-hot n'est qu'un outil pour écrire les maths.

---

## 4) L'entropie croisée : la perte multi-classe

L'**entropie croisée** (*cross-entropy*) compare les probabilités prédites
$\mathbf{p}$ à la vraie classe. Pour **un** exemple :

$$
L = -\sum_{k=1}^{K} y_k \log(p_k) = -\log(p_c)
$$

La somme s'effondre car $\mathbf{y}$ est one-hot : **un seul** terme survit,
celui de la bonne classe $c$. La perte vaut donc simplement $-\log(p_c)$ : elle
est **petite** si le modèle donne une forte probabilité à la bonne classe
($p_c \to 1$), **énorme** s'il se trompe avec confiance ($p_c \to 0$). C'est
exactement l'esprit de la BCE du TP 2.

Sur un **batch** de $N$ images, la perte est la **moyenne** des pertes par
exemple (comme aux TP précédents) :

$$
L_{\text{batch}} = \frac{1}{N} \sum_{i=1}^{N} L_i
= \frac{1}{N} \sum_{i=1}^{N} \big(-\log p_{i,c_i}\big)
$$

où $c_i$ est la vraie classe de l'exemple $i$ et $p_{i,c_i}$ sa probabilité
prédite. Dans le code, `nn.CrossEntropyLoss` calcule softmax **et** entropie
croisée en une seule opération, directement à partir des **logits** :

```python
perte = nn.CrossEntropyLoss()
p = perte(logits, cibles)      # logits (N, 10), cibles = indices (N,)
```

---

## 5) Le gradient se simplifie (encore) en $p - y$

Comme aux TP 2 et 3, le gradient de la perte par rapport aux **logits** se
simplifie remarquablement. Pour un exemple, composante par composante :

$$
\frac{\partial L}{\partial z_k} = p_k - y_k
$$

soit, en vecteur ligne (taille $1 \times K$) :

$$
\frac{\partial L}{\partial \mathbf{z}} = \mathbf{p} - \mathbf{y}
$$

C'est **le même** $\mathbf{p} - \mathbf{y}$ que depuis le TP 2 : la probabilité
prédite moins la cible. Toute la magie de la softmax et de l'entropie croisée
tient dans cette formule d'une simplicité trompeuse.

> **On ne code plus ce gradient.** Au TP 3, on partait de $p - y$ pour
> **remonter** à la main dans la couche cachée. Ici, PyTorch **enregistre** la
> passe avant et calcule tout seul, avec `loss.backward()`, les gradients de
> **tous** les poids (couche cachée comprise). C'est ça, autograd.

---

## 6) Pourquoi c'est bien la généralisation de la sigmoïde + BCE

Avec **$K = 2$ classes**, softmax + entropie croisée **redevient** exactement
sigmoïde + BCE. En effet, pour deux logits $z_1$ et $z_2$ :

$$
p_2 = \frac{e^{z_2}}{e^{z_1} + e^{z_2}}
    = \frac{1}{1 + e^{-(z_2 - z_1)}}
    = \sigma(z_2 - z_1)
$$

On retrouve la **sigmoïde**, appliquée à la **différence** des deux scores. Et
l'entropie croisée à deux classes est **littéralement** la BCE du TP 2. La sortie
binaire n'était donc qu'un **cas particulier** de la sortie multi-classe.

| TP 2 (2 classes)        | TP 4 (K classes)                    |
| ----------------------- | ----------------------------------- |
| sigmoïde $\sigma(z)$    | softmax $\operatorname{softmax}(\mathbf{z})$ |
| perte BCE               | entropie croisée                    |
| gradient $p - y$        | gradient $\mathbf{p} - \mathbf{y}$  |

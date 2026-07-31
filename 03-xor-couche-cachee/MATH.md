# Les maths de la couche cachée (rétropropagation + ReLU)

Ce document explique **pas à pas** comment on calcule les gradients d'un réseau
à **deux couches**. La grande idée : on part de l'erreur de sortie $p - y$ (déjà
vue au TP 2) et on la fait **remonter** couche par couche avec la **règle de la
chaîne**. C'est la **rétropropagation** (*backpropagation*).

---

## 1) Le réseau, couche par couche

Une entrée est un point $x = (x_1, x_2)$. Le réseau enchaîne deux couches.

**Couche cachée** ($H$ neurones), avec les poids $W^{(1)}$ (taille $2 \times H$)
et les biais $b^{(1)}$ (taille $H$) :

$$
z^{(1)} = x\,W^{(1)} + b^{(1)}
\qquad
a^{(1)} = \operatorname{ReLU}\!\big(z^{(1)}\big)
$$

**Couche de sortie** (1 neurone), avec $W^{(2)}$ (taille $H \times 1$) et
$b^{(2)}$ :

$$
z^{(2)} = a^{(1)} W^{(2)} + b^{(2)}
\qquad
p = \sigma\!\big(z^{(2)}\big)
$$

$p$ est la probabilité de la classe $1$, et $y \in \{0, 1\}$ la vraie étiquette.
La perte est la **même BCE** qu'au TP 2 :

$$
L = -\Big[\, y \log(p) + (1 - y)\log(1 - p) \,\Big]
$$

---

## 2) La fonction ReLU et sa dérivée

$$
\operatorname{ReLU}(z) = \max(0,\, z)
$$

Sa dérivée est une **marche d'escalier** : elle vaut $1$ là où $z$ est positif,
$0$ ailleurs (elle laisse passer le gradient, ou le coupe).

$$
\operatorname{ReLU}'(z) =
\begin{cases}
1 & \text{si } z > 0 \\
0 & \text{si } z \le 0
\end{cases}
$$

Dans le code, cela s'écrit simplement `(z1 > 0)`, un tableau de `True`/`False`
que NumPy traite comme des `1`/`0`.

---

## 3) La règle de la chaîne : remonter le gradient

On veut $\dfrac{\partial L}{\partial W^{(1)}}$, $\dfrac{\partial L}{\partial b^{(1)}}$,
$\dfrac{\partial L}{\partial W^{(2)}}$ et $\dfrac{\partial L}{\partial b^{(2)}}$.
On les obtient en descendant le long du réseau, du bout (la perte) vers le
début (l'entrée). À chaque étape on réutilise le résultat de l'étape
précédente : c'est ça, la rétropropagation.

### Étape a — l'erreur de sortie (déjà connue !)

Exactement comme au TP 2 (sigmoïde + BCE), le gradient par rapport au score de
sortie $z^{(2)}$ se simplifie en :

$$
\frac{\partial L}{\partial z^{(2)}} = p - y
\;\equiv\; \delta^{(2)}
$$

On note ce terme $\delta^{(2)}$ (« l'erreur de la couche de sortie »). La
dérivation détaillée est dans le [MATH.md du TP 2](../02-spam-ou-pas-spam/MATH.md).

### Étape b — gradients de la couche de sortie

Comme $z^{(2)} = a^{(1)} W^{(2)} + b^{(2)}$, on a
$\dfrac{\partial z^{(2)}}{\partial W^{(2)}} = a^{(1)}$ et
$\dfrac{\partial z^{(2)}}{\partial b^{(2)}} = 1$. Donc :

$$
\frac{\partial L}{\partial W^{(2)}} = \big(a^{(1)}\big)^{\!\top} \delta^{(2)}
\qquad
\frac{\partial L}{\partial b^{(2)}} = \delta^{(2)}
$$

### Étape c — faire REMONTER l'erreur dans la couche cachée

L'erreur de sortie « redescend » vers les sorties des neurones cachés $a^{(1)}$.
Comme $z^{(2)} = a^{(1)} W^{(2)} + b^{(2)}$ :

$$
\frac{\partial L}{\partial a^{(1)}} = \delta^{(2)} \big(W^{(2)}\big)^{\!\top}
$$

Puis on **traverse le ReLU** (étape a du §2) pour atteindre le score $z^{(1)}$.
On multiplie **terme à terme** par la dérivée du ReLU :

$$
\delta^{(1)}
\;\equiv\; \frac{\partial L}{\partial z^{(1)}}
= \underbrace{\Big(\delta^{(2)} \big(W^{(2)}\big)^{\!\top}\Big)}_{\partial L / \partial a^{(1)}}
  \odot \operatorname{ReLU}'\!\big(z^{(1)}\big)
$$

où $\odot$ est le produit **élément par élément** (chaque neurone caché garde ou
coupe son propre gradient selon que son $z^{(1)}$ était positif ou non).

### Étape d — gradients de la couche cachée

Enfin, comme $z^{(1)} = x\,W^{(1)} + b^{(1)}$, on procède comme à l'étape b :

$$
\frac{\partial L}{\partial W^{(1)}} = x^{\top} \delta^{(1)}
\qquad
\frac{\partial L}{\partial b^{(1)}} = \delta^{(1)}
$$

---

## 4) Le résumé, et le lien avec le code

En moyennant sur les $N$ exemples d'un batch (matrice $X$ de forme $N \times 2$),
les quatre gradients sont :

$$
\delta^{(2)} = \frac{p - y}{N}
$$

$$
\nabla_{W^{(2)}} L = \big(a^{(1)}\big)^{\!\top}\!\delta^{(2)}
\qquad
\nabla_{b^{(2)}} L = \sum_i \delta^{(2)}_i
$$

$$
\delta^{(1)} = \big(\delta^{(2)} (W^{(2)})^{\top}\big) \odot \operatorname{ReLU}'(z^{(1)})
$$

$$
\nabla_{W^{(1)}} L = X^{\top}\!\delta^{(1)}
\qquad
\nabla_{b^{(1)}} L = \sum_i \delta^{(1)}_i
$$

Ce qui correspond **ligne pour ligne** au code de `retropropagation()` :

```python
dz2 = (p - y) / n                # δ² : l'erreur du TP 2, moyennée
grad_W2 = a1.T @ dz2             # ∇_W2
grad_b2 = dz2.sum(axis=0)        # ∇_b2

da1 = dz2 @ W2.T                 # on remonte : ∂L/∂a1
dz1 = da1 * (z1 > 0)             # δ¹ : on traverse le ReLU
grad_W1 = X.T @ dz1              # ∇_W1
grad_b1 = dz1.sum(axis=0)        # ∇_b1
```

Puis la mise à jour est **la même descente de gradient** qu'aux TP 1 et 2 :

```python
W1 -= LEARNING_RATE * grad_W1
b1 -= LEARNING_RATE * grad_b1
W2 -= LEARNING_RATE * grad_W2
b2 -= LEARNING_RATE * grad_b2
```

---

## 5) Pourquoi un seul neurone ne peut pas séparer le XOR

Un neurone unique calcule $z = w \cdot x + b$ et décide selon le signe de $z$.
La frontière $z = 0$ est l'équation d'une **droite** (un hyperplan en dimension
supérieure). Elle coupe le plan en **deux demi-plans**.

Pour le XOR, les classes `1` sont sur une diagonale et les `0` sur l'autre :
il faudrait qu'une **seule droite** garde $(0,1)$ et $(1,0)$ d'un côté, et
$(0,0)$ et $(1,1)$ de l'autre. C'est **géométriquement impossible**.

La couche cachée résout le problème parce que chaque neurone caché trace **sa
propre droite**, et la couche de sortie **combine** ces droites (via le ReLU, qui
introduit de la **non-linéarité**). Le résultat est une frontière **courbe**, ou
faite de morceaux, capable d'isoler les deux diagonales — ce que `resultat.png`
montre visuellement.

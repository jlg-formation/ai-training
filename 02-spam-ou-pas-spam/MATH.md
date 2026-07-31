# Les maths de la classification (sigmoïde + entropie croisée)

Ce document explique **pas à pas** pourquoi le code d'entraînement du TP 2 est
presque identique à celui du TP 1, alors qu'on a changé la sortie et la perte.
Le point clé : le gradient se simplifie en

```python
erreur = p - y
```

exactement comme au TP 1. Voici pourquoi.

---

## 1) Le modèle

Chaque phrase est un vecteur de nombres $\mathbf{x} = (x_1, \dots, x_V)$ (le sac
de mots, $V$ = taille du vocabulaire ; les composantes $x_j$ sont des
scalaires). Le neurone calcule d'abord une valeur réelle :

$$
z = \mathbf{w} \cdot \mathbf{x} + b = \sum_{j=1}^{V} w_j\,x_j + b
$$

$z$ est un **score brut** : un simple nombre réel, non borné (il peut aller de
$-\infty$ à $+\infty$). C'est la **somme pondérée** des mots de la phrase : chaque
mot présent ($x_j$) ajoute son poids $w_j$, et le biais $b$ décale le tout.
Intuitivement, $z$ mesure à quel point la phrase **penche vers le spam** :

- $z > 0$ : la phrase penche **spam** (d'autant plus que $z$ est grand) ;
- $z < 0$ : elle penche **non-spam** ;
- $z = 0$ : parfaitement indécis.

Ce score brut porte un nom standard : on l'appelle le **logit**. Le mot vient de
la fonction **logit**, qui est l'**inverse de la sigmoïde** :

$$
z = \operatorname{logit}(p) = \ln\!\left(\frac{p}{1 - p}\right)
$$

C'est le **log des cotes** (*log-odds*) : le logarithme du rapport
$\frac{p}{1 - p}$ (probabilité spam / probabilité non-spam). Autrement dit, la
sigmoïde transforme un logit en probabilité, et le logit fait le chemin inverse.
On retrouve bien l'intuition ci-dessus : $z > 0$ ⇔ cotes en faveur du spam
($p > 0{,}5$), $z = 0$ ⇔ cotes égales ($p = 0{,}5$).

Ce score n'est pas encore une probabilité (il peut valoir $-4$ ou $+12$). On
l'écrase donc entre $0$ et $1$ avec la **sigmoïde** pour en faire une
**probabilité** :

$$
p = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

- $p$ (`p`) : probabilité prédite que la phrase soit du **spam**.
- $y \in \{0, 1\}$ : la vraie étiquette (1 = spam, 0 = non-spam).

Dans le code :

```python
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def predire_proba(X, w, b):
    return sigmoid(X @ w + b)
```

---

## 2) Pourquoi pas la MSE ? La perte BCE

Avec une probabilité, on n'utilise pas l'erreur quadratique mais l'**entropie
croisée binaire** (*Binary Cross-Entropy*). Pour $N$ exemples (ici $N$ =
nombre de phrases d'entraînement) :

$$
L = \frac{1}{N} \sum_{i=1}^{N}
    -\Big[\, y_i \log(p_i) + (1 - y_i)\log(1 - p_i) \,\Big]
$$

Comment la lire, pour **un** exemple :

- si $y = 1$, il reste $-\log(p)$ : la perte est petite si $p \to 1$, énorme si
  $p \to 0$ ;
- si $y = 0$, il reste $-\log(1 - p)$ : petite si $p \to 0$, énorme si $p \to 1$.

Autrement dit, la BCE **punit très fort une prédiction confiante et fausse**.
Dans le code :

```python
def perte(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)   # évite log(0)
    return np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))
```

---

## 3) Une propriété magique de la sigmoïde

Sa dérivée s'exprime avec elle-même :

$$
\sigma'(z) = \sigma(z)\,\big(1 - \sigma(z)\big) = p\,(1 - p)
$$

C'est ce qui va faire disparaître tous les termes compliqués.

---

## 4) Le gradient se simplifie en $p - y$

On dérive la perte d'un exemple $L_i = -\big[y_i\log p_i + (1-y_i)\log(1-p_i)\big]$ par
rapport à $z_i$, avec $p_i = \sigma(z_i)$.

**Étape a — dériver $L_i$ par rapport à $p_i$ :**

$$
\frac{\partial L_i}{\partial p_i}
= -\frac{y_i}{p_i} + \frac{1 - y_i}{1 - p_i}
= \frac{p_i - y_i}{p_i\,(1 - p_i)}
$$

**Étape b — dériver $p_i$ par rapport à $z_i$** (propriété du §3) :

$$
\frac{\partial p_i}{\partial z_i} = p_i\,(1 - p_i)
$$

**Étape c — règle de la chaîne :** on multiplie, et $p_i(1-p_i)$ se simplifie :

$$
\frac{\partial L_i}{\partial z_i}
= \frac{p_i - y_i}{p_i\,(1 - p_i)} \cdot p_i\,(1 - p_i)
= p_i - y_i
$$

$$
\boxed{\;\frac{\partial L_i}{\partial z_i} = p_i - y_i\;}
$$

Tout le reste s'évanouit : c'est **la même « erreur »** qu'au TP 1.

---

## 5) Gradients par rapport aux poids

Rappel : la perte totale est la **moyenne** des pertes par exemple,
$L = \dfrac{1}{N}\sum_{i=1}^{N} L_i$. Sa dérivée est donc la **moyenne des
dérivées** des $L_i$ (la dérivation passe à travers la somme).

Comme $z_i = \mathbf{w} \cdot \mathbf{x}_i + b$, on a $\dfrac{\partial z_i}{\partial w_j} = x_{ij}$ et
$\dfrac{\partial z_i}{\partial b} = 1$. En combinant avec $\dfrac{\partial L_i}{\partial z_i} = p_i - y_i$ (§4)
et en moyennant sur les $N$ exemples :

$$
\frac{\partial L}{\partial w_j} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)\,x_{ij}
\qquad
\frac{\partial L}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)
$$

En notation matricielle (avec $\mathbf{X}$ la matrice des sacs de mots) :

$$
\mathbf{X} =
\begin{pmatrix}
x_{11} & x_{12} & \cdots & x_{1V} \\
x_{21} & x_{22} & \cdots & x_{2V} \\
\vdots & \vdots & \ddots & \vdots \\
x_{N1} & x_{N2} & \cdots & x_{NV}
\end{pmatrix}
=
\begin{pmatrix}
\mathbf{x}_1^\top \\ \mathbf{x}_2^\top \\ \vdots \\ \mathbf{x}_N^\top
\end{pmatrix}
$$

Chaque **ligne** $i$ est le sac de mots $\mathbf{x}_i$ d'une phrase ($N$ phrases
d'entraînement au total) ; chaque **colonne** $j$ correspond à un mot du
vocabulaire ($V$ mots). Le terme $x_{ij}$ est le nombre d'occurrences du
mot $j$ dans la phrase $i$. Dans le code, c'est exactement `X_train` (de forme
`(N, V)`). Les gradients s'écrivent alors :

$$
\nabla_{\mathbf{w}} L = \frac{1}{N}\, \mathbf{X}^\top (\mathbf{p} - \mathbf{y})
\qquad
\nabla_b L = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)
$$

où $\mathbf{p} = (p_1, \dots, p_N)^\top$ est le vecteur des probabilités prédites
(une par phrase) et $\mathbf{y} = (y_1, \dots, y_N)^\top$ celui des vraies
étiquettes. Dans le
code, ce sont `p_train` et `y_train`.

Ce qui donne, dans le code :

```python
erreur = p_train - y_train
grad_w = X_train.T @ erreur / len(y_train)
grad_b = np.mean(erreur)
```

---

## 6) La mise à jour (identique au TP 1)

On descend le gradient avec le **learning rate** $\eta$ :

$$
w \leftarrow w - \eta\,\nabla_w L
\qquad
b \leftarrow b - \eta\,\nabla_b L
$$

```python
w -= LEARNING_RATE * grad_w
b -= LEARNING_RATE * grad_b
```

**Conclusion :** en changeant la sortie (sigmoïde) *et* la perte (BCE), on
retombe sur la **même** règle de mise à jour que pour la régression linéaire.
La seule chose qui change vraiment côté code, c'est le calcul de `p`.

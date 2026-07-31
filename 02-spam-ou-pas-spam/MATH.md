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

Chaque phrase est un vecteur de nombres $x = (x_1, \dots, x_d)$ (le sac de
mots). Le neurone calcule d'abord une valeur réelle :

$$
z = w \cdot x + b = \sum_{j=1}^{d} w_j\,x_j + b
$$

puis l'écrase entre $0$ et $1$ avec la **sigmoïde** pour en faire une
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
croisée binaire** (*Binary Cross-Entropy*). Pour $N$ exemples :

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

On dérive la perte d'un exemple $L_i = -\big[y\log p + (1-y)\log(1-p)\big]$ par
rapport à $z$, avec $p = \sigma(z)$.

**Étape a — dériver $L$ par rapport à $p$ :**

$$
\frac{\partial L_i}{\partial p}
= -\frac{y}{p} + \frac{1 - y}{1 - p}
= \frac{p - y}{p\,(1 - p)}
$$

**Étape b — dériver $p$ par rapport à $z$** (propriété du §3) :

$$
\frac{\partial p}{\partial z} = p\,(1 - p)
$$

**Étape c — règle de la chaîne :** on multiplie, et $p(1-p)$ se simplifie :

$$
\frac{\partial L_i}{\partial z}
= \frac{p - y}{p\,(1 - p)} \cdot p\,(1 - p)
= p - y
$$

$$
\boxed{\;\frac{\partial L_i}{\partial z} = p - y\;}
$$

Tout le reste s'évanouit : c'est **la même « erreur »** qu'au TP 1.

---

## 5) Gradients par rapport aux poids

Comme $z = w \cdot x + b$, on a $\dfrac{\partial z}{\partial w_j} = x_j$ et
$\dfrac{\partial z}{\partial b} = 1$. En moyennant sur les $N$ exemples :

$$
\frac{\partial L}{\partial w_j} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)\,x_{ij}
\qquad
\frac{\partial L}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)
$$

En notation matricielle (avec $X$ la matrice des sacs de mots) :

$$
\nabla_w L = \frac{1}{N}\, X^\top (p - y)
\qquad
\nabla_b L = \text{moyenne}(p - y)
$$

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

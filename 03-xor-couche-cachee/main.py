"""
TP 3 - Pourquoi plusieurs neurones ? Le XOR et la couche cachée

Au TP 1 puis au TP 2, un SEUL neurone suffisait. Il ne sait tracer qu'une
FRONTIÈRE DROITE. Or certains problèmes ne sont PAS séparables par une droite :
le plus célèbre est le XOR (ou exclusif).

    XOR :  (0,0) -> 0     (1,1) -> 0        (les coins d'une même diagonale)
           (0,1) -> 1     (1,0) -> 1        (les coins de l'autre diagonale)

Impossible de séparer les 1 des 0 avec une seule droite. La solution : ajouter
une COUCHE CACHÉE de plusieurs neurones entre l'entrée et la sortie. C'est LA
nouveauté de ce TP.

Architecture (toujours en NumPy pur, tout à la main) :

    2 entrées  ->  couche cachée de H neurones (ReLU)  ->  1 sortie (sigmoïde)

On réutilise EXACTEMENT la sigmoïde et la perte BCE du TP 2. La seule mécanique
nouvelle est la RÉTROPROPAGATION : faire « remonter » le gradient à travers la
couche cachée (règle de la chaîne). Le détail est dans MATH.md.

H est notre variable d'expérience : avec H = 8, le réseau RÉSOUT le XOR ; avec
H = 1, il RETOMBE sur un quasi-neurone unique et ÉCHOUE (voir le README).

Lancer avec :  uv run main.py
"""

import numpy as np

# Pour que les résultats soient reproductibles d'une exécution à l'autre
np.random.seed(0)


# ======================================================================
# CONSTANTE PÉDAGOGIQUE : le nombre de neurones de la couche cachée
# ======================================================================
# H = 8 -> le réseau apprend une frontière COURBE et résout le XOR (~100 %).
# H = 1 -> une seule "cellule" cachée : la frontière n'a qu'un seul "pli",
#          l'accuracy plafonne vers ~75 %, loin des 100 %. Essaie les deux !
H = 8


# ----------------------------------------------------------------------
# 1) Génération des données : un XOR BRUITÉ (4 nuages de points)
# ----------------------------------------------------------------------
# Le XOR "pur" n'a que 4 points. Pour avoir un vrai jeu train/test et une jolie
# frontière à dessiner, on remplace chaque coin par un NUAGE de points gaussiens
# autour de ce coin. La classe reste celle du coin (diagonale = 0, anti-diag = 1).
CENTRES = [
    ((0.0, 0.0), 0),   # coin bas-gauche  -> classe 0
    ((1.0, 1.0), 0),   # coin haut-droite -> classe 0
    ((0.0, 1.0), 1),   # coin haut-gauche -> classe 1
    ((1.0, 0.0), 1),   # coin bas-droite  -> classe 1
]
POINTS_PAR_COIN = 150      # nuage de 150 points autour de chaque coin
ECART = 0.15               # dispersion (écart-type) des nuages


def generer_xor():
    """Fabrique le jeu complet : X de forme (N, 2) et y de forme (N,)."""
    X = []
    y = []
    for (cx, cy), classe in CENTRES:
        nuage = np.random.normal(
            loc=[cx, cy], scale=ECART, size=(POINTS_PAR_COIN, 2)
        )
        X.append(nuage)
        y += [classe] * POINTS_PAR_COIN
    X = np.vstack(X)
    y = np.array(y, dtype=float)

    # On mélange puis on coupe en train (70 %) / test (30 %)
    ordre = np.random.permutation(len(y))
    X, y = X[ordre], y[ordre]
    n_train = int(0.7 * len(y))
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


X_train, y_train, X_test, y_test = generer_xor()


# ----------------------------------------------------------------------
# 2) Le modèle : 2 -> H (ReLU) -> 1 (sigmoïde)
# ----------------------------------------------------------------------
# Convention de nommage (comme dans README.md / MATH.md) :
#   - MAJUSCULE -> matrice   : X (batch), W1 (2 x H), W2 (H x 1)
#   - minuscule -> vecteur   : b1, z1, a1 (taille H), et p, y (batch)
#   - maigre    -> scalaire  : b2, z2
# Deux couches de poids :
#   - W1 (2 x H) + b1 (H)  : de l'entrée vers la couche cachée
#   - W2 (H x 1) + b2 (1)  : de la couche cachée vers la sortie
#
# INITIALISATION : au TP 2 on partait de zéros. Ici c'est INTERDIT pour la
# couche cachée : si tous les poids valent 0, tous les neurones cachés calculent
# la même chose et reçoivent le même gradient -> ils restent identiques à jamais.
# Il faut BRISER LA SYMÉTRIE avec de l'aléatoire. On utilise l'init "He"
# (randn * sqrt(2/n_entrées)), bien adaptée au ReLU.
W1 = np.random.randn(2, H) * np.sqrt(2.0 / 2)
b1 = np.zeros(H)
W2 = np.random.randn(H, 1) * np.sqrt(2.0 / H)
b2 = np.zeros(1)


def relu(z):
    """Activation de la couche cachée : garde le positif, écrase le négatif."""
    return np.maximum(0.0, z)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def avant(X):
    """Passe AVANT (forward). Renvoie la proba p ET les valeurs intermédiaires
    (z1, a1) dont la rétropropagation aura besoin."""
    z1 = X @ W1 + b1          # (N, H) : score de chaque neurone caché
    a1 = relu(z1)             # (N, H) : sortie de la couche cachée
    z2 = a1 @ W2 + b2         # (N, 1) : score de sortie (logit)
    p = sigmoid(z2)           # (N, 1) : probabilité de la classe 1
    return p.ravel(), z1, a1


def predire_proba(X):
    """Probabilité de la classe 1 (raccourci qui ignore les intermédiaires)."""
    p, _, _ = avant(X)
    return p


# ----------------------------------------------------------------------
# 3) La fonction de perte (loss) : BCE, identique au TP 2
# ----------------------------------------------------------------------
# Elle ne sert QU'À AFFICHER le progrès. Ce qui entraîne, c'est le gradient
# (étape 5), calculé par rétropropagation (étape 4).
def perte(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)          # évite log(0)
    return np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))


def accuracy(p, y):
    """Proportion de bonnes réponses (on tranche à 0.5)."""
    return np.mean((p >= 0.5).astype(float) == y)


# ----------------------------------------------------------------------
# 4) La RÉTROPROPAGATION : la seule vraie nouveauté du TP
# ----------------------------------------------------------------------
# On veut le gradient de la perte par rapport aux 4 paquets de poids. On part de
# la sortie et on "remonte" couche par couche avec la règle de la chaîne.
#
#   dz2 = p - y                (identique au TP 2 : gradient BCE + sigmoïde)
#   -> gradients de la couche de sortie (W2, b2)
#   da1 = dz2 @ W2.T           on fait REMONTER l'erreur dans la couche cachée
#   dz1 = da1 * (z1 > 0)       on traverse le ReLU (sa dérivée vaut 1 si z>0, sinon 0)
#   -> gradients de la couche cachée (W1, b1)
#
# Le détail mathématique est dans MATH.md.
def retropropagation(X, y, z1, a1, p):
    n = len(y)
    y = y.reshape(-1, 1)            # (N, 1) pour coller à la forme de p
    p = p.reshape(-1, 1)

    # --- couche de SORTIE ---
    dz2 = (p - y) / n              # (N, 1) : l'erreur du TP 2, moyennée
    grad_W2 = a1.T @ dz2          # (H, 1)
    grad_b2 = dz2.sum(axis=0)     # (1,)

    # --- on REMONTE vers la couche CACHÉE ---
    da1 = dz2 @ W2.T              # (N, H) : combien chaque neurone caché a pesé
    dz1 = da1 * (z1 > 0)         # (N, H) : on traverse le ReLU
    grad_W1 = X.T @ dz1          # (2, H)
    grad_b1 = dz1.sum(axis=0)    # (H,)

    return grad_W1, grad_b1, grad_W2, grad_b2


# ----------------------------------------------------------------------
# 5) L'entraînement : la descente de gradient (identique aux TP précédents)
# ----------------------------------------------------------------------
LEARNING_RATE = 0.1
EPOCHS = 2000


def main():
    global W1, b1, W2, b2

    print(f"Neurones cachés H = {H}")
    print(f"Exemples d'entraînement : {len(y_train)} | de test : {len(y_test)}\n")

    historique_train = []
    historique_test = []

    for epoch in range(EPOCHS):
        # a) Passe avant : on garde les intermédiaires pour la rétropropagation
        p_train, z1, a1 = avant(X_train)

        # b) Rétropropagation : les gradients des 4 paquets de poids
        grad_W1, grad_b1, grad_W2, grad_b2 = retropropagation(
            X_train, y_train, z1, a1, p_train
        )

        # c) Mise à jour des poids (même règle qu'aux TP 1 et 2)
        W1 -= LEARNING_RATE * grad_W1
        b1 -= LEARNING_RATE * grad_b1
        W2 -= LEARNING_RATE * grad_W2
        b2 -= LEARNING_RATE * grad_b2

        # Suivi de la loss sur train ET test (pour la courbe)
        historique_train.append(perte(p_train, y_train))
        historique_test.append(perte(predire_proba(X_test), y_test))

        if epoch % 200 == 0 or epoch == EPOCHS - 1:
            print(
                f"Epoch {epoch:4d} | loss_train = {historique_train[-1]:.4f} "
                f"| loss_test = {historique_test[-1]:.4f}"
            )

    # ------------------------------------------------------------------
    # 6) Résultat : accuracy train / test
    # ------------------------------------------------------------------
    acc_train = accuracy(predire_proba(X_train), y_train)
    acc_test = accuracy(predire_proba(X_test), y_test)
    print(f"\nAccuracy train : {acc_train:.1%}")
    print(f"Accuracy test  : {acc_test:.1%}")
    if acc_test > 0.9:
        print("=> La couche cachée a RÉSOLU le XOR (frontière courbe).")
    else:
        print("=> Le réseau N'ARRIVE PAS à séparer le XOR (essaie H = 8).")

    # ------------------------------------------------------------------
    # 7) Visualisation : la frontière de décision + la courbe de loss
    # ------------------------------------------------------------------
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- (a) Frontière de décision ---
    # On évalue le réseau sur une grille fine et on colore le fond selon la
    # probabilité prédite : on VOIT la forme de la frontière apprise.
    marge = 0.4
    xs = np.linspace(X_train[:, 0].min() - marge, X_train[:, 0].max() + marge, 300)
    ys = np.linspace(X_train[:, 1].min() - marge, X_train[:, 1].max() + marge, 300)
    gx, gy = np.meshgrid(xs, ys)
    grille = np.c_[gx.ravel(), gy.ravel()]
    proba_grille = predire_proba(grille).reshape(gx.shape)

    fond = ax1.contourf(gx, gy, proba_grille, levels=50, cmap="RdBu_r", alpha=0.8)
    ax1.contour(gx, gy, proba_grille, levels=[0.5], colors="black", linewidths=1.5)
    fig.colorbar(fond, ax=ax1, label="proba classe 1")
    # Les points de test, coloriés par leur vraie classe
    ax1.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap="RdBu_r",
        edgecolors="k", s=25, linewidths=0.5,
    )
    ax1.set_title(f"Frontière de décision (H = {H})")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")

    # --- (b) Courbe de loss train vs test ---
    ax2.plot(historique_train, label="loss entraînement", color="blue")
    ax2.plot(historique_test, label="loss test", color="red")
    ax2.set_title("TP 3 - XOR : loss au fil des epochs")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss (BCE)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("resultat.png", dpi=120)
    print("\nGraphique enregistré dans resultat.png")


if __name__ == "__main__":
    main()

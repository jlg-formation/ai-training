"""
TP 1 - Le neurone unique : régression linéaire

Objectif : apprendre les paramètres a et b de la droite  y = a*x + b
à partir de données générées avec la vraie relation  y = 2x + 3.

On n'utilise QUE NumPy : pas de framework d'IA, tout est fait à la main
pour bien comprendre :
  - les données (x, y)
  - la fonction de perte (loss)
  - la descente de gradient (comment les poids a et b évoluent)

Lancer avec :  uv run main.py
"""

import numpy as np

# Pour que les résultats soient reproductibles d'une exécution à l'autre
np.random.seed(0)


# ----------------------------------------------------------------------
# 1) Génération des données
# ----------------------------------------------------------------------
# On choisit la "vérité" que le modèle devra retrouver.
A_VRAI = 2.0
B_VRAI = 3.0

N = 101                                   # nombre d'exemples
print(f"==>> N: {N}")
x = np.linspace(-5, 5, N)                 # 100 points entre -5 et 5
print(f"==>> x: {x}")
bruit = np.random.normal(0, 1.0, size=N)  # un peu de bruit pour faire réaliste
print(f"==>> bruit: {bruit}")
y = A_VRAI * x + B_VRAI + bruit           # y = 2x + 3 (+ bruit)


# ----------------------------------------------------------------------
# 2) Le modèle : un seul neurone  ->  y_pred = a*x + b
# ----------------------------------------------------------------------
# On initialise a et b au hasard : le modèle ne connaît pas encore la vérité.
a = np.random.randn()
b = np.random.randn()


def predire(x, a, b):
    """Sortie du neurone pour une entrée x."""
    return a * x + b


# ----------------------------------------------------------------------
# 3) La fonction de perte (loss) : erreur quadratique moyenne (MSE)
# ----------------------------------------------------------------------
# Elle mesure à quel point les prédictions sont éloignées de la réalité.
# Plus la loss est petite, meilleur est le modèle.
def perte(y_pred, y):
    return np.mean((y_pred - y) ** 2)


# ----------------------------------------------------------------------
# 4) L'entraînement : la descente de gradient
# ----------------------------------------------------------------------
# À chaque étape (epoch) :
#   - on calcule les prédictions
#   - on calcule l'erreur
#   - on calcule le gradient (la pente de la loss par rapport à a et b)
#   - on déplace a et b dans le sens qui fait DIMINUER la loss
LEARNING_RATE = 0.01   # taille des pas ; trop grand = ça diverge, trop petit = ça rame
EPOCHS = 200           # nombre de passages sur les données


def main():
    global a, b

    print(f"Vraie relation à retrouver : y = {A_VRAI}x + {B_VRAI}\n")
    print(f"Départ (aléatoire) : a = {a:.3f}, b = {b:.3f}\n")

    for epoch in range(EPOCHS):
        # a) Prédiction du modèle
        y_pred = predire(x, a, b)

        # b) Erreur entre prédiction et réalité
        erreur = y_pred - y

        # c) Gradients (dérivées de la MSE par rapport à a et b)
        grad_a = 2 * np.mean(erreur * x)
        grad_b = 2 * np.mean(erreur)

        # d) Mise à jour des poids : on va dans le sens opposé au gradient
        a -= LEARNING_RATE * grad_a
        b -= LEARNING_RATE * grad_b

        # Affichage régulier pour voir la loss diminuer et les poids évoluer
        if epoch % 20 == 0 or epoch == EPOCHS - 1:
            loss = perte(y_pred, y)
            print(f"Epoch {epoch:3d} | loss = {loss:8.4f} | a = {a:.3f} | b = {b:.3f}")

    # ------------------------------------------------------------------
    # 5) Résultat final
    # ------------------------------------------------------------------
    print(f"\nModèle appris   : y = {a:.3f}x + {b:.3f}")
    print(f"Vraie relation  : y = {A_VRAI}x + {B_VRAI}")

    # ------------------------------------------------------------------
    # 6) Visualisation : on enregistre un graphique
    # ------------------------------------------------------------------
    import matplotlib.pyplot as plt

    plt.scatter(x, y, s=15, label="Données (bruitées)")
    plt.plot(x, predire(x, a, b), color="red", linewidth=2, label="Droite apprise")
    plt.plot(x, A_VRAI * x + B_VRAI, color="green", linestyle="--", label="Vraie droite")
    plt.title("TP 1 - Régression linéaire (1 neurone)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("resultat.png", dpi=120)
    print("\nGraphique enregistré dans resultat.png")


if __name__ == "__main__":
    main()

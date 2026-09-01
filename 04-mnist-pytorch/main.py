"""
TP 4 - Reconnaissance de chiffres MNIST : la PREMIÈRE fois en PyTorch

Aux TP 1 à 3, on a TOUT écrit à la main avec NumPy : la passe avant, la perte,
et surtout la RÉTROPROPAGATION (les gradients dérivés une à une). C'était le but
pédagogique : comprendre la mécanique. Maintenant qu'on l'a comprise, on la
laisse à un FRAMEWORK. C'est LA nouveauté du TP : PyTorch.

Le problème, lui, est le prolongement direct du TP 3 :

    TP 3 : 2 entrées      -> couche cachée (ReLU) -> 1 sortie   (2 classes)
    TP 4 : 784 entrées    -> couche cachée (ReLU) -> 10 sorties (10 classes)

Une image MNIST fait 28 x 28 = 784 pixels ; il y a 10 chiffres (0 à 9). C'est le
MÊME MLP qu'au TP 3, juste plus large, et réécrit avec PyTorch.

Ce qui change concrètement (voir README.md pour le tableau complet) :
  - la rétropropagation "à la main" -> loss.backward()  (AUTOGRAD)
  - W -= lr * grad                  -> optimizer.step()
  - le dataset traité d'un coup     -> des MINI-BATCHS via un DataLoader
  - sigmoïde + BCE (2 classes)      -> softmax + cross-entropy (10 classes)
  - CPU implicite                   -> .to(device) (CPU par défaut, GPU si dispo)

PIÈGE À RETENIR : nn.CrossEntropyLoss attend les LOGITS bruts (elle applique le
softmax elle-même). La couche de sortie ne doit donc PAS avoir d'activation.

Lancer avec :  uv run main.py
"""

import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Reproductibilité (l'équivalent PyTorch de np.random.seed du TP 3)
torch.manual_seed(0)


# ======================================================================
# CONSTANTES PÉDAGOGIQUES : les leviers à faire varier (voir README.md)
# ======================================================================
H = 128  # neurones de la couche cachée (comme le H du TP 3)
LEARNING_RATE = 0.1  # pas de la descente de gradient
EPOCHS = 5  # nombre de passages complets sur les données d'entraînement
BATCH_SIZE = 64  # nombre d'images par mini-batch


# ----------------------------------------------------------------------
# 0) Le "device" : CPU ou GPU
# ----------------------------------------------------------------------
# En PyTorch, un calcul se fait là où sont les données. On choisit donc un
# "device" et on y enverra le modèle ET les données avec .to(device).
# Par défaut ce sera le CPU (suffisant pour ce petit MLP). Si tu as un GPU
# NVIDIA installé avec CUDA, PyTorch le détecte et l'utilise automatiquement.
device = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------
# 1) Les données : MNIST via un Dataset + un DataLoader
# ----------------------------------------------------------------------
# torchvision télécharge MNIST la PREMIÈRE fois (~11 Mo) dans ../data (le dossier
# data/ PARTAGÉ à la racine du projet), puis le relit depuis le disque. Comme les
# TP 6 et 7 pointent vers le même ../data, le téléchargement n'a lieu qu'une fois
# pour les trois TP (download=True = « télécharge SI absent »). ToTensor convertit
# l'image en tenseur de forme (1, 28, 28) avec des pixels ramenés dans [0, 1].
#   - train=True  : les 60 000 images d'entraînement
#   - train=False : les 10 000 images de test (le split officiel, comme le
#                   train/test qu'on découpait nous-mêmes aux TP 2 et 3)
jeu_train = datasets.MNIST(
    root="../data", train=True, download=True, transform=ToTensor()
)
jeu_test = datasets.MNIST(
    root="../data", train=False, download=True, transform=ToTensor()
)

# Le DataLoader découpe le jeu en MINI-BATCHS et (pour l'entraînement) mélange
# les images à chaque epoch. Nouveauté du TP : on n'entraîne plus sur tout le
# jeu d'un coup, mais batch par batch (descente de gradient "stochastique").
loader_train = DataLoader(jeu_train, batch_size=BATCH_SIZE, shuffle=True)
loader_test = DataLoader(jeu_test, batch_size=BATCH_SIZE)


# ----------------------------------------------------------------------
# 2) Le modèle : 784 -> H (ReLU) -> 10, en nn.Module
# ----------------------------------------------------------------------
# Un modèle PyTorch est une classe qui hérite de nn.Module. On y déclare les
# couches dans __init__, et l'enchaînement dans forward. nn.Linear EST la couche
# du TP 3 (le "X @ W + b"), mais PyTorch crée et gère les poids tout seul.
class ReseauMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.couche_cachee = nn.Linear(28 * 28, H)  # W1 (784 x H) + b1
        self.sortie = nn.Linear(H, 10)  # W2 (H x 10) + b2

    def forward(self, x):
        x = x.view(x.size(0), -1)  # aplatit (N, 1, 28, 28) -> (N, 784)
        x = torch.relu(self.couche_cachee(x))
        return self.sortie(x)  # LOGITS bruts : PAS de softmax ici !


modele = ReseauMLP().to(device)


# ----------------------------------------------------------------------
# 3) La perte et l'optimiseur
# ----------------------------------------------------------------------
# nn.CrossEntropyLoss = softmax + entropie croisée en une seule opération. C'est
# la généralisation à 10 classes de la sigmoïde + BCE du TP 2 (détail dans
# MATH.md). Elle prend les LOGITS et l'indice de la vraie classe (0 à 9).
perte = nn.CrossEntropyLoss()

# L'optimiseur encapsule la mise à jour des poids (le "W -= lr * grad" fait à la
# main aux TP précédents). SGD = Stochastic Gradient Descent, exactement la même
# descente de gradient, mais gérée par PyTorch.
optimiseur = torch.optim.SGD(modele.parameters(), lr=LEARNING_RATE)


# ----------------------------------------------------------------------
# 4) Fonctions d'entraînement et d'évaluation (une epoch chacune)
# ----------------------------------------------------------------------
def entrainer_une_epoch():
    """Un passage complet sur le jeu d'entraînement, batch par batch."""
    modele.train()  # mode entraînement
    perte_totale = 0.0
    for images, cibles in loader_train:
        images, cibles = images.to(device), cibles.to(device)

        # Les 4 gestes clés de PyTorch, à la place de notre code NumPy :
        optimiseur.zero_grad()  # a) remet les gradients à zéro
        logits = modele(images)  # b) passe avant (appelle forward)
        p = perte(logits, cibles)  # c) calcule la perte
        p.backward()  # d) AUTOGRAD : tous les gradients
        optimiseur.step()  #    met à jour les poids

        perte_totale += p.item()
    return perte_totale / len(loader_train)


def evaluer(loader):
    """Renvoie (perte moyenne, accuracy) sur un loader, sans apprendre."""
    modele.eval()  # mode évaluation
    perte_totale = 0.0
    bonnes = 0
    total = 0
    with torch.no_grad():  # pas de gradient : plus rapide
        for images, cibles in loader:
            images, cibles = images.to(device), cibles.to(device)
            logits = modele(images)
            perte_totale += perte(logits, cibles).item()
            predictions = logits.argmax(dim=1)  # classe au plus gros logit
            bonnes += (predictions == cibles).sum().item()
            total += cibles.size(0)
    return perte_totale / len(loader), bonnes / total


# ----------------------------------------------------------------------
# 5) La boucle d'entraînement
# ----------------------------------------------------------------------
def main():
    print(f"Device utilisé : {device}")
    n_parametres = sum(p.numel() for p in modele.parameters())
    print(f"Neurones cachés H = {H} | paramètres du modèle : {n_parametres:,}")
    print(
        f"Images train : {len(jeu_train)} | test : {len(jeu_test)} "
        f"| batch = {BATCH_SIZE}\n"
    )

    historique_train = []
    historique_test = []

    for epoch in range(EPOCHS):
        t0 = time.time()
        loss_train = entrainer_une_epoch()
        loss_test, acc_test = evaluer(loader_test)
        historique_train.append(loss_train)
        historique_test.append(loss_test)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | loss_train = {loss_train:.4f} "
            f"| loss_test = {loss_test:.4f} | acc_test = {acc_test:.1%} "
            f"| {time.time() - t0:.1f} s"
        )

    # ------------------------------------------------------------------
    # 6) Résultat final
    # ------------------------------------------------------------------
    _, acc_train = evaluer(loader_train)
    _, acc_test = evaluer(loader_test)
    print(f"\nAccuracy train : {acc_train:.1%}")
    print(f"Accuracy test  : {acc_test:.1%}")

    # ------------------------------------------------------------------
    # 7) Visualisation : prédictions, courbe de loss, matrice de confusion
    # ------------------------------------------------------------------
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 5))

    # --- (a) Une poignée d'images de test avec la prédiction ---
    # On prend le premier batch de test pour illustrer (vert = correct, rouge =
    # erreur). C'est l'équivalent visuel du "voir ce que le modèle a appris".
    images, cibles = next(iter(loader_test))
    modele.eval()
    with torch.no_grad():
        predictions = modele(images.to(device)).argmax(dim=1).cpu()

    for i in range(10):
        ax = fig.add_subplot(2, 10, i + 1)
        ax.imshow(images[i].squeeze(), cmap="gray")
        correct = predictions[i].item() == cibles[i].item()
        ax.set_title(str(predictions[i].item()), color="green" if correct else "red")
        ax.axis("off")

    # --- (b) Courbe de loss train vs test ---
    ax_loss = fig.add_subplot(2, 2, 3)
    ax_loss.plot(
        range(1, EPOCHS + 1), historique_train, label="loss entraînement", color="blue"
    )
    ax_loss.plot(range(1, EPOCHS + 1), historique_test, label="loss test", color="red")
    ax_loss.set_title("TP 4 - MNIST : loss au fil des epochs")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss (cross-entropy)")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # --- (c) Matrice de confusion 10 x 10 (quels chiffres se confondent) ---
    # confusion[vrai, prédit] = nombre d'images de la classe "vrai" classées
    # comme "prédit". La diagonale = les bonnes réponses ; hors diagonale = les
    # confusions typiques (ex. 4 <-> 9, 3 <-> 5).
    confusion = torch.zeros(10, 10, dtype=torch.int)
    with torch.no_grad():
        for imgs, cibs in loader_test:
            preds = modele(imgs.to(device)).argmax(dim=1).cpu()
            for vrai, predit in zip(cibs, preds):
                confusion[vrai, predit] += 1

    ax_conf = fig.add_subplot(2, 2, 4)
    im = ax_conf.imshow(confusion, cmap="Blues")
    ax_conf.set_title("Matrice de confusion (test)")
    ax_conf.set_xlabel("chiffre prédit")
    ax_conf.set_ylabel("vrai chiffre")
    ax_conf.set_xticks(range(10))
    ax_conf.set_yticks(range(10))
    fig.colorbar(im, ax=ax_conf, fraction=0.046)

    plt.tight_layout()
    plt.savefig("resultat.png", dpi=120)
    print("\nGraphique enregistré dans resultat.png")


if __name__ == "__main__":
    main()

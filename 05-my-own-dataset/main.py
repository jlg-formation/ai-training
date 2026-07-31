"""
TP 5 - Ton propre dataset : reconnaître pommes vs poires

NOUVEAUTÉ DU TP : ce n'est PLUS un jeu tout prêt (MNIST). C'EST TOI qui fabriques
le dataset, en rangeant tes images dans des dossiers (voir telecharger_dataset.py
et le README). La leçon n'est pas le modèle mais LES DONNÉES : leur qualité, leur
quantité, l'équilibre entre les classes.

Pour que le dataset soit le SEUL facteur qui compte, on EMPRUNTE un réseau déjà
entraîné : ResNet18, appris sur des millions d'images (ImageNet). On GÈLE tout
son corps (ses poids ne bougent plus) et on ne réentraîne QUE sa dernière couche,
remplacée par une sortie à 2 classes (pomme / poire). Ainsi le modèle n'est
jamais le facteur limitant : ce que tu observes vient bien des données.

    TP 4 : MNIST fourni, propre, 28x28 gris   -> MLP entraîné de zéro
    TP 5 : TES photos, hétérogènes, couleur   -> ResNet18 gelé + tête neuve

(On démontera le fonctionnement interne d'un CNN au TP suivant. Ici, on l'utilise
comme une "boîte" qui transforme une image en caractéristiques.)

Lancer avec :  uv run main.py
(Avant, il faut des images : uv run telecharger_dataset.py, puis TRIER à la main.)
"""

import time

import pillow_avif  # noqa: F401  -> enregistre le format AVIF auprès de Pillow
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18

# Reproductibilité (comme torch.manual_seed du TP 4)
torch.manual_seed(0)


# ======================================================================
# CONSTANTES PÉDAGOGIQUES : les leviers à faire varier (voir README.md)
# ======================================================================
TAILLE_IMAGE = 224     # ResNet a été entraîné sur des images 224 x 224
LEARNING_RATE = 0.01   # pas de la descente de gradient
EPOCHS = 10            # passages complets sur le jeu d'entraînement
BATCH_SIZE = 8         # petit : on a peu d'images


# ----------------------------------------------------------------------
# 0) Le "device" : CPU ou GPU (identique au TP 4)
# ----------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------
# 1) Les transformations : uniformiser des photos hétérogènes
# ----------------------------------------------------------------------
# Contrairement à MNIST, tes photos ont des tailles et des couleurs variées. Il
# faut donc TOUT ramener au même format avant d'entrer dans le réseau :
#   - Resize      : même taille pour toutes (obligatoire pour empiler en batch)
#   - ToTensor    : image -> tenseur (3, H, W), pixels dans [0, 1]
#   - Normalize   : on recentre avec les statistiques d'ImageNet, celles avec
#                   lesquelles ResNet a appris (sinon les valeurs "détonnent").
transformation = transforms.Compose([
    transforms.Resize((TAILLE_IMAGE, TAILLE_IMAGE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ----------------------------------------------------------------------
# 2) TON dataset : ImageFolder lit les classes depuis les sous-dossiers
# ----------------------------------------------------------------------
# ImageFolder remplace le datasets.MNIST du TP 4. Il déduit les classes du NOM
# des sous-dossiers : dataset/train/pommes -> classe "pommes", etc. C'est le
# cœur du TP : le dataset n'est plus téléchargé tout prêt, c'est toi qui l'as
# rangé dossier par dossier.
jeu_train = datasets.ImageFolder("dataset/train", transform=transformation)
jeu_test = datasets.ImageFolder("dataset/test", transform=transformation)
CLASSES = jeu_train.classes            # ex. ['poires', 'pommes'] (ordre alpha)

loader_train = DataLoader(jeu_train, batch_size=BATCH_SIZE, shuffle=True)
loader_test = DataLoader(jeu_test, batch_size=BATCH_SIZE)


# ----------------------------------------------------------------------
# 3) Le modèle : ResNet18 pré-entraîné, gelé, avec une tête neuve
# ----------------------------------------------------------------------
# On télécharge ResNet18 AVEC ses poids appris sur ImageNet (la 1re fois seulement).
modele = resnet18(weights=ResNet18_Weights.DEFAULT)

# On GÈLE tout le corps : ces poids ne seront pas modifiés pendant l'entraînement.
for parametre in modele.parameters():
    parametre.requires_grad = False

# On remplace la dernière couche (fc) par une couche NEUVE à 2 sorties. Elle est
# la seule entraînable : c'est une simple régression logistique sur les
# caractéristiques extraites par le corps gelé.
modele.fc = nn.Linear(modele.fc.in_features, len(CLASSES))
modele = modele.to(device)


# ----------------------------------------------------------------------
# 4) La perte et l'optimiseur (SEULE la tête est optimisée)
# ----------------------------------------------------------------------
perte = nn.CrossEntropyLoss()
# On ne passe QUE les paramètres de la tête à l'optimiseur : le reste est gelé.
optimiseur = torch.optim.SGD(modele.fc.parameters(), lr=LEARNING_RATE)


# ----------------------------------------------------------------------
# 5) Entraînement et évaluation (repris tels quels du TP 4)
# ----------------------------------------------------------------------
def entrainer_une_epoch():
    """Un passage complet sur le jeu d'entraînement, batch par batch."""
    modele.train()
    perte_totale = 0.0
    for images, cibles in loader_train:
        images, cibles = images.to(device), cibles.to(device)
        optimiseur.zero_grad()
        logits = modele(images)
        p = perte(logits, cibles)
        p.backward()
        optimiseur.step()
        perte_totale += p.item()
    return perte_totale / len(loader_train)


def evaluer(loader):
    """Renvoie (perte moyenne, accuracy) sur un loader, sans apprendre."""
    modele.eval()
    perte_totale = 0.0
    bonnes = 0
    total = 0
    with torch.no_grad():
        for images, cibles in loader:
            images, cibles = images.to(device), cibles.to(device)
            logits = modele(images)
            perte_totale += perte(logits, cibles).item()
            predictions = logits.argmax(dim=1)
            bonnes += (predictions == cibles).sum().item()
            total += cibles.size(0)
    return perte_totale / len(loader), bonnes / total


# ----------------------------------------------------------------------
# 6) La boucle d'entraînement
# ----------------------------------------------------------------------
def main():
    print(f"Device utilisé : {device}")
    n_entrainables = sum(p.numel() for p in modele.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in modele.parameters())
    print(f"Classes : {CLASSES}")
    print(f"Paramètres entraînables : {n_entrainables:,} / {n_total:,} "
          f"(le reste est gelé)")
    print(f"Images train : {len(jeu_train)} | test : {len(jeu_test)} "
          f"| batch = {BATCH_SIZE}\n")

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
    # 7) Résultat final
    # ------------------------------------------------------------------
    _, acc_train = evaluer(loader_train)
    _, acc_test = evaluer(loader_test)
    print(f"\nAccuracy train : {acc_train:.1%}")
    print(f"Accuracy test  : {acc_test:.1%}")

    # ------------------------------------------------------------------
    # 8) Visualisation : prédictions, courbe de loss, matrice de confusion
    # ------------------------------------------------------------------
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 5))

    # --- (a) Quelques images de test avec la prédiction (vert = ok, rouge = erreur) ---
    images, cibles = next(iter(loader_test))
    modele.eval()
    with torch.no_grad():
        predictions = modele(images.to(device)).argmax(dim=1).cpu()

    # On "dé-normalise" pour réafficher des couleurs correctes.
    moyenne = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    ecart = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    n_apercu = min(8, images.size(0))
    for i in range(n_apercu):
        ax = fig.add_subplot(2, n_apercu, i + 1)
        img = (images[i] * ecart + moyenne).clamp(0, 1).permute(1, 2, 0)
        ax.imshow(img)
        correct = predictions[i].item() == cibles[i].item()
        ax.set_title(CLASSES[predictions[i]], color="green" if correct else "red")
        ax.axis("off")

    # --- (b) Courbe de loss train vs test ---
    ax_loss = fig.add_subplot(2, 2, 3)
    ax_loss.plot(range(1, EPOCHS + 1), historique_train, label="loss entraînement", color="blue")
    ax_loss.plot(range(1, EPOCHS + 1), historique_test, label="loss test", color="red")
    ax_loss.set_title("TP 5 - Pommes vs poires : loss au fil des epochs")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss (cross-entropy)")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # --- (c) Matrice de confusion 2 x 2 ---
    # confusion[vrai, prédit] = nombre d'images de la classe "vrai" classées
    # comme "prédit". La diagonale = les bonnes réponses.
    n_classes = len(CLASSES)
    confusion = torch.zeros(n_classes, n_classes, dtype=torch.int)
    with torch.no_grad():
        for imgs, cibs in loader_test:
            preds = modele(imgs.to(device)).argmax(dim=1).cpu()
            for vrai, predit in zip(cibs, preds):
                confusion[vrai, predit] += 1

    ax_conf = fig.add_subplot(2, 2, 4)
    im = ax_conf.imshow(confusion, cmap="Blues")
    ax_conf.set_title("Matrice de confusion (test)")
    ax_conf.set_xlabel("classe prédite")
    ax_conf.set_ylabel("vraie classe")
    ax_conf.set_xticks(range(n_classes))
    ax_conf.set_yticks(range(n_classes))
    ax_conf.set_xticklabels(CLASSES)
    ax_conf.set_yticklabels(CLASSES)
    for vrai in range(n_classes):
        for predit in range(n_classes):
            ax_conf.text(predit, vrai, int(confusion[vrai, predit]),
                        ha="center", va="center")
    fig.colorbar(im, ax=ax_conf, fraction=0.046)

    plt.tight_layout()
    plt.savefig("resultat.png", dpi=120)
    print("\nGraphique enregistré dans resultat.png")


if __name__ == "__main__":
    main()

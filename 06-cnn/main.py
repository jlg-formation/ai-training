"""
TP 6 - Un vrai CNN sur MNIST : la CONVOLUTION

Au TP 4, on reconnaissait déjà les chiffres MNIST, mais avec un MLP : on
APLATISSAIT l'image 28 x 28 en un vecteur de 784 nombres. En faisant ça, on
jette une information capitale : la STRUCTURE SPATIALE. Le pixel (5, 5) et le
pixel (5, 6) sont voisins dans l'image, mais deviennent deux entrées quelconques
parmi 784 pour un MLP, qui doit tout réapprendre de zéro.

LA nouveauté du TP : la COUCHE DE CONVOLUTION (nn.Conv2d). Au lieu d'un poids par
pixel, on apprend de petits FILTRES (ici 3 x 3) qu'on fait GLISSER sur toute
l'image. Un même filtre détecte le même motif (un bord, un coin, une boucle)
PARTOUT dans l'image : c'est le partage de poids. On empile ces couches pour
détecter des motifs de plus en plus complexes.

    TP 4 : image -> APLATIR -> Linear(784->H) -> ReLU -> Linear(H->10)
    TP 6 : image -> [Conv -> BN -> ReLU -> Pool] x2 -> APLATIR -> Linear -> 10

Le reste (perte cross-entropy, optimiseur, boucle d'entraînement) est identique
au TP 4 : on ne change QUE le modèle. On en profite pour deux gestes utiles :
  - BatchNorm et Dropout : deux régularisations qui stabilisent et améliorent
    l'entraînement (on vise > 99 % sur le test) ;
  - l'EXPORT ONNX : on sauvegarde le modèle entraîné dans un format standard,
    lisible par n'importe quel langage. Le dossier front/ s'en servira pour
    reconnaître, DANS LE NAVIGATEUR, un chiffre dessiné à la souris.

Lancer avec :  uv run main.py
"""

import shutil
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, ToTensor

# Reproductibilité (comme au TP 4)
torch.manual_seed(0)


# ======================================================================
# CONSTANTES PÉDAGOGIQUES : les leviers à faire varier (voir README.md)
# ======================================================================
LEARNING_RATE = 1e-3  # pas de la descente de gradient (Adam aime les petits pas)
EPOCHS = 12  # passages complets sur le jeu d'entraînement
BATCH_SIZE = 128  # nombre d'images par mini-batch

# Normalisation standard de MNIST : moyenne et écart-type du jeu d'entraînement.
# On centre-réduit les pixels ((x - moy) / ecart_type). IMPORTANT : le front
# devra appliquer EXACTEMENT la même normalisation au chiffre dessiné.
MNIST_MOYENNE = 0.1307
MNIST_ECART_TYPE = 0.3081


# ----------------------------------------------------------------------
# 0) Le "device" : CPU ou GPU (identique au TP 4)
# ----------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------
# 1) Les données : MNIST, avec normalisation cette fois
# ----------------------------------------------------------------------
# Nouveauté par rapport au TP 4 : on compose deux transformations. ToTensor
# ramène les pixels dans [0, 1], puis Normalize les centre-réduit. Un CNN profond
# s'entraîne plus vite et plus stablement sur des entrées centrées-réduites.
transformation = Compose(
    [
        ToTensor(),
        Normalize((MNIST_MOYENNE,), (MNIST_ECART_TYPE,)),
    ]
)

jeu_train = datasets.MNIST(
    root="./data", train=True, download=True, transform=transformation
)
jeu_test = datasets.MNIST(
    root="./data", train=False, download=True, transform=transformation
)

loader_train = DataLoader(jeu_train, batch_size=BATCH_SIZE, shuffle=True)
loader_test = DataLoader(jeu_test, batch_size=BATCH_SIZE)


# ----------------------------------------------------------------------
# 2) Le modèle : un CNN (LA nouveauté du TP)
# ----------------------------------------------------------------------
# Deux blocs de convolution, puis un classifieur dense. On lit la topologie de
# haut en bas comme le trajet d'une image :
#
#   entrée (1, 28, 28)
#     Conv2d(1 -> 32, noyau 3x3, padding 1)  -> (32, 28, 28)   détecte des bords
#     BatchNorm + ReLU
#     Conv2d(32 -> 64, noyau 3x3, padding 1) -> (64, 28, 28)   motifs plus riches
#     BatchNorm + ReLU
#     MaxPool 2x2                            -> (64, 14, 14)    on divise par 2
#     Dropout                                                   régularisation
#
#     Conv2d(64 -> 128, noyau 3x3, padding 1)-> (128, 14, 14)  motifs complexes
#     BatchNorm + ReLU
#     MaxPool 2x2                            -> (128, 7, 7)     on divise encore
#     Dropout
#
#     APLATIR                                -> 128*7*7 = 6272
#     Linear(6272 -> 128) + ReLU + Dropout
#     Linear(128 -> 10)                      -> 10 LOGITS (pas de softmax ici)
#
# padding=1 avec un noyau 3x3 conserve la taille (28x28 reste 28x28) : ce sont
# les MaxPool qui réduisent la résolution, pas les convolutions.
class ReseauCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # --- Bloc 1 : deux convolutions puis un pooling ---
        self.bloc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (64, 28, 28) -> (64, 14, 14)
            nn.Dropout(0.25),
        )

        # --- Bloc 2 : une convolution plus profonde puis un pooling ---
        self.bloc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (128, 14, 14) -> (128, 7, 7)
            nn.Dropout(0.25),
        )

        # --- Classifieur dense : aplatir puis deux couches linéaires ---
        self.classifieur = nn.Sequential(
            nn.Flatten(),  # (128, 7, 7) -> 6272
            nn.Linear(128 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),  # 10 LOGITS bruts (softmax dans la perte)
        )

    def forward(self, x):
        x = self.bloc1(x)
        x = self.bloc2(x)
        return self.classifieur(x)


modele = ReseauCNN().to(device)


# ----------------------------------------------------------------------
# 3) La perte et l'optimiseur
# ----------------------------------------------------------------------
# Identiques au TP 4 dans l'esprit. On passe juste de SGD à Adam : un optimiseur
# qui adapte le pas pour chaque poids, souvent plus rapide à converger sur un CNN.
perte = nn.CrossEntropyLoss()
optimiseur = torch.optim.Adam(modele.parameters(), lr=LEARNING_RATE)


# ----------------------------------------------------------------------
# 4) Entraînement et évaluation (une epoch chacune) - repris du TP 4
# ----------------------------------------------------------------------
def entrainer_une_epoch():
    """Un passage complet sur le jeu d'entraînement, batch par batch."""
    modele.train()  # mode entraînement (BatchNorm/Dropout actifs)
    perte_totale = 0.0
    for images, cibles in loader_train:
        images, cibles = images.to(device), cibles.to(device)

        optimiseur.zero_grad()  # a) remet les gradients à zéro
        logits = modele(images)  # b) passe avant
        p = perte(logits, cibles)  # c) calcule la perte
        p.backward()  # d) autograd : tous les gradients
        optimiseur.step()  #    met à jour les poids

        perte_totale += p.item()
    return perte_totale / len(loader_train)


def evaluer(loader):
    """Renvoie (perte moyenne, accuracy) sur un loader, sans apprendre."""
    modele.eval()  # mode évaluation (BatchNorm/Dropout figés)
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
# 5) Export ONNX : sauvegarder le modèle pour le navigateur
# ----------------------------------------------------------------------
# ONNX (Open Neural Network Exchange) est un format standard de modèle. On y
# exporte le CNN entraîné pour que le front (onnxruntime-web) puisse l'exécuter
# DANS LE NAVIGATEUR, sans Python ni serveur.
def exporter_onnx(chemin_onnx):
    modele.eval()
    # Une entrée "factice" de la bonne forme (1 image, 1 canal, 28x28) : ONNX
    # trace le graphe du modèle en le faisant tourner une fois dessus.
    entree_factice = torch.randn(1, 1, 28, 28, device=device)
    torch.onnx.export(
        modele,
        entree_factice,
        chemin_onnx,
        input_names=["input"],
        output_names=["logits"],
        # L'axe 0 (le batch) est dynamique : le front enverra 1 image à la fois,
        # mais le modèle accepterait n'importe quel nombre d'images.
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13,
        dynamo=False,  # exportateur classique (pas besoin d'onnxscript)
    )

    # Validation : onnx.checker vérifie que le graphe exporté est cohérent.
    import onnx

    modele_onnx = onnx.load(chemin_onnx)
    onnx.checker.check_model(modele_onnx)


# ----------------------------------------------------------------------
# 6) La boucle d'entraînement
# ----------------------------------------------------------------------
def main():
    print(f"Device utilise : {device}")
    n_parametres = sum(p.numel() for p in modele.parameters())
    print(f"Parametres du modele : {n_parametres:,}")
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
    # Résultat final
    # ------------------------------------------------------------------
    _, acc_train = evaluer(loader_train)
    _, acc_test = evaluer(loader_test)
    print(f"\nAccuracy train : {acc_train:.1%}")
    print(f"Accuracy test  : {acc_test:.1%}")

    # ------------------------------------------------------------------
    # Export ONNX (vers le dossier front/public pour le navigateur)
    # ------------------------------------------------------------------
    chemin_onnx = "modele.onnx"
    exporter_onnx(chemin_onnx)
    print(f"\nModele exporte et valide : {chemin_onnx}")

    # On copie le modèle là où le front Vite le servira (front/public/).
    dossier_public = Path("front") / "public"
    dossier_public.mkdir(parents=True, exist_ok=True)
    shutil.copy(chemin_onnx, dossier_public / "modele.onnx")
    print(f"Modele copie dans {dossier_public / 'modele.onnx'}")

    # ------------------------------------------------------------------
    # Visualisation : prédictions, courbe de loss, matrice de confusion
    # (repris du TP 4)
    # ------------------------------------------------------------------
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 5))

    # --- (a) Une poignée d'images de test avec la prédiction ---
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
    ax_loss.set_title("TP 6 - CNN MNIST : loss au fil des epochs")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss (cross-entropy)")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # --- (c) Matrice de confusion 10 x 10 ---
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
    print("\nGraphique enregistre dans resultat.png")


if __name__ == "__main__":
    main()

"""
TP 7 - GAN : deux reseaux qui s'AFFRONTENT pour GENERER des chiffres

Jusqu'ici, tous nos reseaux etaient DISCRIMINATIFS : on leur montrait une image
et ils repondaient une etiquette (un prix au TP 1, spam/pas spam au TP 2, un
chiffre 0-9 aux TP 4 et 6). Ce TP bascule dans le GENERATIF : au lieu de
RECONNAITRE un chiffre, on veut en FABRIQUER de nouveaux, qui ressemblent a de
vrais MNIST mais n'existent dans aucun jeu de donnees.

LA nouveaute du TP : l'entrainement ADVERSAIRE (Generative Adversarial Network,
Goodfellow 2014). On entraine DEUX reseaux en meme temps, l'un contre l'autre :

  - le GENERATEUR (G) part d'un vecteur de BRUIT aleatoire z et produit une image
    28x28. C'est un faussaire : il essaie de fabriquer de faux billets.
  - le DISCRIMINATEUR (D) recoit une image et dit si elle est VRAIE (issue de
    MNIST) ou FAUSSE (fabriquee par G). C'est le policier qui traque les faux.

Les deux progressent ensemble : D apprend a mieux reperer les faux, ce qui force
G a faire des faux plus credibles, ce qui force D a s'ameliorer encore... A la
fin, G produit des chiffres si realistes que D ne fait plus la difference (il
repond "vrai" environ une fois sur deux, au hasard).

Ce qu'on reutilise des TP precedents (rien de neuf ici) :
  - MNIST via torchvision + DataLoader (TP 4)
  - des MLP nn.Linear (TP 3/4), .to(device), la boucle d'entrainement
  - la perte BCE (entropie croisee binaire) du TP 2 : ici D fait une
    classification binaire "vrai vs faux".

Ce qui est VRAIMENT nouveau :
  - DEUX modeles, DEUX optimiseurs, DEUX pertes dans la meme boucle
  - pas d'"accuracy" ni de jeu de test : on JUGE le resultat a l'oeil, sur les
    images generees (resultat.png).

Lancer avec :  uv run main.py
"""

import shutil
import time
import warnings
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, ToTensor

# Reproductibilite (comme aux TP 4 et 6)
torch.manual_seed(0)


# ======================================================================
# CONSTANTES PEDAGOGIQUES : les leviers a faire varier (voir README.md)
# ======================================================================
DIM_BRUIT = 64  # taille du vecteur de bruit z (l'entree du generateur)
LEARNING_RATE = 2e-4  # petit pas : les GAN sont instables, Adam aime les petits pas
EPOCHS = 40  # passages complets sur le jeu d'entrainement
BATCH_SIZE = 128  # nombre d'images par mini-batch
BETA1 = 0.5  # Adam : on baisse beta1 de 0.9 a 0.5 (convention GAN, stabilise)


# ----------------------------------------------------------------------
# 0) Le "device" : CPU ou GPU (identique aux TP 4 et 6)
# ----------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------
# 1) Les donnees : MNIST normalise dans [-1, 1]
# ----------------------------------------------------------------------
# Nouveaute par rapport au TP 6 : on centre-reduit avec moyenne 0.5 et
# ecart-type 0.5, ce qui ramene les pixels de [0, 1] vers [-1, 1]. C'est
# volontaire : la derniere couche du generateur est une Tanh, qui sort justement
# dans [-1, 1]. Les vraies et les fausses images vivent ainsi dans le MEME
# intervalle, sinon D distinguerait les faux juste a leur plage de valeurs.
transformation = Compose(
    [
        ToTensor(),
        Normalize((0.5,), (0.5,)),
    ]
)

jeu_train = datasets.MNIST(
    root="./data", train=True, download=True, transform=transformation
)

# drop_last=True : on jette le dernier mini-batch incomplet pour que tous les
# batchs aient exactement BATCH_SIZE images (plus simple pour les etiquettes).
loader_train = DataLoader(
    jeu_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)


# ----------------------------------------------------------------------
# 2) Le GENERATEUR : bruit (64) -> image (1, 28, 28)
# ----------------------------------------------------------------------
# Un MLP qui remonte le bruit vers une image : 64 -> 160 -> 256 -> 784. La
# derniere couche est une Tanh, donc chaque pixel sort dans [-1, 1], comme les
# vraies images normalisees ci-dessus. (~253 000 parametres.)
class Generateur(nn.Module):
    def __init__(self):
        super().__init__()
        self.reseau = nn.Sequential(
            nn.Linear(DIM_BRUIT, 160),
            nn.ReLU(),
            nn.Linear(160, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Tanh(),  # pixels dans [-1, 1]
        )

    def forward(self, z):
        image_plate = self.reseau(z)  # (N, 784)
        return image_plate.view(z.size(0), 1, 28, 28)  # (N, 1, 28, 28)


# ----------------------------------------------------------------------
# 3) Le DISCRIMINATEUR : image (1, 28, 28) -> 1 logit (vrai vs faux)
# ----------------------------------------------------------------------
# Un MLP de classification binaire, exactement dans l'esprit du TP 2, mais sur
# 784 entrees. Il sort UN logit brut ; la sigmoide est appliquee par la perte
# BCEWithLogitsLoss (comme la cross-entropy des TP 4/6 applique le softmax).
#   - LeakyReLU : variante de ReLU qui laisse passer un petit gradient meme pour
#     les entrees negatives ; c'est la convention pour les GAN (evite que le
#     discriminateur "meure" et cesse d'envoyer du gradient au generateur).
#   - Dropout : regularisation (vue au TP 6), utile pour ne pas rendre D trop
#     fort trop vite.
class Discriminateur(nn.Module):
    def __init__(self):
        super().__init__()
        self.reseau = nn.Sequential(
            nn.Flatten(),  # (N, 1, 28, 28) -> (N, 784)
            nn.Linear(28 * 28, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 160),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(160, 1),  # 1 logit brut (pas de sigmoide ici)
        )

    def forward(self, image):
        return self.reseau(image)


generateur = Generateur().to(device)
discriminateur = Discriminateur().to(device)


# ----------------------------------------------------------------------
# 4) DEUX pertes, DEUX optimiseurs (le coeur de la nouveaute)
# ----------------------------------------------------------------------
# Une seule fonction de perte (BCE binaire), mais DEUX optimiseurs distincts :
# un pas de G ne doit toucher QUE les poids de G, un pas de D que ceux de D.
perte = nn.BCEWithLogitsLoss()
opt_g = torch.optim.Adam(
    generateur.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999)
)
opt_d = torch.optim.Adam(
    discriminateur.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999)
)

# Un bruit FIXE, tire une fois pour toutes : on le repasse dans G a la fin (et on
# pourrait le faire a chaque epoch) pour voir les MEMES 64 images s'ameliorer.
bruit_fixe = torch.randn(64, DIM_BRUIT, device=device)


# ----------------------------------------------------------------------
# 5) Une epoch d'entrainement adversaire
# ----------------------------------------------------------------------
def entrainer_une_epoch():
    """Un passage complet sur MNIST : a chaque batch, on entraine D puis G."""
    generateur.train()
    discriminateur.train()
    perte_d_totale = 0.0
    perte_g_totale = 0.0

    for images_reelles, _ in loader_train:  # on ignore les etiquettes 0-9 !
        images_reelles = images_reelles.to(device)
        n = images_reelles.size(0)

        # Les cibles du discriminateur : 1 = "vrai", 0 = "faux".
        cible_vrai = torch.ones(n, 1, device=device)
        cible_faux = torch.zeros(n, 1, device=device)

        # === 1) Entrainer le DISCRIMINATEUR ============================
        # But : bien classer les vraies (-> 1) ET les fausses (-> 0).
        bruit = torch.randn(n, DIM_BRUIT, device=device)
        images_fausses = generateur(bruit)

        # detach() : on coupe le lien vers G. Ici on ne met a jour que D, on ne
        # veut PAS que ce gradient remonte dans le generateur.
        perte_sur_vraies = perte(discriminateur(images_reelles), cible_vrai)
        perte_sur_fausses = perte(discriminateur(images_fausses.detach()), cible_faux)
        perte_d = perte_sur_vraies + perte_sur_fausses

        opt_d.zero_grad()
        perte_d.backward()
        opt_d.step()

        # === 2) Entrainer le GENERATEUR ===============================
        # But : TROMPER D, c.-a-d. lui faire classer les fausses comme VRAIES.
        # On reutilise les memes images fausses, mais SANS detach : le gradient
        # traverse D (fige ici) et remonte jusque dans les poids de G.
        perte_g = perte(discriminateur(images_fausses), cible_vrai)

        opt_g.zero_grad()
        perte_g.backward()
        opt_g.step()

        perte_d_totale += perte_d.item()
        perte_g_totale += perte_g.item()

    n_batchs = len(loader_train)
    return perte_d_totale / n_batchs, perte_g_totale / n_batchs


# ----------------------------------------------------------------------
# 5 bis) Apercu de fin d'epoch : 10 images AU HASARD
# ----------------------------------------------------------------------
# A la fin de chaque epoch, on tire un NOUVEAU bruit (10 vecteurs au hasard) et
# on enregistre les 10 images generees en une bandelette, pour suivre a l'oeil
# la progression du generateur. Ce dossier est ignore par Git (voir .gitignore).
DOSSIER_APERCUS = Path("apercus")


def enregistrer_apercu(numero_epoch):
    """Genere 10 images au hasard et les enregistre dans apercus/."""
    import matplotlib.pyplot as plt

    generateur.eval()
    with torch.no_grad():
        bruit = torch.randn(10, DIM_BRUIT, device=device)
        images = generateur(bruit).cpu()
    generateur.train()

    # Pixels [-1, 1] (sortie Tanh) -> [0, 1] pour l'affichage.
    images = (images + 1.0) / 2.0

    # On colle les 10 images cote a cote en une seule bandelette 28 x (10*28).
    bandelette = torch.zeros(28, 10 * 28)
    for k in range(10):
        bandelette[:, k * 28 : (k + 1) * 28] = images[k, 0]

    DOSSIER_APERCUS.mkdir(exist_ok=True)
    chemin = DOSSIER_APERCUS / f"epoch_{numero_epoch:02d}.png"
    # cmap "gray_r" : on inverse (fond blanc, chiffre noir).
    plt.imsave(chemin, bandelette.numpy(), cmap="gray_r")


# ----------------------------------------------------------------------
# 5 ter) Export ONNX du GENERATEUR (pour le site web)
# ----------------------------------------------------------------------
# On exporte UNIQUEMENT le generateur : le site tirera un bruit z au hasard,
# l'enverra au modele et affichera l'image produite. Le discriminateur, lui, ne
# sert qu'a l'entrainement et n'a aucune utilite apres coup.
def exporter_generateur_onnx(chemin_onnx):
    generateur.eval()
    # Entree factice : 1 vecteur de bruit de taille DIM_BRUIT. ONNX trace le
    # graphe du modele en le faisant tourner une fois dessus.
    bruit_factice = torch.randn(1, DIM_BRUIT, device=device)
    # On garde l'exportateur classique (dynamo=False) : noms d'E/S stables pour
    # le site et pas de dependance onnxscript. Il est deprecie -> on tait juste
    # l'avertissement pour ne pas polluer la sortie.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        torch.onnx.export(
            generateur,
            bruit_factice,
            chemin_onnx,
            input_names=["bruit"],
            output_names=["image"],
            # L'axe 0 (le batch) est dynamique : le site enverra 1 bruit a la fois.
            dynamic_axes={"bruit": {0: "batch"}, "image": {0: "batch"}},
            opset_version=13,
            dynamo=False,  # exportateur classique (pas besoin d'onnxscript)
        )

    import onnx

    onnx.checker.check_model(onnx.load(chemin_onnx))


# ----------------------------------------------------------------------
# 6) La boucle d'entrainement
# ----------------------------------------------------------------------
def main():
    print(f"Device utilise : {device}")
    n_param_g = sum(p.numel() for p in generateur.parameters())
    n_param_d = sum(p.numel() for p in discriminateur.parameters())
    print(f"Parametres generateur    : {n_param_g:,}")
    print(f"Parametres discriminateur : {n_param_d:,}")
    print(
        f"Images train : {len(jeu_train)} | batch = {BATCH_SIZE} | bruit z = {DIM_BRUIT}\n"
    )

    historique_d = []
    historique_g = []

    for epoch in range(EPOCHS):
        t0 = time.time()
        loss_d, loss_g = entrainer_une_epoch()
        historique_d.append(loss_d)
        historique_g.append(loss_g)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | perte_D = {loss_d:.4f} "
            f"| perte_G = {loss_g:.4f} | {time.time() - t0:.1f} s"
        )
        # Apercu de 10 images au hasard, pour suivre la progression a l'oeil.
        enregistrer_apercu(epoch + 1)

    # ------------------------------------------------------------------
    # Export ONNX du generateur (vers front/public pour le navigateur)
    # ------------------------------------------------------------------
    chemin_onnx = "generateur.onnx"
    exporter_generateur_onnx(chemin_onnx)
    print(f"\nGenerateur exporte et valide : {chemin_onnx}")

    dossier_public = Path("front") / "public"
    dossier_public.mkdir(parents=True, exist_ok=True)
    shutil.copy(chemin_onnx, dossier_public / "generateur.onnx")
    print(f"Modele copie dans {dossier_public / 'generateur.onnx'}")

    # ------------------------------------------------------------------
    # 7) Visualisation : chiffres generes + courbes de perte
    # ------------------------------------------------------------------
    import matplotlib.pyplot as plt

    # On genere 64 images a partir du bruit FIXE, en mode eval (Dropout fige).
    generateur.eval()
    with torch.no_grad():
        images_generees = generateur(bruit_fixe).cpu()

    # Les pixels sont dans [-1, 1] (sortie Tanh) : on repasse dans [0, 1] pour
    # l'affichage ( (x + 1) / 2 ).
    images_generees = (images_generees + 1.0) / 2.0

    # On assemble les 64 images (1, 28, 28) en une grande mosaique 8x8.
    grille = torch.zeros(8 * 28, 8 * 28)
    for i in range(8):
        for j in range(8):
            image = images_generees[i * 8 + j, 0]
            grille[i * 28 : (i + 1) * 28, j * 28 : (j + 1) * 28] = image

    fig = plt.figure(figsize=(12, 6))

    # --- (a) La mosaique des chiffres generes ---
    ax_grille = fig.add_subplot(1, 2, 1)
    ax_grille.imshow(grille, cmap="gray")
    ax_grille.set_title("TP 7 - GAN : 64 chiffres GENERES (aucun n'existe dans MNIST)")
    ax_grille.axis("off")

    # --- (b) Les deux courbes de perte (D et G) ---
    # A lire ensemble : elles montrent le bras de fer. Un GAN "sain" garde les
    # deux pertes dans un equilibre, sans qu'aucune ne s'effondre a zero.
    ax_loss = fig.add_subplot(1, 2, 2)
    ax_loss.plot(
        range(1, EPOCHS + 1), historique_d, label="perte discriminateur", color="red"
    )
    ax_loss.plot(
        range(1, EPOCHS + 1), historique_g, label="perte generateur", color="blue"
    )
    ax_loss.set_title("Pertes au fil des epochs (le bras de fer)")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("perte (BCE)")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("resultat.png", dpi=120)
    print("\nGraphique enregistre dans resultat.png")


if __name__ == "__main__":
    main()

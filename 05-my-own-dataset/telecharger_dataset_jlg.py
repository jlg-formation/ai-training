"""
TP 5 - Télécharger LE dataset de JLG (pommes / poires) tel quel

Ce script est une ALTERNATIVE à telecharger_dataset.py : au lieu d'aller pêcher
des images un peu partout (résultat différent à chaque fois, images à trier), il
récupère UN dataset figé, déjà constitué et déjà trié, hébergé en ligne :

    https://jlg-consulting.com/orsys/DRN/dataset-pommes-poires/dataset.zip

Intérêt pédagogique : TOUT LE MONDE part des MÊMES images. Combiné au
`torch.manual_seed(0)` de main.py, l'entraînement devient alors **parfaitement
reproductible** — pratique pour comparer des réglages sans que le dataset change.

Le .zip contient déjà l'arborescence attendue :

    dataset/train/pommes   dataset/train/poires
    dataset/test/pommes    dataset/test/poires

Lancer :  uv run telecharger_dataset_jlg.py

ATTENTION : ce script REMPLACE le dossier dataset/ existant par la version de
JLG. Si tu avais trié des images à la main, sauvegarde-les avant.
"""

import io
import os
import shutil
import time
import urllib.error
import urllib.request
import zipfile

# ======================================================================
# RÉGLAGES
# ======================================================================
URL = "https://jlg-consulting.com/orsys/DRN/dataset-pommes-poires/dataset.zip"
DOSSIER = "dataset"        # où extraire (le zip contient déjà "dataset/...")
MAX_TENTATIVES = 4         # essais de téléchargement avant d'abandonner
ATTENTE = 5                # pause (s) entre deux tentatives ratées
ENTETE = {"User-Agent": "ai-training-tp5/1.0 (exercice pedagogique)"}


# ----------------------------------------------------------------------
# 1) Télécharger le .zip en mémoire (avec quelques tentatives)
# ----------------------------------------------------------------------
def telecharger_zip():
    """Renvoie le contenu brut du .zip, ou None si tout échoue."""
    for tentative in range(1, MAX_TENTATIVES + 1):
        try:
            print(f"Téléchargement de {URL}")
            print(f"  tentative {tentative}/{MAX_TENTATIVES}...")
            requete = urllib.request.Request(URL, headers=ENTETE)
            with urllib.request.urlopen(requete, timeout=60) as reponse:
                donnees = reponse.read()
            print(f"  reçu : {len(donnees) / 1_000_000:.1f} Mo")
            return donnees
        except (urllib.error.URLError, TimeoutError) as erreur:
            print(f"  échec ({erreur}) — pause {ATTENTE}s puis nouvel essai")
            time.sleep(ATTENTE)
    print("Impossible de télécharger le dataset. Vérifie ta connexion.")
    return None


# ----------------------------------------------------------------------
# 2) Extraire le .zip, en remplaçant proprement l'ancien dossier dataset/
# ----------------------------------------------------------------------
def extraire(donnees):
    """Décompresse le .zip dans le dossier courant, écrase l'ancien dataset/."""
    if os.path.isdir(DOSSIER):
        print(f"Suppression de l'ancien dossier '{DOSSIER}/' (remplacé par JLG)")
        shutil.rmtree(DOSSIER)

    print("Extraction du .zip...")
    with zipfile.ZipFile(io.BytesIO(donnees)) as archive:
        archive.extractall(".")            # le zip contient déjà "dataset/..."


# ----------------------------------------------------------------------
# 3) Vérifier ce qui a été installé
# ----------------------------------------------------------------------
def compter():
    """Affiche le nombre d'images par dossier et signale les dossiers vides."""
    print("\n================= BILAN =================")
    tout_present = True
    for split in ("train", "test"):
        for classe in ("pommes", "poires"):
            chemin = os.path.join(DOSSIER, split, classe)
            nb = len(os.listdir(chemin)) if os.path.isdir(chemin) else 0
            tout_present = tout_present and nb > 0
            print(f"  {split}/{classe:8s} : {nb} images   {'OK' if nb else 'VIDE !'}")
    return tout_present


# ----------------------------------------------------------------------
# 4) Boucle principale
# ----------------------------------------------------------------------
def main():
    donnees = telecharger_zip()
    if donnees is None:
        return
    extraire(donnees)
    if compter():
        print("\nDataset de JLG installé. Tu peux lancer :  uv run main.py")
    else:
        print("\nATTENTION : dataset incomplet. Relance le script.")


if __name__ == "__main__":
    main()

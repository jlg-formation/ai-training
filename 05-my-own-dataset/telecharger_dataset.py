"""
TP 5 - Récupération d'images (pommes / poires) depuis Internet

Ce script CONSTRUIT le dataset à ta place en puisant dans PLUSIEURS sources
d'images, ce qui le rend bien plus robuste qu'une source unique :

    - Bing Images (via la librairie icrawler)  -> AUCUNE clé, source principale
    - Pexels        (API)                       -> clé gratuite (voir plus bas)
    - Pixabay       (API)                       -> clé gratuite
    - Unsplash      (API)                       -> clé gratuite
    - Wikimedia Commons (API)                   -> AUCUNE clé, source d'appoint

Les images sont rangées dans :

    dataset/train/pommes   dataset/train/poires
    dataset/test/pommes    dataset/test/poires

TOUT L'INTÉRÊT DU TP est là : une collecte automatique renvoie des images
BRUITÉES — dessins, logos, mauvaises espèces... Après le téléchargement, OUVRE
les dossiers et FAIS LE TRI À LA MAIN : supprime tout ce qui ne montre pas
clairement le bon fruit. C'est ça, "construire un dataset de qualité".

------------------------------------------------------------------------------
CLÉS D'API (facultatif) : Bing + Wikimedia fonctionnent SANS clé. Pour activer
Pexels/Pixabay/Unsplash, crée un compte gratuit sur chaque site, récupère ta
clé, et déclare-la en variable d'environnement AVANT de lancer le script :

    PowerShell :  $env:PEXELS_API_KEY="..."; $env:PIXABAY_API_KEY="..."; $env:UNSPLASH_ACCESS_KEY="..."
    cmd.exe    :  set PEXELS_API_KEY=...   (idem pour les autres)

Une source sans clé est simplement SAUTÉE (message dans les logs). On ne code
JAMAIS une clé en dur dans le fichier.
------------------------------------------------------------------------------

Lancer :  uv run telecharger_dataset.py
"""

import json
import logging
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from io import BytesIO

from PIL import Image

try:
    from icrawler.builtin import BingImageCrawler
except ImportError:                            # icrawler non installé
    BingImageCrawler = None


# ======================================================================
# RÉGLAGES : les leviers à faire varier pour la leçon "qualité des données"
# ======================================================================
# Pour chaque classe : le nom du dossier -> les mots-clés de recherche.
CLASSES = {
    "pommes": "apple fruit",
    "poires": "pear fruit",
}
IMAGES_PAR_CLASSE = 80      # nombre visé par classe (AVANT le tri manuel)
PROPORTION_TEST = 0.2       # 20 % des images -> test/, le reste -> train/
TAILLE_MAX = 512            # on réduit les images trop grandes (côté max, pixels)

# Robustesse réseau (pour les téléchargements par URL)
DELAI_POLITESSE = 0.3       # pause (s) entre deux requêtes qui réussissent
MAX_TENTATIVES = 4          # essais par requête avant d'abandonner
ATTENTE_INITIALE = 5        # 1re pause (s) après un 429 ; elle double à chaque essai
ATTENTE_MAX = 60            # plafond de la pause (s)

# Clés d'API lues dans l'environnement (None si non définies -> source sautée).
CLE_PEXELS = os.environ.get("PEXELS_API_KEY")
CLE_PIXABAY = os.environ.get("PIXABAY_API_KEY")
CLE_UNSPLASH = os.environ.get("UNSPLASH_ACCESS_KEY")

# User-Agent explicite (obligatoire pour Wikimedia, poli pour les autres).
ENTETE = {"User-Agent": "ai-training-tp5/1.0 (exercice pedagogique)"}


# ----------------------------------------------------------------------
# 0) Outils bas niveau : lecture réseau robuste + normalisation d'image
# ----------------------------------------------------------------------
def lire_octets(url, quoi, entetes=None):
    """Renvoie le contenu brut d'une URL, ou None. Réessaie sur 429 / erreur réseau."""
    attente = ATTENTE_INITIALE
    for tentative in range(1, MAX_TENTATIVES + 1):
        try:
            requete = urllib.request.Request(url, headers={**ENTETE, **(entetes or {})})
            with urllib.request.urlopen(requete, timeout=30) as reponse:
                return reponse.read()
        except urllib.error.HTTPError as erreur:
            if erreur.code == 429:             # trop de requêtes : on patiente
                print(f"      rate-limit (429) sur {quoi} — pause {attente}s "
                      f"puis reessai {tentative}/{MAX_TENTATIVES}")
                time.sleep(attente)
                attente = min(attente * 2, ATTENTE_MAX)
                continue
            print(f"      {quoi} ignore (HTTP {erreur.code})")
            return None
        except (urllib.error.URLError, TimeoutError) as erreur:
            print(f"      souci reseau sur {quoi} ({erreur}) — pause {attente}s "
                  f"puis reessai {tentative}/{MAX_TENTATIVES}")
            time.sleep(attente)
            attente = min(attente * 2, ATTENTE_MAX)
    print(f"      {quoi} abandonne apres {MAX_TENTATIVES} tentatives")
    return None


def normaliser(image):
    """Ramène une image en RGB et la réduit à TAILLE_MAX (côté max)."""
    image = image.convert("RGB")
    image.thumbnail((TAILLE_MAX, TAILLE_MAX))
    return image


def image_depuis_url(url, quoi):
    """Télécharge une URL et renvoie une image PIL normalisée, ou None."""
    brut = lire_octets(url, quoi)
    if brut is None:
        return None
    try:
        return normaliser(Image.open(BytesIO(brut)))
    except Exception as erreur:                # fichier illisible (format exotique)
        print(f"      {quoi} illisible ({erreur})")
        return None


def image_depuis_fichier(chemin):
    """Charge un fichier image et renvoie une image PIL normalisée, ou None."""
    try:
        with Image.open(chemin) as image:
            return normaliser(image.copy())
    except Exception:
        return None


def json_api(url, entetes=None):
    """Interroge une API JSON. Renvoie le dict décodé, ou {} en cas d'échec."""
    brut = lire_octets(url, "API", entetes)
    if brut is None:
        return {}
    try:
        return json.loads(brut)
    except json.JSONDecodeError:
        return {}


# ----------------------------------------------------------------------
# 1) Les sources qui renvoient des URLs d'images
# ----------------------------------------------------------------------
def source_pexels(mots_cles):
    if not CLE_PEXELS:
        print("  - Pexels   : sauté (pas de clé PEXELS_API_KEY)")
        return []
    url = "https://api.pexels.com/v1/search?" + urllib.parse.urlencode(
        {"query": mots_cles, "per_page": min(IMAGES_PAR_CLASSE, 80)})
    donnees = json_api(url, {"Authorization": CLE_PEXELS})
    urls = [photo["src"]["large"] for photo in donnees.get("photos", [])]
    print(f"  - Pexels   : {len(urls)} images")
    return urls


def source_pixabay(mots_cles):
    if not CLE_PIXABAY:
        print("  - Pixabay  : sauté (pas de clé PIXABAY_API_KEY)")
        return []
    url = "https://pixabay.com/api/?" + urllib.parse.urlencode(
        {"key": CLE_PIXABAY, "q": mots_cles, "image_type": "photo",
         "per_page": min(IMAGES_PAR_CLASSE, 200)})
    donnees = json_api(url)
    urls = [hit["webformatURL"] for hit in donnees.get("hits", [])]
    print(f"  - Pixabay  : {len(urls)} images")
    return urls


def source_unsplash(mots_cles):
    if not CLE_UNSPLASH:
        print("  - Unsplash : sauté (pas de clé UNSPLASH_ACCESS_KEY)")
        return []
    url = "https://api.unsplash.com/search/photos?" + urllib.parse.urlencode(
        {"query": mots_cles, "per_page": 30})       # 30 = maximum Unsplash
    donnees = json_api(url, {"Authorization": f"Client-ID {CLE_UNSPLASH}"})
    urls = [r["urls"]["regular"] for r in donnees.get("results", [])]
    print(f"  - Unsplash : {len(urls)} images")
    return urls


def source_wikimedia(mots_cles):
    """Wikimedia Commons, avec pagination (l'API rend 50 résultats max par page)."""
    urls = []
    offset = 0
    while len(urls) < IMAGES_PAR_CLASSE:
        params = urllib.parse.urlencode({
            "action": "query", "format": "json", "generator": "search",
            "gsrsearch": mots_cles, "gsrnamespace": 6, "gsrlimit": 50,
            "gsroffset": offset, "prop": "imageinfo", "iiprop": "url",
            "iiurlwidth": TAILLE_MAX,
        })
        donnees = json_api("https://commons.wikimedia.org/w/api.php?" + params)
        pages = donnees.get("query", {}).get("pages", {})
        if not pages:
            break
        for page in sorted(pages.values(), key=lambda p: p.get("index", 0)):
            infos = page.get("imageinfo")
            if infos:
                url = infos[0].get("thumburl") or infos[0].get("url")
                if url:
                    urls.append(url)
        suite = donnees.get("continue")
        if not suite or "gsroffset" not in suite:
            break
        offset = suite["gsroffset"]
    print(f"  - Wikimedia: {len(urls)} images")
    return urls


# ----------------------------------------------------------------------
# 2) La source Bing, qui télécharge DIRECTEMENT dans un dossier (via icrawler)
# ----------------------------------------------------------------------
def source_bing(mots_cles, dossier):
    """Télécharge des images Bing dans `dossier` grâce à icrawler (sans clé)."""
    if BingImageCrawler is None:
        print("  - Bing     : sauté (icrawler non installé : uv add icrawler)")
        return
    try:
        crawler = BingImageCrawler(
            downloader_threads=4,
            storage={"root_dir": dossier},
            log_level=logging.WARNING,         # icrawler est très bavard par défaut
        )
        crawler.crawl(keyword=mots_cles, max_num=IMAGES_PAR_CLASSE)
    except Exception as erreur:
        print(f"  - Bing     : erreur icrawler ({erreur})")


# ----------------------------------------------------------------------
# 3) Traiter une classe : collecter depuis toutes les sources, puis répartir
# ----------------------------------------------------------------------
def traiter_classe(classe, mots_cles):
    """Rassemble les images d'une classe et les enregistre. Renvoie le nb gardé."""
    print(f"\n=== Classe '{classe}' (recherche : '{mots_cles}') ===")
    dossier_train = os.path.join("dataset", "train", classe)
    dossier_test = os.path.join("dataset", "test", classe)
    os.makedirs(dossier_train, exist_ok=True)
    os.makedirs(dossier_test, exist_ok=True)

    images = []            # liste d'images PIL déjà normalisées (RGB, réduites)

    # (A) Bing via icrawler : télécharge dans un dossier temporaire, puis on charge.
    dossier_bing = os.path.join("dataset", "_tmp_bing", classe)
    os.makedirs(dossier_bing, exist_ok=True)
    print("  -> collecte Bing (icrawler)...")
    source_bing(mots_cles, dossier_bing)
    for nom in sorted(os.listdir(dossier_bing)):
        image = image_depuis_fichier(os.path.join(dossier_bing, nom))
        if image is not None:
            images.append(image)
    print(f"  - Bing     : {len(images)} images exploitables")

    # (B) Sources par URL (les sans-clé + celles dont la clé est fournie).
    print("  -> collecte des URLs (Pexels, Pixabay, Unsplash, Wikimedia)...")
    urls = []
    urls += source_pexels(mots_cles)
    urls += source_pixabay(mots_cles)
    urls += source_unsplash(mots_cles)
    urls += source_wikimedia(mots_cles)
    urls = list(dict.fromkeys(urls))           # dédoublonne en gardant l'ordre

    for i, url in enumerate(urls, start=1):
        if len(images) >= IMAGES_PAR_CLASSE:
            break
        nom_court = urllib.parse.unquote(url.rsplit("/", 1)[-1])[:50]
        print(f"  [{i}/{len(urls)}] {nom_court}")
        image = image_depuis_url(url, nom_court)
        if image is not None:
            images.append(image)
            print(f"      OK {image.width}x{image.height}  (gardees : {len(images)})")
        time.sleep(DELAI_POLITESSE)

    # (C) Répartition train/test et enregistrement.
    pas_test = round(1 / PROPORTION_TEST)      # ex. 0.2 -> 1 image sur 5 en test
    for i, image in enumerate(images):
        en_test = (i % pas_test == 0)          # 1re, 6e, 11e... -> test/
        dossier = dossier_test if en_test else dossier_train
        image.save(os.path.join(dossier, f"{classe}_{i:03d}.jpg"), "JPEG", quality=90)

    shutil.rmtree(os.path.join("dataset", "_tmp_bing", classe), ignore_errors=True)
    print(f"  -> {len(images)} images gardees pour '{classe}'")
    return len(images)


# ----------------------------------------------------------------------
# 4) Boucle principale
# ----------------------------------------------------------------------
def main():
    total = {}
    for classe, mots_cles in CLASSES.items():
        total[classe] = traiter_classe(classe, mots_cles)

    shutil.rmtree(os.path.join("dataset", "_tmp_bing"), ignore_errors=True)

    print("\n================= BILAN =================")
    for classe, nb in total.items():
        print(f"  {classe:8s} : {nb} images   {'OK' if nb else 'VIDE !'}")

    if any(nb == 0 for nb in total.values()):
        print("\nATTENTION : une classe est vide. Vérifie ta connexion et relance")
        print("le script (uv run telecharger_dataset.py) — il complètera ce qui manque.")
        return

    print("\nTermine. OUVRE MAINTENANT les dossiers dataset/ et supprime les")
    print("images qui ne montrent pas clairement le bon fruit :")
    print("c'est l'etape 'qualite' du TP, et c'est elle qui fait la difference.")


if __name__ == "__main__":
    main()

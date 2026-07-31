# TP 5 — Ton propre dataset (pommes vs poires)

## Objectif

Jusqu'ici, les données arrivaient toutes prêtes (MNIST au TP 4). **Cette fois,
c'est toi qui fabriques le dataset.** La nouveauté du TP n'est ni le modèle ni
l'outil : c'est **la donnée elle-même**. Tu vas découvrir que, dans le monde
réel, l'essentiel du travail est **de constituer et nettoyer un bon jeu
d'images** — pas de coder le réseau.

```
TP 4 :  MNIST fourni, propre, 28x28 gris   -> MLP entraîné de zéro
TP 5 :  TES photos, hétérogènes, couleur   -> ResNet18 gelé + tête neuve
```

## MNIST vs ton dataset : ce qui change

| MNIST (TP 4)                     | Ton dataset (TP 5)                         |
| -------------------------------- | ------------------------------------------ |
| téléchargé tout prêt             | **c'est toi qui le construis**             |
| déjà propre et étiqueté          | **bruité : à trier à la main**             |
| toutes les images en 28×28       | tailles variées → il faut **redimensionner** |
| niveaux de gris (1 canal)        | couleur (3 canaux RGB)                     |
| 60 000 images                    | quelques dizaines → risque de **surapprentissage** |
| classes parfaitement équilibrées | à toi de surveiller le **déséquilibre**    |

## L'outil clé : `ImageFolder`

`torchvision.datasets.ImageFolder` remplace le `datasets.MNIST` du TP 4. Il
déduit les **classes du nom des sous-dossiers** :

```
dataset/
├── train/
│   ├── pommes/   ← classe "pommes"
│   └── poires/   ← classe "poires"
└── test/
    ├── pommes/
    └── poires/
```

Ranger une image dans le bon dossier, **c'est l'étiqueter**. C'est tout le
principe : le dataset, c'est l'organisation de tes fichiers.

## Étape 1 — Récupérer des images

Le script `telecharger_dataset.py` puise dans **plusieurs sources** (pour ne pas
dépendre d'un seul site), convertit les images en `.jpg` et les répartit en
`train/` (80 %) et `test/` (20 %) :

| Source | Clé nécessaire ? | Comment obtenir la clé |
| --- | --- | --- |
| **Bing Images** (via `icrawler`) | Non — source principale | — |
| **Wikimedia Commons** | Non — source d'appoint | — |
| **Pexels** | Oui (facultatif) | compte gratuit sur pexels.com/api |
| **Pixabay** | Oui (facultatif) | compte gratuit sur pixabay.com/api/docs |
| **Unsplash** | Oui (facultatif) | compte gratuit sur unsplash.com/developers |

```powershell
uv run telecharger_dataset.py
```

**Bing + Wikimedia suffisent** (aucune clé requise). Pour activer les trois
autres sources, déclare tes clés en **variables d'environnement** avant de
lancer le script — on ne met **jamais** une clé en dur dans le code :

```powershell
$env:PEXELS_API_KEY   = "ta_cle"
$env:PIXABAY_API_KEY  = "ta_cle"
$env:UNSPLASH_ACCESS_KEY = "ta_cle"
uv run telecharger_dataset.py
```

Une source sans clé est simplement **sautée** (message dans les logs).

### Variante : le dataset figé de JLG (reproductible)

Si tu veux **exactement les mêmes images que tout le monde** (utile pour comparer
des réglages sans que le dataset change), télécharge le dataset déjà constitué et
déjà trié :

```powershell
uv run telecharger_dataset_jlg.py
```

Il récupère un `.zip` en ligne et l'installe dans `dataset/` (64 + 64 en train,
16 + 16 en test). Combiné au `torch.manual_seed(0)` de `main.py`, l'entraînement
devient alors **parfaitement reproductible**. Ce script **remplace** le dossier
`dataset/` existant : sauvegarde ton tri manuel avant de le lancer.

## Étape 2 — TRIER (le cœur du TP)

Une recherche automatique renvoie des images **bruitées** : dessins, paniers,
compotes, mauvaises espèces… **Ouvre les dossiers `dataset/` et supprime à la
main tout ce qui ne montre pas clairement le bon fruit.** C'est exactement le
travail d'un praticien : la **qualité des données** se gagne ici, pas dans le
code.

> Astuce : garde des images **variées** (fonds, angles, éclairages) pour que le
> modèle apprenne le fruit et pas le décor. Et vérifie que **test/** contient
> des images **différentes** de **train/** (sinon l'accuracy ment).

## Étape 3 — Entraîner

```powershell
uv run main.py
```

Ou, depuis n'importe où dans le projet :

```powershell
mise run tp5-data   # télécharger (étape 1)
mise run tp5        # entraîner (étape 3)
```

## Pourquoi un modèle **pré-entraîné** ?

Avec seulement quelques dizaines d'images, un réseau entraîné **de zéro**
échouerait — et on ne saurait pas si c'est la faute du **modèle** ou des
**données**. Pour que **le dataset soit le seul facteur qui compte**, on emprunte
**ResNet18**, un réseau déjà entraîné sur des millions d'images (ImageNet) :

- on **gèle** tout son corps (ses poids ne bougent plus) ;
- on remplace seulement sa **dernière couche** par une sortie à **2 classes**,
  la seule qu'on entraîne.

Le corps gelé sert d'**extracteur de caractéristiques** : il transforme une image
en un vecteur qui résume ce qu'elle contient. On entraîne juste une petite
« régression logistique » par-dessus. C'est ce qu'on appelle le **transfer
learning** (apprentissage par transfert).

> **On avance un peu sur le programme :** l'idée d'emprunter un modèle
> pré-entraîné reviendra en détail plus tard (fine-tuning). Ici on l'utilise dans
> sa forme la plus simple, uniquement pour **neutraliser le modèle** et se
> concentrer sur les données. On démontera l'intérieur d'un CNN au **TP 6**.

## Les pièges du monde réel (et comment le code les règle)

1. **Tailles variables** → `transforms.Resize((224, 224))` : impossible d'empiler
   des images de tailles différentes dans un batch sans les uniformiser.
2. **Couleur + normalisation** → `ToTensor` puis `Normalize(...)` avec les
   **statistiques d'ImageNet** (celles avec lesquelles ResNet a appris).
3. **Formats variés** → le script sort du `.jpg` propre. Le plugin
   `pillow-avif-plugin` (importé dans `main.py`) permet en plus de lire tes
   éventuelles images `.avif` personnelles.
4. **Peu d'images** → surapprentissage garanti : le modèle peut mémoriser le
   train et échouer en test. C'est **la** démonstration à observer.

## Ce que tu dois observer

- Même avec **peu d'images**, l'accuracy test est **bonne** (grâce au modèle
  pré-entraîné) : la preuve que le transfer learning fait des miracles.
- Si tu **retires des images** ou **déséquilibres** les classes, l'écart
  train/test se creuse : c'est l'effet direct de la **quantité** et de
  l'**équilibre**.
- La **matrice de confusion 2×2** montre quelles images se font confondre
  (souvent celles mal triées à l'étape 2).

## À expérimenter

- **Quantité** : entraîne avec 10, puis 30, puis 60 images par classe. Où
  l'accuracy décolle-t-elle ?
- **Qualité** : laisse volontairement quelques images bruitées → observe la
  chute.
- **Déséquilibre** : mets 50 pommes pour 10 poires → l'accuracy globale
  devient trompeuse.
- **Prétraitement** : change `TAILLE_IMAGE`, ou ajoute de la *data augmentation*
  (`transforms.RandomHorizontalFlip()`), et compare.

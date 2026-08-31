# TP 6 — CNN sur MNIST, export ONNX et reconnaissance dans le navigateur

## Contexte

Sixième TP du parcours pédagogique `ai-training`. Après le TP 4 qui traitait
MNIST avec un réseau **entièrement connecté** (MLP) sous PyTorch, ce TP introduit
**une seule notion nouvelle : la convolution / les réseaux convolutifs (CNN)**.

L'export ONNX et l'application web de dessin sont un **bonus applicatif
secondaire** : ils rendent le TP concret et motivant (voir son modèle
reconnaître un chiffre qu'on dessine à la souris), mais ne sont pas le cœur
pédagogique. La convolution reste la notion centrale à enseigner.

## Objectif

1. **Entraîner un CNN** sur MNIST (chiffres manuscrits 0-9, images 28×28 en
   niveaux de gris) avec PyTorch, en visant une **précision élevée** (> 99 % sur
   le jeu de test).
2. **Exporter le modèle entraîné au format ONNX**.
3. **Développer un front-end-only** (aucun serveur backend) qui charge le modèle
   ONNX dans le navigateur, laisse l'utilisateur **dessiner un chiffre à la
   souris** sur un canvas, puis affiche la **prédiction** accompagnée des
   **probabilités des 10 classes**.

## Périmètre fonctionnel

### Partie entraînement (Python / PyTorch)

- Point d'entrée `main.py`, lançable via `uv run main.py` (comme les autres TP).
- **Chargement des données** : `torchvision.datasets.MNIST` avec
  `download=True`, dans un dossier local `06-cnn/data/` (re-téléchargement propre
  au TP6, on ne réutilise pas les données du TP4).
- **Architecture CNN « plus profonde »** pour viser > 99 % : empilement de
  couches de convolution + pooling, avec **BatchNorm** et **Dropout** pour la
  régularisation, suivi de couches denses. Le code reste explicite et commenté
  (bandeaux de section en français), fidèle au style pédagogique du parcours.
- **Matériel** : utiliser le **GPU (CUDA) s'il est disponible**, sinon repli
  automatique sur CPU (`device = "cuda" if torch.cuda.is_available() else
  "cpu"`).
- **Boucle d'entraînement** classique : données → modèle → perte
  (cross-entropy) → descente de gradient → évaluation sur le test.
- **Export ONNX** du modèle entraîné (`torch.onnx.export`), avec une entrée de
  forme `[1, 1, 28, 28]`. Le fichier `.onnx` produit doit être copié dans
  `front/public/` pour être servi au navigateur.
- Sauvegarde de la figure d'entraînement (courbes de perte/précision) en
  `resultat.png` via `plt.savefig` (pas de `plt.show`).

### Partie front-end (Vite + TypeScript)

- **Site construit avec Vite + TypeScript**, dépendances gérées par **Bun**
  (runtime, gestionnaire de paquets et lanceur de scripts).
- Inférence via **`onnxruntime-web`** installé en dépendance Bun (pas de CDN).
- **Canvas de dessin** (~280×280) où l'utilisateur trace un chiffre à la souris.
- **Prétraitement fidèle à MNIST** avant l'inférence :
  - conversion en niveaux de gris, **chiffre blanc sur fond noir** ;
  - recadrage sur la boîte englobante du tracé ;
  - **centrage par centre de masse** ;
  - réduction (downscale) vers **28×28** ;
  - normalisation identique à celle de l'entraînement (mêmes moyenne/écart-type).
- **Affichage du résultat** : le **chiffre prédit** + les **10 probabilités**
  (par ex. barres de probabilité pour chaque classe 0-9).
- Bouton pour **effacer** le canvas et recommencer.

## Hors périmètre

- Pas de backend, pas d'API serveur : l'inférence se fait **entièrement dans le
  navigateur**.
- Pas de `MATH.md` ni de dérivation mathématique de la convolution pour ce TP
  (documentation limitée au `README.md`).
- Pas de réutilisation des données MNIST du TP4.
- Pas de déploiement en production : exécution **en local** uniquement.

## Utilisateurs cibles

Étudiant·e·s du parcours `ai-training` (public interne, pédagogique). Le TP doit
rester lisible et reproductible, la clarté primant sur la performance ou
l'élégance.

## Contraintes techniques

- **Entraînement** : Python 3.12 + PyTorch + torchvision, exécuté via `uv` /
  `mise` avec le `.venv` partagé à la racine. Ajouter les dépendances avec
  `uv add <paquet>` **à la racine** (ne jamais créer de `pyproject.toml` ou de
  `.venv` dans le dossier du TP). GPU CUDA exploité si présent.
- **Front** : **Bun** + **Vite + TypeScript**, dépendance `onnxruntime-web`.
  Le `.onnx` est servi depuis `front/public/`.
- **Compatibilité** : commandes fonctionnant sous PowerShell **et** cmd.exe.
  Éviter les caractères non-cp1252 dans les `print()` Python (risque
  d'`UnicodeEncodeError` en sortie redirigée).

## Organisation du code

```
06-cnn/
  main.py            # entraînement PyTorch + export ONNX
  README.md          # objectif, fonctionnement, à observer, à expérimenter
  data/              # MNIST téléchargé par torchvision (généré)
  resultat.png       # courbes d'entraînement (généré)
  front/             # application web Vite + TypeScript
    public/
      modele.onnx    # modèle exporté (copié depuis l'entraînement)
    src/
    package.json
    bun.lock
    ...
```

## Exécution

- **Entraînement** : depuis `06-cnn/`, `uv run main.py` (télécharge MNIST,
  entraîne, exporte `modele.onnx`, copie dans `front/public/`). Ajouter une tâche
  `tp6` dans `mise.toml` sur le modèle des autres TP.
- **Front** : depuis `06-cnn/front/`, `bun install` puis **`bun run dev`**
  (serveur de développement Vite en local). Le navigateur charge le modèle ONNX
  depuis `public/` et fait l'inférence côté client.

## Critères de succès

- Le CNN atteint **> 99 %** de précision sur le jeu de test MNIST.
- Le modèle est exporté en `.onnx` valide et chargeable par `onnxruntime-web`.
- Le site Vite se lance en local (`bun run dev`) sans backend.
- Un chiffre dessiné à la souris est correctement **prétraité (façon MNIST)** et
  **reconnu**, avec affichage du chiffre prédit et des probabilités des 10
  classes.
- Le code d'entraînement reste **clair et commenté** en français, cohérent avec
  le style des TP précédents.
- Un `README.md` documente l'objectif, le fonctionnement, ce qu'il faut observer
  et ce qu'on peut expérimenter.

## Points en suspens

- **Détail exact de l'architecture** (nombre de couches conv, tailles de noyaux,
  canaux, position des BatchNorm/Dropout, optimiseur, learning rate, nombre
  d'époques) : à figer au moment de l'implémentation, en visant > 99 % tout en
  restant pédagogiquement lisible.
- **Paramètres de normalisation** : réutiliser les valeurs standard MNIST
  (moyenne ≈ 0.1307, écart-type ≈ 0.3081) ou les recalculer — à trancher, en
  gardant la **même normalisation** entre entraînement et front.
- **Ergonomie du canvas** (épaisseur du trait, lissage) à ajuster pour que le
  prétraitement produise des entrées proches des exemples MNIST.

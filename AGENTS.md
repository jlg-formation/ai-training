# AGENTS.md — AI Training

Parcours pédagogique de **TP progressifs** pour comprendre l'entraînement d'une
IA depuis zéro (1 neurone → modèles avancés). Voir [README.md](README.md) pour
l'installation et [SUJETS_TPS.md](SUJETS_TPS.md) pour les énoncés et la théorie.

## Contexte : ce projet est un support de cours

- **Langue : français.** Code, commentaires, noms de variables (`perte`,
  `predire`, `bruit`), docstrings, README et messages : tout est en français.
- **But pédagogique avant tout.** La clarté prime sur la performance ou
  l'élégance. Chaque TP introduit **une seule notion nouvelle** (cf. le tableau
  de [SUJETS_TPS.md](SUJETS_TPS.md)).
- **Ne pas sur-optimiser.** Éviter les abstractions, la vectorisation obscure ou
  les patterns avancés qui masquent la notion enseignée. Préférer un code long
  et explicite à un code court et malin.
- **Écrire les choix dans les fichiers, pas seulement dans le chat.** Dès qu'une
  décision de conception ou de notation est prise (ex. « le vocabulaire ne se
  construit que sur le train », « on note $V$ la taille du vocabulaire »),
  l'inscrire **immédiatement** dans le `README.md` ou le `MATH.md` concerné. Une
  explication donnée seulement en réponse ne suffit pas : le fichier doit être
  auto-suffisant pour l'étudiant.

## Environnement & commandes (Windows / PowerShell)

- Outils gérés par **mise** ([mise.toml](mise.toml)) : Python 3.12 + **uv**.
- **Un seul `.venv` partagé** à la racine ; aucune install par TP. Dépendances
  communes dans [pyproject.toml](pyproject.toml) (`numpy`, `matplotlib`).

```powershell
mise run install   # = uv sync : crée .venv/ et installe les dépendances
mise run tp1       # lance le TP 1 (tâche mise, depuis n'importe où)
mise tasks         # liste les tâches disponibles
```

Depuis un dossier de TP : `uv run main.py` (uv remonte au `pyproject.toml`
racine et réutilise le `.venv` partagé).

- **Ajouter une dépendance** : `uv add <paquet>` **à la racine** uniquement.
- **Ne jamais** créer un `pyproject.toml`, un `.venv` ou un `requirements.txt`
  dans un dossier de TP.
- **Compatibilité cmd.exe.** Tout doit fonctionner aussi bien depuis l'invite de
  commandes Windows (`cmd`) que depuis PowerShell. `mise activate` n'y est pas
  supporté, mais `mise run tpN` et `mise exec -- uv run main.py` marchent tels
  quels. Éviter la syntaxe propre à un seul shell (pipes PowerShell, `$env:…`,
  cmdlets) dans les commandes, tâches mise et README destinés aux étudiants.

## Structure d'un TP

Un dossier par TP nommé `NN-slug` (ex. `01-regression-lineaire`) contenant
**seulement du code** :

- `main.py` — point d'entrée, lançable par `uv run main.py`.
- `README.md` — objectif, fonctionnement, « ce que tu dois observer », « à
  expérimenter ».
- `MATH.md` (optionnel) — dérivation mathématique en KaTeX (`$...$`, `$$...$$`).
  Suivre les **conventions mathématiques** du projet (vecteurs **lignes**,
  notation gras/casse, produit $\odot$…) décrites dans
  [README.md § Conventions mathématiques](README.md#conventions-mathématiques).
  Les formules doivent correspondre **ligne pour ligne** au code NumPy du TP.

Voir [01-regression-lineaire](01-regression-lineaire/) comme modèle de référence.

## Discipline de notation dans les `MATH.md`

Ces règles évitent les allers-retours les plus fréquents sur les dérivations :

- **Définir chaque symbole dès sa première apparition.** N'introduire ni
  matrice, ni vecteur, ni scalaire ($\mathbf{X}$, $\mathbf{p}$, $z$…) sans dire
  aussitôt ce qu'il vaut et sa taille.
- **Un symbole = une seule signification.** Ne jamais réutiliser une lettre pour
  deux choses (ex. $N$ = nombre d'exemples **et** $V$ = taille du vocabulaire,
  jamais le même symbole pour les deux).
- **Notation des indices cohérente des deux côtés d'une égalité.** Si un membre
  porte l'indice $i$ (ex. $z_i$, $p_i$, $y_i$, $L_i$), l'autre aussi.
- **Rappeler le lien exemple ↔ agrégat.** Toujours expliciter comment une
  quantité par exemple se relie à sa moyenne sur le batch (ex.
  $L = \tfrac{1}{N}\sum_i L_i$).
- **Distinguer vecteurs lignes / colonnes.** Préciser la forme ($1 \times d$,
  $d \times 1$) et rappeler qu'un $^\top$ bascule de l'un à l'autre.

## Conventions Mermaid (diagrammes de topologie)

Quand un `README.md` illustre le réseau avec un diagramme Mermaid :

- **Matrices en gras MAJUSCULE**, vecteurs en gras minuscule (même convention
  que les `MATH.md`).
- **Indices compatibles Mermaid** : la syntaxe `x_V` ne rend pas ; utiliser une
  écriture qui s'affiche correctement dans les nœuds.
- **Entrées omises** : marquer la coupure par `...` entre les premières entrées
  et la dernière (ex. entre $x_3$ et $x_V$).
- **Sigmoïde dans le neurone** : la fonction d'activation fait partie du neurone,
  la représenter à l'intérieur du nœud (pas comme une étape séparée).

## Conventions de code (suivre le style de TP 1)

- **NumPy uniquement** pour les premiers TP ; introduire un framework (PyTorch)
  seulement quand le sujet l'exige (cf. TP 4+ dans [SUJETS_TPS.md](SUJETS_TPS.md)).
- Commentaires abondants en français avec des **bandeaux de section**
  (`# ---- 1) Génération des données ----`) qui suivent les étapes du TP.
- Constantes en MAJUSCULES (`LEARNING_RATE`, `EPOCHS`, `A_VRAI`).
- `np.random.seed(0)` pour des résultats reproductibles.
- Structure typique : génération des données → modèle → perte → boucle
  d'entraînement (descente de gradient) → résultat → visualisation matplotlib.
- Les figures sont **enregistrées** (`plt.savefig("resultat.png")`), pas
  affichées en interactif (`plt.show()`), pour rester exécutable sans écran.

## Créer un nouveau TP

1. Créer `NN-slug/` avec `main.py` et `README.md` (calqués sur le TP 1).
2. Ajouter une tâche dans [mise.toml](mise.toml) :

   ```toml
   [tasks.tpN]
   description = "TP N - <titre>"
   dir = "NN-slug"
   run = "uv run main.py"
   ```

3. Ajouter une ligne au tableau « TP disponibles » de [README.md](README.md).

## Ajouter un site web à un TP (front ONNX)

Motif récurrent (TP 6, 7…) : un modèle entraîné en PyTorch est exporté en **ONNX**
puis réutilisé dans un petit site `NN-slug/front/` (Bun + Vite + TypeScript +
`onnxruntime-web`).

- Les dépendances Node sont **factorisées à la racine** via les **workspaces
  Bun** : le [package.json](package.json) racine déclare `"workspaces":
  ["*/front"]`. Le glob capte **automatiquement** tout nouveau `NN-slug/front` —
  **ne rien changer à la racine**.
- Chaque `front/` garde **son propre** `package.json` (avec ses
  `devDependencies` : `vite`, `typescript`, `vite-plugin-static-copy`) et sa
  config (`vite.config.ts`, `tsconfig.json`). Cette duplication de config est
  **assumée** ; ne pas déplacer l'outillage à la racine.
- **Linker Bun** : garder le **défaut (isolé)**. Ne **pas** forcer
  `linker = "hoisted"` (bunfig) : sur Windows il casse la résolution des binaires
  natifs `esbuild`/`rollup` en workspace (`could not find bin metadata file`),
  non réparé par `bun install --force`. En mode isolé, les paquets sont stockés
  une seule fois dans `node_modules/.bun/` et chaque `front/node_modules/<pkg>`
  est un lien symbolique.
- **Copie des `.wasm` d'ONNX** dans `vite.config.ts` : résoudre le dossier `dist`
  via Node, pas par un chemin relatif fixe (le paquet n'est plus dans
  `front/node_modules`). `onnxruntime-web` n'expose **pas** `./package.json` dans
  ses `exports` ; passer par un sous-chemin `.wasm` exporté :
  `path.dirname(require.resolve("onnxruntime-web/ort-wasm-simd-threaded.wasm"))`.
- Ajouter une tâche `tpN-site` dans [mise.toml](mise.toml) (`dir = "NN-slug/front"`,
  `run = ["bun install", "bun run dev"]`) et documenter le site dans le
  `README.md` du TP.

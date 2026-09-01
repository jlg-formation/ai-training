# AI Training — Apprendre à entraîner des IA depuis zéro

Un parcours d'exercices (TP) progressifs pour **comprendre** l'entraînement
d'une IA, en partant d'un seul neurone jusqu'à des modèles plus avancés.
Chaque TP introduit **une seule notion nouvelle**.

> Le détail pédagogique de chaque TP se trouve dans [SUJETS_TPS.md](SUJETS_TPS.md).

## Prérequis

- [**mise**](https://mise.jdx.dev) (gestionnaire d'outils et de tâches). Il
  installe automatiquement Python et uv pour ce projet. Vérifier :

  ```powershell
  mise --version
  ```

  Installation si besoin (Windows PowerShell) :

  ```powershell
  winget install jdx.mise
  ```

mise lit le fichier [`mise.toml`](mise.toml) et fournit **Python 3.12** et
**[uv](https://docs.astral.sh/uv/)** : rien d'autre à installer à la main.

## Installation

La première fois, autoriser la config puis installer les outils et les
dépendances (un **seul environnement partagé** pour tous les TP) :

```powershell
# à la racine du dépôt
mise trust
mise install     # installe Python + uv (déclarés dans mise.toml)
mise run install # = uv sync : crée .venv/ et installe numpy, matplotlib, ...
```

Les dépendances communes sont déclarées dans `pyproject.toml` et installées
une seule fois dans `.venv/`.

## Organisation du projet

```
ai-training/
├── mise.toml               # outils (python, uv) + tâches (mise run tpN)
├── pyproject.toml          # dépendances partagées par tous les TP
├── uv.lock                 # versions verrouillées
├── .venv/                  # UN SEUL environnement Python (non versionné)
├── SUJETS_TPS.md           # énoncés et théorie des TP
├── README.md               # ce fichier
└── NN-slug/                # un dossier par TP (ex. 01-regression-lineaire)
    ├── main.py             # code du TP
    └── README.md           # explications propres au TP
```

Chaque TP vit dans un dossier `NN-slug` et ne contient **que du code** : il
réutilise l'environnement de la racine, rien n'est réinstallé localement.

## Lancer un TP

Le plus simple, via une tâche mise (depuis n'importe où dans le projet) :

```powershell
mise run tp1
```

Lister les tâches disponibles :

```powershell
mise tasks
```

Ou directement avec uv, depuis le dossier du TP :

```powershell
cd 01-regression-lineaire
uv run main.py
```

`uv` remonte automatiquement jusqu'au `pyproject.toml` de la racine et utilise
le `.venv` partagé.

> **cmd.exe (invite de commandes)** : `mise activate` n'est pas supporté, mais
> `mise run tp1` et `mise exec -- uv run main.py` fonctionnent tels quels. Pour
> utiliser `python`/`uv` directement, ajoute une fois les _shims_ au PATH :
> `setx PATH "%PATH%;%LOCALAPPDATA%\mise\shims"` (rouvre le cmd ensuite).

### TP disponibles

| Dossier                                           | TP                              | Tâche mise |
| ------------------------------------------------- | ------------------------------- | ---------- |
| [01-regression-lineaire](01-regression-lineaire/) | Régression linéaire (1 neurone) | `tp1`      |
| [02-spam-ou-pas-spam](02-spam-ou-pas-spam/)       | Spam ou pas spam (classification) | `tp2`    |
| [03-xor-couche-cachee](03-xor-couche-cachee/)     | XOR et couche cachée (MLP)      | `tp3`      |
| [04-mnist-pytorch](04-mnist-pytorch/)             | Chiffres MNIST (première fois en PyTorch) | `tp4` |
| [05-my-own-dataset](05-my-own-dataset/)           | Ton propre dataset (pommes vs poires, transfer learning) | `tp5` |
| [06-cnn](06-cnn/)                                 | CNN sur MNIST (convolution) + reconnaissance ONNX dans le navigateur | `tp6` |
| [07-gan](07-gan/)                                 | GAN sur MNIST (générer des chiffres, entraînement adversaire) + génération ONNX dans le navigateur | `tp7` |

## Sites web des TP (front)

Certains TP (6, 7…) réutilisent le modèle entraîné dans un **petit site web**
(`NN-slug/front/`) qui charge le modèle **ONNX** et l'exécute dans le navigateur
(Bun + Vite + TypeScript + `onnxruntime-web`).

Comme pour Python, les dépendances Node sont **factorisées à l'échelle du dépôt**
via les [workspaces Bun](https://bun.sh/docs/install/workspaces) : le
[`package.json`](package.json) racine déclare `"workspaces": ["*/front"]`, si
bien qu'**un seul `node_modules` existe à la racine** (dans `node_modules/.bun/`)
au lieu d'un par TP. Chaque `front/` garde un dossier `node_modules/` **léger**
qui ne contient que des **liens symboliques** vers ce magasin partagé (aucune
copie des fichiers ; non versionné).

Installer une fois, depuis la racine :

```powershell
bun install
```

Puis lancer le site d'un TP (installe au besoin, puis démarre Vite) :

```powershell
mise run tp6-site   # ou tp7-site
```

> Le glob `*/front` capte **automatiquement** tout futur TP ayant un dossier
> `front/` : rien à changer à la racine. On garde le **linker isolé** de Bun (par
> défaut) ; le linker `hoisted` est à éviter sur Windows (il casse la résolution
> des binaires natifs `esbuild`/`rollup` en workspace).

## Conventions mathématiques

Les fichiers `MATH.md` des TP dérivent les gradients à la main. Comme il existe
**plusieurs notations valides** en algèbre linéaire, ce chapitre fixe les choix
faits dans tout le projet pour éviter les surprises — notamment si tu compares
avec un livre de référence.

### Vecteurs lignes vs vecteurs colonnes

C'est **le** point qui déroute le plus souvent. Deux conventions coexistent.
Pour un **seul exemple** $\mathbf{x}$ (à $d$ entrées) et une couche à $h$
neurones :

- **Convention « colonne »** (littérature académique : Goodfellow, Bishop,
  Murphy…). Un vecteur $\mathbf{x}$ est une **colonne** ($d \times 1$) et la
  couche s'écrit :

  $$\mathbf{z} = \mathbf{W}^\top \mathbf{x} + \mathbf{b}$$

  C'est la notation mathématique classique, où un vecteur est une colonne par
  défaut et où l'on transpose la matrice de poids.

- **Convention « ligne »** (code : NumPy, PyTorch, TensorFlow). Le vecteur
  $\mathbf{x}$ est une **ligne** ($1 \times d$) et la couche s'écrit :

  $$\mathbf{z} = \mathbf{x}\,\mathbf{W} + \mathbf{b}$$

Les deux décrivent **la même couche** : elles sont simplement transposées l'une
de l'autre.

> **Choix du projet : la convention « ligne ».** Tous les vecteurs
> ($\mathbf{x}$, $\mathbf{z}$, $\mathbf{b}$, $\boldsymbol{\delta}$…) sont des
> **vecteurs lignes**.

**Pourquoi ce choix ?** Pour que les `MATH.md` correspondent **ligne pour
ligne** au code NumPy, sans transposition mentale :

| Maths (livres, colonnes)                                | Maths (ce projet, lignes)                            | Code NumPy          |
| ------------------------------------------------------- | ---------------------------------------------------- | ------------------- |
| $\mathbf{z} = \mathbf{W}^\top\mathbf{x} + \mathbf{b}$   | $\mathbf{z} = \mathbf{x}\,\mathbf{W} + \mathbf{b}$   | `z = x @ W + b`     |
| $\nabla_{\mathbf{W}} = \boldsymbol{\delta}\,\mathbf{x}^\top$ | $\nabla_{\mathbf{W}} = \mathbf{x}^\top\boldsymbol{\delta}$ | `grad_W = X.T @ dz` |

Concrètement, la convention ligne évite d'avoir des `.T` partout dans le code,
ce qui est plus lisible pour débuter. Si tu lis un article de recherche, garde
juste en tête qu'il utilise très probablement la convention colonne : passe de
l'une à l'autre en **transposant**.

### Et pour un batch ? (et un piège de dimensions)

Jusqu'ici on a raisonné sur **un seul** exemple. En pratique on traite les $N$
exemples **d'un coup** : on les empile en **lignes** dans une matrice
$\mathbf{X}$ ($N \times d$), ce qui donne

$$\mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{b}$$

où chaque **ligne** de $\mathbf{Z}$ est le $\mathbf{z}$ d'un exemple.

**Mais attention, un détail cloche.** Si tu regardes les tailles, $\mathbf{X}\mathbf{W}$
est une matrice $N \times h$ ($N$ exemples, $h$ neurones) alors que $\mathbf{b}$
n'est qu'une **ligne** $1 \times h$. En algèbre linéaire stricte, **on ne peut
pas additionner deux tableaux de tailles différentes** ! L'écriture
$+\,\mathbf{b}$ est donc un **abus de notation** : tout le monde comprend qu'on
ajoute le **même biais $\mathbf{b}$ à chaque ligne**, et le code (NumPy) le fait
tout seul (c'est le *broadcasting*).

Pour écrire la **même chose proprement**, on multiplie $\mathbf{b}$ à gauche par
un vecteur colonne de **uns** $\mathbf{1}_N$ (taille $N \times 1$) :

$$\mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{1}_N\,\mathbf{b}
\qquad\text{avec}\qquad
\underbrace{\mathbf{1}_N}_{N \times 1}\underbrace{\mathbf{b}}_{1 \times h}
= \underbrace{\mathbf{1}_N\mathbf{b}}_{N \times h}$$

Le produit $\mathbf{1}_N\mathbf{b}$ est simplement la matrice $N \times h$ dont
**chaque ligne est $\mathbf{b}$** (le biais recopié $N$ fois). Cette fois les
deux termes sont bien en $N \times h$ et l'addition est **parfaitement
légale** — c'est exactement ce que le broadcasting fabrique à ta place.

> **À retenir :** $\mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{b}$ est la forme
> courte et pratique ; $\mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{1}_N\mathbf{b}$
> en est la version rigoureuse. Les deux disent la même chose : ajouter le même
> biais à chaque exemple.

### Notation (gras, casse)

Convention typographique reprise dans chaque `MATH.md` :

- **Vecteurs** : gras minuscule ($\mathbf{x}$, $\mathbf{b}$, $\mathbf{z}$,
  $\boldsymbol{\delta}$).
- **Matrices** : gras MAJUSCULE ($\mathbf{X}$, $\mathbf{W}$).
- **Scalaires** : maigre ($b^{(2)}$, $z^{(2)}$, $p$, $y$).
- **Exposants entre parenthèses** : numéro de **couche** ($\mathbf{W}^{(1)}$ =
  couche cachée, $\mathbf{W}^{(2)}$ = couche de sortie), à ne pas confondre avec
  une puissance.

### Autres conventions

- **Produit terme à terme** noté $\odot$ (produit de Hadamard), distinct du
  produit matriciel.
- **Moyenne sur le batch** : les gradients sont moyennés sur les $N$ exemples
  (division par $N$), ce qui rend le pas d'apprentissage indépendant de la
  taille du batch.
- **KaTeX** : les formules sont écrites en `$...$` (en ligne) et `$$...$$`
  (bloc), rendues par la plupart des lecteurs Markdown.

## Ajouter une dépendance

Si un futur TP a besoin d'une nouvelle bibliothèque, on l'ajoute **une seule
fois** à la racine, et elle devient disponible pour tous les TP :

```powershell
uv add nom-du-paquet
```

## Créer un nouveau TP

1. Créer un dossier `NN-slug` (ex. `02-spam-ou-pas-spam`).
2. Y ajouter un `main.py` (et un `README.md` d'explications).
3. Ajouter une tâche dans [`mise.toml`](mise.toml) :

   ```toml
   [tasks.tp2]
   description = "TP 2 - Spam ou pas spam"
   dir = "02-spam-ou-pas-spam"
   run = "uv run main.py"
   ```

4. Lancer avec `mise run tp2` (ou `uv run main.py` depuis le dossier).

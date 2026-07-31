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

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

Voir [01-regression-lineaire](01-regression-lineaire/) comme modèle de référence.

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

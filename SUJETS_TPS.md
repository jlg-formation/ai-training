# AI Training

## Prompts

Pour quelqu'un qui a jamais fait d'IA et d'entraînement d'IA, je voudrais faire un tuto rapide, comment, enfin faire des exercices très progressifs et qui partent de zéro pour entraîner des IA. Voilà, sur des architectures toutes simples au début, et puis progressivement plus compliquées pour bien mettre en valeur, on va dire, des aspects qu'il faut montrer quand on entraîne des IA. Donc, entraîner des IA sur son PC, et pourquoi pas après entraîner des IA sur des systèmes de cloud si ça devient plus lourd.

## Introduction

C'est une excellente idée de progression. Si le but est de **comprendre** l'entraînement d'une IA et pas seulement d'utiliser des modèles existants, je partirais d'un parcours où chaque exercice introduit une seule notion nouvelle. Les étudiants voient ainsi immédiatement l'effet de ce qu'ils changent.

Voici un parcours en 11 étapes.

| Étape | Projet                                    | Ce qu'on apprend                     |
| ----- | ----------------------------------------- | ------------------------------------ |
| 1     | Une régression linéaire (1 neurone)       | données, perte, descente de gradient |
| 2     | Classification binaire (spam ou non spam) | sigmoid, accuracy, overfitting       |
| 3     | Réseau à une couche cachée                | intérêt des couches cachées          |
| 4     | Classification d'images MNIST             | CNN, GPU, epochs                     |
| 5     | Détection d'objets simple                 | annotations, bounding boxes          |
| 6     | Classification d'images personnelles      | création de dataset                  |
| 7     | GAN (générer des images)                  | génératif, entraînement adversaire   |
| 8     | Fine-tuning d'un petit LLM                | tokenizer, LoRA, VRAM                |
| 9     | Fine-tuning multimodal                    | texte + image                        |
| 10    | Entraînement distribué                    | plusieurs GPU                        |
| 11    | Cloud                                     | Runpod, Vast.ai, Lambda Labs, etc.   |

## TP 1 — Le neurone unique

Un seul neurone.

Entrée :

```
x
```

Sortie :

```
y = ax+b
```

Objectif :

Apprendre à retrouver

```
y = 2x + 3
```

à partir de données générées.

L'étudiant voit :

- la loss diminuer
- les poids évoluer
- le principe de la descente de gradient

Aucune IA moderne n'est nécessaire.

Simple NumPy.

---

## TP 2 — Spam ou pas spam

Quelques centaines de phrases.

```
"Bonjour"

"Promotion"

"Achetez"

```

Objectif :

Classer :

```
Spam

Pas spam
```

Notions :

- classification
- sigmoid
- fonction de perte
- train/test

Puis montrer le surapprentissage.

---

## TP 3 — Pourquoi plusieurs neurones ?

On montre que certains problèmes ne sont pas séparables.

Exemple classique :

XOR

Puis on ajoute :

```
Entrée

↓

Couche cachée

↓

Sortie
```

L'étudiant voit immédiatement pourquoi plusieurs neurones sont utiles.

---

## TP 4 — Reconnaissance de chiffres

Le célèbre MNIST.

```
0
1
2
...
9
```

Cette fois on introduit PyTorch.

L'étudiant découvre :

- Dataset
- DataLoader
- GPU
- Epochs
- Batch size

---

## TP 5 — Images personnelles

Au lieu de MNIST :

Deux dossiers

```
Chats/

Chiens/
```

ou

```
Pommes

Poires
```

L'étudiant construit son premier dataset.

Il comprend :

- qualité des données
- quantité
- déséquilibre

---

## TP 6 — Petit CNN

Architecture très simple.

On compare :

```
MLP

vs

CNN
```

Les performances sautent immédiatement.

On comprend pourquoi les CNN existent.

---

## TP 7 — Générer des images (GAN)

Jusqu'ici, tous les réseaux étaient **discriminatifs** : on montre une donnée, ils
rendent une **étiquette**. On bascule dans le **génératif** : au lieu de
**reconnaître** un chiffre, on veut en **fabriquer** de nouveaux.

La nouveauté : l'**entraînement adversaire** (Goodfellow 2014). Deux réseaux
s'affrontent :

```
Générateur      →  fabrique de fausses images (le faussaire)

Discriminateur  →  vrai ou faux ? (le policier)
```

On les entraîne **ensemble** : le discriminateur apprend à démasquer les faux, ce
qui pousse le générateur à faire des faux plus crédibles.

Notions :

- modèle génératif vs discriminatif
- deux réseaux, deux pertes, deux optimiseurs
- l'équilibre (les pertes n'atteignent pas zéro)
- mode collapse

---

## TP 8 — Détection d'objets

Une seule classe.

Par exemple :

```
Repérer les voitures.
```

Introduction des annotations.

```
image

+

rectangle
```

Comprendre :

- bounding boxes
- précision
- rappel

---

## TP 9 — Premier LLM

Petit modèle.

Par exemple :

- SmolLM
- TinyLlama
- Qwen 0.5B

Pas d'entraînement complet.

Seulement un fine-tuning.

Les étudiants découvrent :

- tokenizer
- embeddings
- LoRA
- VRAM

---

## TP 10 — Entraîner sur son PC

Comparer plusieurs cartes.

```
CPU

GPU NVIDIA

GPU AMD

Apple Silicon
```

Mesurer :

- temps
- mémoire
- consommation

Montrer les outils :

- PyTorch
- CUDA
- ROCm
- MPS

---

## TP 11 — Passer au cloud

Même notebook.

Mais cette fois :

- Google Colab
- Runpod
- Vast.ai
- Lambda Labs

Comparer :

- coût
- vitesse
- disponibilité

Puis montrer comment reprendre exactement le même code qu'en local.

---

# Ce que j'ajouterais entre chaque TP

À chaque étape, faire mesurer aux étudiants ce qui change réellement :

- Temps d'entraînement
- Taille du dataset
- Taille du modèle
- Nombre de paramètres
- Occupation mémoire
- Utilisation GPU
- Accuracy
- Loss
- Temps par epoch

Ils comprendront rapidement quels paramètres influencent les performances et le coût.

## Technologies proposées

Je limiterais volontairement les outils au début :

- **TP 1 à 3** : Python + NumPy (pour comprendre les bases sans magie)
- **TP 4 à 8** : PyTorch (framework de référence pour l'apprentissage profond)
- **TP 9 à 11** : Hugging Face (Transformers, Datasets, PEFT/LoRA) + plateformes cloud (Colab, Runpod, Vast.ai)

Ainsi, les apprenants progressent d'un simple neurone codé à la main jusqu'au fine-tuning d'un LLM moderne, tout en gardant une continuité dans les concepts et les outils.

# TP 7 — GAN : générer des chiffres avec deux réseaux qui s'affrontent

## Objectif

Tous les TP précédents étaient **discriminatifs** : on montrait une donnée au
réseau et il rendait une **étiquette** (un prix au TP 1, spam/pas spam au TP 2,
un chiffre 0-9 aux TP 4 et 6). Ce TP bascule dans le **génératif** : au lieu de
**reconnaître** un chiffre, on veut en **fabriquer** de nouveaux, qui ressemblent
à de vrais MNIST mais **n'existent dans aucun jeu de données**.

**La notion nouvelle de ce TP : l'entraînement adversaire** (_Generative
Adversarial Network_, Goodfellow 2014). On entraîne **deux réseaux en même
temps, l'un contre l'autre** :

- le **générateur** (G) part d'un vecteur de **bruit** aléatoire $\mathbf{z}$ et
  produit une image 28×28. C'est un **faussaire** qui fabrique de faux billets.
- le **discriminateur** (D) reçoit une image et dit si elle est **vraie** (issue
  de MNIST) ou **fausse** (fabriquée par G). C'est le **policier** qui traque les
  faux.

Les deux progressent ensemble : D apprend à mieux repérer les faux, ce qui force
G à faire des faux plus crédibles, ce qui force D à s'améliorer encore… À
l'équilibre, G produit des chiffres si réalistes que D ne fait plus la
différence (il répond « vrai » environ **une fois sur deux**, au hasard).

## Ce qui change par rapport au TP 6

| TP 6 (CNN)                                | TP 7 (GAN)                                          |
| ----------------------------------------- | --------------------------------------------------- |
| **reconnaître** un chiffre (0-9)          | **générer** de nouveaux chiffres                    |
| **un** réseau                             | **deux** réseaux (G et D) entraînés ensemble        |
| une perte, un optimiseur                  | **deux** pertes, **deux** optimiseurs               |
| on utilise les étiquettes 0-9             | on les **ignore** (on veut juste « ça ressemble »)  |
| on mesure une **accuracy** sur un test    | pas d'accuracy : on **juge à l'œil** les images     |
| pixels dans `[0, 1]`                      | pixels dans `[-1, 1]` (pour matcher la `Tanh` de G) |

On **réutilise** en revanche tout le reste : MNIST + `DataLoader` (TP 4), des MLP
`nn.Linear` (TP 3/4), `.to(device)`, la perte **BCE** binaire du TP 2 (le
discriminateur fait une classification vrai/faux), `Dropout` (TP 6).

## Les deux réseaux

```
GÉNÉRATEUR G                      DISCRIMINATEUR D
bruit z (64)                      image (1, 28, 28)
  Linear(64  → 160) + ReLU          Flatten            → 784
  Linear(160 → 256) + ReLU          Linear(784 → 256) + LeakyReLU + Dropout
  Linear(256 → 784) + Tanh          Linear(256 → 160) + LeakyReLU + Dropout
  reshape            → (1,28,28)    Linear(160 → 1)    → 1 logit (vrai/faux)
```

Les deux réseaux sont **en miroir** et pèsent chacun **~250 000 paramètres**
(≈ 253k pour G, ≈ 242k pour D).

- **G** _remonte_ le bruit vers une image ; sa dernière couche `Tanh` sort dans
  `[-1, 1]`, exactement la plage des vraies images normalisées.
- **D** est un simple classifieur binaire (comme au TP 2), qui sort **un logit**
  (la sigmoïde est appliquée par la perte `BCEWithLogitsLoss`).
- **`LeakyReLU`** au lieu de `ReLU` dans D : elle laisse passer un petit gradient
  même pour les entrées négatives. C'est la convention GAN, pour éviter que D
  « meure » et cesse d'envoyer du gradient utile à G.

## La boucle d'entraînement (le point clé)

À **chaque mini-batch**, on fait **deux pas de gradient**, dans cet ordre :

1. **Pas du discriminateur.** On lui montre un lot de **vraies** images (cible 1)
   et un lot de **fausses** produites par G (cible 0). On met à jour **D
   seulement** : les fausses sont `.detach()`ées pour que ce gradient **ne
   remonte pas** dans G.
2. **Pas du générateur.** On repasse les fausses images dans D (cette fois **sans
   `detach`**) avec la cible **1** : G cherche à faire dire « vrai » à D. Le
   gradient traverse D (figé) et met à jour **G seulement**.

Deux `nn.Module`, deux `torch.optim.Adam`, deux `loss.backward()` : c'est toute
la nouveauté. Le détail mathématique (le jeu minimax et pourquoi chaque perte
prend telle cible) est dans [MATH.md](MATH.md).

## Lancer l'entraînement

Depuis la racine du dépôt :

```
mise run tp7
```

ou depuis ce dossier : `uv run main.py`.

Le script télécharge MNIST dans `data/` (déjà présent si tu as fait le TP 4 ou
6), entraîne le GAN (GPU si disponible, sinon CPU), puis enregistre dans
**`resultat.png`** :

- une **mosaïque 8×8 de chiffres générés** à partir d'un bruit fixe ;
- les **deux courbes de perte** (D et G) au fil des epochs.

À **chaque fin d'epoch**, il enregistre aussi une bandelette de **10 images
tirées au hasard** dans le dossier **`apercus/`** (`epoch_01.png`,
`epoch_02.png`…), pour suivre à l'œil la progression du générateur. Ce dossier
est **ignoré par Git**.

Enfin, il **exporte le générateur au format ONNX** (`generateur.onnx`) et le
copie dans **`front/public/generateur.onnx`** pour le site (voir plus bas). Seul
le générateur est exporté : le discriminateur ne sert qu'à l'entraînement.

## Lancer le site de génération (front)

Comme au TP 6, le front utilise **Bun**, **Vite** et **TypeScript**, avec
`onnxruntime-web` pour l'inférence dans le navigateur.

```
mise run tp7-site
```

ou manuellement depuis ce dossier :

```
cd front
bun install
bun run dev
```

Ouvre l'URL affichée : à chaque clic sur **« Nouvelle image »**, le site tire un
vecteur de bruit $\mathbf{z}$ **au hasard** (un point de l'espace d'entrée du
générateur), l'envoie au modèle ONNX et affiche le chiffre 28×28 produit — fond
blanc, chiffre noir.

> Le site a besoin de `front/public/generateur.onnx`, donc lance d'abord
> l'entraînement (`mise run tp7`) au moins une fois.

## Ce que tu dois observer

- Les **premières epochs** produisent du bruit informe, puis des taches, puis
  peu à peu des formes de chiffres reconnaissables.
- Les **deux pertes ne descendent pas vers zéro** comme dans les TP précédents :
  elles oscillent dans un **équilibre**. C'est normal — c'est un bras de fer, pas
  une minimisation classique. Si l'une s'effondre à 0 et l'autre explose, le GAN
  s'est « déséquilibré » (voir ci-dessous).
- Les chiffres générés sont **variés** (on ne veut pas que G produise toujours le
  même « 8 » : ce défaut s'appelle le _mode collapse_).

## À expérimenter

- **Nombre d'epochs** (`EPOCHS`) : passe de 40 à 20 (plus flou) ou 80 (plus net,
  plus long). C'est le levier le plus visible.
- **Taille du bruit** (`DIM_BRUIT`) : 64 par défaut. Trop petit (ex. 2), G manque
  de « liberté » pour couvrir les 10 chiffres.
- **Learning rate** et **`BETA1`** : les GAN sont **instables**. `lr = 2e-4` et
  `beta1 = 0.5` sont les réglages classiques (DCGAN). Monte le `lr` à `1e-3` pour
  voir l'entraînement diverger.
- **Déséquilibre G/D** : entraîne D **deux fois** par batch (ajoute un second pas
  de D) et observe comment G a plus de mal à suivre.
- **De MLP à convolutions** : remplace les `Linear` par des `ConvTranspose2d`
  (dans G) et des `Conv2d` (dans D) pour obtenir un **DCGAN**, nettement plus net.
  C'est le prolongement naturel avec la convolution du TP 6.

## Aller plus loin

Un GAN entraîné sur MNIST est le plus petit exemple d'une famille qui a explosé :
génération de visages (StyleGAN), de photos, puis les modèles de **diffusion**
(Stable Diffusion, DALL·E) qui ont depuis largement pris le relais. Le principe
« un réseau qui fabrique, jugé par la qualité du résultat » reste central.

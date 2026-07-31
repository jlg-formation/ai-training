# TP 2 — Spam ou pas spam (classification binaire, 1 neurone)

## Objectif

Apprendre à un **seul neurone** à classer des phrases en deux catégories :

- **spam** (étiquette `1`)
- **non-spam** (étiquette `0`)

On réutilise **exactement** le neurone et la descente de gradient du TP 1. Les
nouveautés :

- le neurone a maintenant **V entrées** (une par mot du vocabulaire, $V$ = taille
  du vocabulaire), donc **V poids**, au lieu d'une seule entrée au TP 1 ;
- la sortie passe dans une **sigmoïde** : elle devient une **probabilité** entre
  `0` et `1` (au lieu d'une valeur quelconque) ;
- la perte devient la **BCE** (entropie croisée binaire) au lieu de la MSE.

Tout est fait **à la main avec NumPy**, sans framework d'IA.

## Des mots vers des nombres : le sac de mots

Un neurone calcule avec des nombres, pas avec du texte. On transforme donc
chaque phrase en **vecteur de comptes** : pour chaque mot d'un **vocabulaire**,
on compte combien de fois il apparaît dans la phrase. C'est le **sac de mots**
(*bag of words*).

Exemple avec le vocabulaire `[bonjour, gratuit, projet]` :

```
"gratuit gratuit projet"  ->  [0, 2, 1]
```

## Le modèle

Comme au TP 1, mais l'entrée $\mathbf{x}$ et les poids $\mathbf{w}$ sont
maintenant des **vecteurs** de taille **V** (un poids par mot du vocabulaire) :

$$
z = \mathbf{w} \cdot \mathbf{x} + b \qquad \text{(produit scalaire)}
$$

$$
p = \sigma(z) \qquad \text{(probabilité que la phrase soit du spam)}
$$

> **Convention de notation.** Les **vecteurs** sont en **gras minuscule**
> ($\mathbf{x}$, $\mathbf{w}$), la **matrice** de tous les sacs de mots en
> **majuscule grasse** ($\mathbf{X}$, voir [MATH.md](MATH.md)), et les
> **scalaires** en maigre ($z$, $b$).

### Topologie du réseau

```mermaid
graph LR
    x1(("x<sub>1</sub>")) -- "w<sub>1</sub>" --> N["neurone<br/>z = <b>w</b>·<b>x</b> + b<br/>p = σ(z)"]
    x2(("x<sub>2</sub>")) -- "w<sub>2</sub>" --> N
    x3(("x<sub>3</sub>")) -- "w<sub>3</sub>" --> N
    dots["⋮"]:::vide ~~~ N
    xv(("x<sub>V</sub>")) -- "w<sub>V</sub>" --> N
    b(("1")) -- "b" --> N
    N --> p(("p ∈ [0,1]"))
    classDef vide fill:none,stroke:none;
```

**V entrées** (une par mot du vocabulaire) et donc **V poids**, contre une seule
entrée au TP 1. Le neurone calcule d'abord le score $z = \mathbf{w}\cdot\mathbf{x} + b$,
puis lui applique **en interne** la **sigmoïde** $\sigma$ qui le transforme en
probabilité entre `0` et `1`. La sigmoïde fait donc **partie** du neurone.

Chaque **synapse** (poids `w`) est liée à **un mot du vocabulaire choisi**. Le
**nombre total de synapses = la taille V de ce vocabulaire**, et non le nombre de
mots d'une phrase. Un mot présent dans une phrase mais **absent du vocabulaire**
n'a aucune synapse : il est tout simplement ignoré. Un poids positif tire vers
« spam », un poids négatif vers « non-spam ».

**Que met-on dans le vocabulaire ?** C'est un **choix** (voir le switch
`VOCAB_COMPLET` plus bas). Dans ce TP, deux options :

- soit une **liste choisie à la main** (VOCAB_COMPLET=False), limitée aux mots jugés utiles
  (`mots_spam` + `mots_neutres`) ;
- soit **tous les mots vus dans le jeu d'entraînement** (y compris les mots
  rares et de remplissage) (VOCAB_COMPLET=True).

Dans les deux cas, on ne met **que des mots vus à l'entraînement** — jamais un
mot qui n'apparaît que dans le jeu de test (ce serait une **fuite de données**).

Au départ, tous les poids valent `0`. L'entraînement les ajuste pour que `p`
soit proche de `1` sur les spams et de `0` sur les messages normaux.

## Pourquoi la BCE et plus la MSE ?

Au TP 1 (régression), la sortie était une valeur quelconque et on mesurait
l'erreur avec la **MSE** (erreur quadratique moyenne). Ici la sortie est une
**probabilité**, et la MSE devient un mauvais choix. Trois raisons :

- **La MSE punit mal les erreurs de probabilité.** Prédire `p = 0.01` pour un
  vrai spam est une faute grave, mais la MSE ne la compte que `(1 - 0.01)² ≈ 1`.
  La BCE, elle, vaut `-log(0.01) ≈ 4.6` : elle **punit très fort une prédiction
  confiante ET fausse**, exactement ce qu'on veut.

- **La MSE freine l'apprentissage (gradients qui s'éteignent).** Combinée à la
  sigmoïde, la MSE fait apparaître un facteur `p·(1 - p)` dans le gradient. Quand
  le neurone se trompe **avec confiance** (`p` proche de `0` ou `1`), ce facteur
  devient minuscule : le gradient s'annule et le modèle **n'apprend presque
  plus**, même en pleine erreur.

- **La BCE donne un gradient propre.** Avec la sigmoïde, son gradient se
  simplifie exactement en `p - y` (voir [MATH.md](MATH.md)) : pas de facteur qui
  l'écrase, et on retrouve **la même mécanique qu'au TP 1**. Plus l'erreur est
  grande, plus la correction est forte.

En résumé : la BCE est la perte **naturelle** pour des probabilités ; la MSE
apprendrait plus lentement et resterait bloquée sur les erreurs les plus sûres.

## Comment ça marche

À chaque **epoch** :

1. On calcule les probabilités $\mathbf{p} = \sigma(\mathbf{X}\,\mathbf{w} + b)$.
2. On calcule l'erreur $\mathbf{p} - \mathbf{y}$.
3. On calcule le **gradient** (même forme qu'au TP 1, en version vecteur).
4. On déplace $\mathbf{w}$ et `b` dans le sens qui **fait diminuer la loss**.

Le détail mathématique (pourquoi le gradient se simplifie en `p - y`) est dans
[MATH.md](MATH.md).

## Le switch `VOCAB_COMPLET` : voir le surapprentissage

En haut de [main.py](main.py), une constante bascule entre deux régimes :

| `VOCAB_COMPLET` | Vocabulaire            | Exemples train | Ce qu'on observe                 |
| --------------- | ---------------------- | -------------- | -------------------------------- |
| `False`         | restreint (mots utiles) | 300            | le modèle **généralise** bien    |
| `True`          | complet (+ mots rares)  | 30             | le modèle **surapprend**         |

En mode `True`, le vocabulaire contient plein de **mots rares** sans lien avec
le spam. Avec peu d'exemples, le neurone leur attribue des poids au hasard et
**mémorise** le jeu d'entraînement : l'accuracy train grimpe à ~100 % mais
l'accuracy test décroche. C'est le **surapprentissage**.

## Lancer le TP (avec uv)

Depuis ce dossier :

```powershell
uv run main.py
```

Ou, depuis n'importe où dans le projet :

```powershell
mise run tp2
```

## Ce que tu dois observer

- La **loss diminue** sur l'entraînement.
- La sortie est une **probabilité** : sur les phrases-test, certaines phrases
  ambiguës tombent **près de 0.5** (le modèle hésite), ce qui montre l'intérêt
  de la sigmoïde par rapport à un simple `0/1`.
- En mode `VOCAB_COMPLET = True`, la **loss de test remonte** alors que la loss
  d'entraînement continue de baisser (`resultat.png`) : c'est la signature du
  surapprentissage.

## À expérimenter

- Passe `VOCAB_COMPLET` de `False` à `True` et compare les deux `resultat.png`.
- Change `TAILLE_TRAIN` : avec plus d'exemples, l'overfitting disparaît-il ?
- Change `LEARNING_RATE` ou `EPOCHS`.
- Ajoute tes propres phrases dans `phrases_demo` et regarde la probabilité de
  spam prédite.
- Ajoute des mots à `mots_spam` / `mots_neutres` : le modèle les exploite-t-il ?

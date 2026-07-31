# TP 2 — Spam ou pas spam (classification binaire, 1 neurone)

## Objectif

Apprendre à un **seul neurone** à classer des phrases en deux catégories :

- **spam** (étiquette `1`)
- **non-spam** (étiquette `0`)

On réutilise **exactement** le neurone et la descente de gradient du TP 1. La
**seule** nouveauté :

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

Comme au TP 1, mais l'entrée `x` et les poids `w` sont maintenant des
**vecteurs** (un poids par mot) :

```
z = w · x + b            (produit scalaire)
p = sigmoid(z)           probabilité que la phrase soit du spam
```

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

1. On calcule les probabilités `p = sigmoid(w·x + b)`.
2. On calcule l'erreur `p - y`.
3. On calcule le **gradient** (même forme qu'au TP 1, en version vecteur).
4. On déplace `w` et `b` dans le sens qui **fait diminuer la loss**.

Le détail mathématique (pourquoi le gradient se simplifie en `p - y`) est dans
[MATH.md](MATH.md).

## Le bouton `VOCAB_COMPLET` : voir le surapprentissage

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

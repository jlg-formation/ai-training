"""
TP 2 - Spam ou pas spam : classification binaire (un seul neurone)

On garde EXACTEMENT le même neurone et la même descente de gradient qu'au TP 1.
La seule nouveauté :
  - la sortie passe dans une SIGMOÏDE  ->  elle devient une PROBABILITÉ (entre 0 et 1)
  - la perte devient la BCE (entropie croisée binaire) au lieu de la MSE

But : classer des phrases en « spam » (1) ou « non spam » (0).

On n'utilise QUE NumPy. Les phrases sont générées en français (pas de
téléchargement, résultats reproductibles). Chaque phrase est transformée en
nombres par un « sac de mots » (bag of words) : on compte les mots d'un
vocabulaire.

Deux modes, pilotés par la constante VOCAB_COMPLET :
  - False : vocabulaire restreint aux mots discriminants -> le modèle généralise
  - True  : vocabulaire complet + peu d'exemples -> SURAPPRENTISSAGE (overfitting)

Lancer avec :  uv run main.py
"""

import numpy as np

# Pour que les résultats soient reproductibles d'une exécution à l'autre
np.random.seed(0)


# ======================================================================
# CONSTANTE PÉDAGOGIQUE : bascule entre run « sain » et run « overfit »
# ======================================================================
# False -> vocabulaire restreint (mots vraiment utiles) + beaucoup d'exemples d'entraînement
# True  -> vocabulaire complet (avec plein de mots rares) + peu d'exemples d'entraînement
#          => le modèle mémorise les mots rares : train ~100 %, test moins bon.
VOCAB_COMPLET = False


# ----------------------------------------------------------------------
# 1) Génération des données : des phrases spam / non-spam en français
# ----------------------------------------------------------------------
# Trois familles de mots :
#   - mots_spam    : signalent le spam ("gratuit", "promo", ...)
#   - mots_neutres : signalent un message normal ("bonjour", "réunion", ...)
#   - mots_divers  : mots de remplissage SANS signal (bruit). Ils sont ignorés
#                    en vocabulaire restreint, mais deviennent des pièges à
#                    surapprentissage en vocabulaire complet (mots rares).
mots_spam = [
    "gratuit", "promotion", "achetez", "urgent", "gagné", "argent",
    "cliquez", "offre", "cadeau", "félicitations", "réduction", "crédit",
]
mots_neutres = [
    "bonjour", "réunion", "demain", "projet", "rendez-vous", "merci",
    "document", "déjeuner", "collègue", "rapport", "midi", "café",
]
mots_divers = [
    "chat", "chien", "maison", "voiture", "livre", "musique", "jardin",
    "vacances", "pluie", "soleil", "train", "ville", "montagne", "cuisine",
    "film", "photo", "route", "fleur", "table", "fenêtre", "porte", "stylo",
    "papier", "horloge", "lampe", "clé", "vélo", "bateau", "avion", "nuage",
]


def generer_phrase(est_spam):
    """Fabrique une phrase (liste de mots) selon sa classe.

    On mélange un peu les familles pour créer des cas AMBIGUS : un spam peut
    contenir un mot neutre, et inversement. C'est ce chevauchement qui rend la
    sigmoïde utile (des probabilités proches de 0.5).
    """
    if est_spam:
        signal, autre = mots_spam, mots_neutres
    else:
        signal, autre = mots_neutres, mots_spam

    mots = []
    # 2 à 3 mots de la « bonne » famille (le vrai signal)
    mots += list(np.random.choice(signal, size=np.random.randint(2, 4), replace=False))
    # parfois 1 mot de l'autre famille -> ambiguïté volontaire
    if np.random.rand() < 0.35:
        mots += list(np.random.choice(autre, size=1))
    # 1 à 2 mots de remplissage (bruit sans signal)
    mots += list(np.random.choice(mots_divers, size=np.random.randint(1, 3), replace=False))

    np.random.shuffle(mots)
    return " ".join(mots)


def generer_jeu(n):
    """Génère n phrases avec une étiquette 0 (non-spam) ou 1 (spam)."""
    phrases = []
    etiquettes = []
    for _ in range(n):
        est_spam = np.random.rand() < 0.5
        phrases.append(generer_phrase(est_spam))
        etiquettes.append(1 if est_spam else 0)
    return phrases, np.array(etiquettes, dtype=float)


# On génère un grand jeu, puis on le coupe en train / test.
# Le jeu de TEST est FIXE (mêmes phrases dans les deux modes) pour pouvoir
# comparer honnêtement.
phrases_all, y_all = generer_jeu(600)

phrases_test = phrases_all[:200]
y_test = y_all[:200]
phrases_pool = phrases_all[200:]      # réserve d'entraînement
y_pool = y_all[200:]

# Taille du jeu d'entraînement : PETITE en mode overfit, grande sinon.
TAILLE_TRAIN = 30 if VOCAB_COMPLET else 300
phrases_train = phrases_pool[:TAILLE_TRAIN]
y_train = y_pool[:TAILLE_TRAIN]


# ----------------------------------------------------------------------
# 2) Sac de mots : transformer une phrase en vecteur de nombres
# ----------------------------------------------------------------------
# Le vocabulaire est la liste des mots que le modèle « connaît ».
# On construit un vecteur qui compte, pour chaque mot du vocabulaire,
# combien de fois il apparaît dans la phrase.
if VOCAB_COMPLET:
    # Tout le vocabulaire vu dans le jeu d'entraînement (y compris les mots rares)
    mots_vus = set()
    for phrase in phrases_train:
        mots_vus.update(phrase.split())
    vocabulaire = sorted(mots_vus)
else:
    # Vocabulaire restreint aux seuls mots discriminants
    vocabulaire = sorted(mots_spam + mots_neutres)

# V = taille du vocabulaire = nombre d'ENTRÉES (et de poids) du neurone.
V = len(vocabulaire)

index_mot = {mot: i for i, mot in enumerate(vocabulaire)}


def vectoriser(phrases):
    """Transforme une liste de phrases en matrice (n_phrases x V)."""
    X = np.zeros((len(phrases), V))
    for i, phrase in enumerate(phrases):
        for mot in phrase.split():
            if mot in index_mot:            # les mots hors vocabulaire sont ignorés
                X[i, index_mot[mot]] += 1
    return X


X_train = vectoriser(phrases_train)
X_test = vectoriser(phrases_test)


# ----------------------------------------------------------------------
# 3) Le modèle : un seul neurone + sigmoïde  ->  probabilité de spam
# ----------------------------------------------------------------------
# z = w . x + b          (comme au TP 1, mais x et w sont des vecteurs)
# p = sigmoid(z)         (on écrase z entre 0 et 1 : c'est une PROBABILITÉ)
w = np.zeros(V)   # V poids : un par mot du vocabulaire (= une entrée du neurone)
b = 0.0


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def predire_proba(X, w, b):
    """Probabilité que chaque phrase soit du spam (entre 0 et 1)."""
    return sigmoid(X @ w + b)


# ----------------------------------------------------------------------
# 4) La fonction de perte (loss) : entropie croisée binaire (BCE)
# ----------------------------------------------------------------------
# Elle punit fortement une prédiction confiante ET fausse.
#
# ATTENTION : cette fonction ne sert QU'À AFFICHER le progrès (les prints et la
# courbe). Elle n'est JAMAIS appelée pour entraîner le modèle. Ce qui entraîne,
# c'est le GRADIENT (étape 5). Et ce gradient EST la dérivée de CETTE loss :
# passer de la MSE (TP 1) à la BCE (TP 2), c'est justement changer la formule du
# gradient. Avec la BCE + sigmoïde, cette dérivée se simplifie en  erreur = p - y.
def perte(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)          # évite log(0)
    return np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))


def accuracy(p, y):
    """Proportion de bonnes réponses (on tranche à 0.5)."""
    predictions = (p >= 0.5).astype(float)
    return np.mean(predictions == y)


# ----------------------------------------------------------------------
# 5) L'entraînement : la descente de gradient (identique au TP 1)
# ----------------------------------------------------------------------
# Le gradient de la BCE avec sigmoïde se simplifie en  erreur = p - y
# (voir MATH.md). On retrouve donc EXACTEMENT la même mécanique qu'au TP 1.
LEARNING_RATE = 0.5
EPOCHS = 400


def main():
    global w, b

    print(f"Mode VOCAB_COMPLET = {VOCAB_COMPLET}")
    print(f"Taille du vocabulaire V (nb d'entrées/poids) : {V}")
    print(f"Exemples d'entraînement : {len(y_train)} | de test : {len(y_test)}\n")

    historique_train = []
    historique_test = []

    for epoch in range(EPOCHS):
        # a) Prédictions (probabilités) sur le jeu d'entraînement
        p_train = predire_proba(X_train, w, b)

        # b) Erreur entre probabilité prédite et vraie étiquette
        erreur = p_train - y_train

        # c) Gradients (même forme qu'au TP 1, en version vecteur)
        # C'est ICI que le choix BCE vs MSE se joue : cette ligne EST la dérivée
        # de la BCE. Avec la MSE on aurait dû écrire un gradient différent.
        grad_w = X_train.T @ erreur / len(y_train)
        grad_b = np.mean(erreur)

        # d) Mise à jour des poids
        w -= LEARNING_RATE * grad_w
        b -= LEARNING_RATE * grad_b

        # Suivi de la loss sur train ET test (pour voir le surapprentissage)
        # NB : perte() n'est utilisée QUE pour ce suivi/affichage, pas pour
        # mettre à jour w et b (ça, c'est le gradient ci-dessus).
        historique_train.append(perte(p_train, y_train))
        historique_test.append(perte(predire_proba(X_test, w, b), y_test))

        if epoch % 40 == 0 or epoch == EPOCHS - 1:
            print(
                f"Epoch {epoch:3d} | loss_train = {historique_train[-1]:.4f} "
                f"| loss_test = {historique_test[-1]:.4f}"
            )

    # ------------------------------------------------------------------
    # 6) Résultat : accuracy et écart train / test
    # ------------------------------------------------------------------
    acc_train = accuracy(predire_proba(X_train, w, b), y_train)
    acc_test = accuracy(predire_proba(X_test, w, b), y_test)
    print(f"\nAccuracy train : {acc_train:.1%}")
    print(f"Accuracy test  : {acc_test:.1%}")
    if acc_train - acc_test > 0.1:
        print("=> Gros écart train/test : le modèle SURAPPREND (overfitting).")

    # ------------------------------------------------------------------
    # 7) La sortie est une PROBABILITÉ, pas un simple 0/1
    # ------------------------------------------------------------------
    # On affiche la proba de spam sur quelques phrases, dont des cas ambigus
    # qui doivent tomber près de 0.5.
    # Chaque phrase vient avec sa réponse ATTENDUE : 1 = spam, 0 = normal.
    phrases_demo = [
        ("gratuit promotion cliquez", 1),      # très spam
        ("gagné cadeau urgent argent", 1),     # très spam
        ("bonjour réunion demain", 0),         # clairement normal
        ("rendez-vous document midi", 0),      # clairement normal
        ("gratuit réunion", 1),                # ambigu (1 spam + 1 neutre) : proba ~0.5
        ("offre argent bonjour demain", 0),    # ambigu (2 spam + 2 neutres) : proba ~0.5
    ]
    textes = [phrase for phrase, _ in phrases_demo]
    proba_demo = predire_proba(vectoriser(textes), w, b)
    print("\nProbabilité de SPAM sur des phrases de démonstration :")
    for (phrase, attendu), proba in zip(phrases_demo, proba_demo):
        prediction = 1 if proba >= 0.5 else 0
        predit = "SPAM  " if prediction == 1 else "normal"
        voulu = "SPAM  " if attendu == 1 else "normal"
        if 0.35 < proba < 0.65:                 # proba proche de 0.5 : le neurone hésite
            verdict = "? "
        elif prediction == attendu:
            verdict = "OK"
        else:
            verdict = "KO"
        print(
            f"  [{verdict}]  proba={proba:5.1%}  "
            f"prédit={predit}  attendu={voulu}  \"{phrase}\""
        )

    # ------------------------------------------------------------------
    # 8) Visualisation : la loss train vs test au fil des epochs
    # ------------------------------------------------------------------
    import matplotlib.pyplot as plt

    plt.plot(historique_train, label="loss entraînement", color="blue")
    plt.plot(historique_test, label="loss test", color="red")
    titre = "TP 2 - Spam : "
    titre += "SURAPPRENTISSAGE" if VOCAB_COMPLET else "run sain"
    plt.title(titre)
    plt.xlabel("epoch")
    plt.ylabel("loss (BCE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("resultat.png", dpi=120)
    print("\nGraphique enregistré dans resultat.png")


if __name__ == "__main__":
    main()

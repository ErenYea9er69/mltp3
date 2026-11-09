"""Métriques d'évaluation pour l'Atelier 3.

Catégories : facturation, technique, compte, commercial.
"""

CATEGORIES = ["facturation", "technique", "compte", "commercial"]


def accuracy(vrais, predits):
    """Part de tickets correctement classés."""
    if not vrais:
        return 0.0
    bons = sum(1 for v, p in zip(vrais, predits) if v == p)
    return bons / len(vrais)


def scores_par_classe(vrais, predits, classes=CATEGORIES):
    """Calcule precision, recall, f1 et support pour chaque classe."""
    resultat = {}
    for c in classes:
        vp = sum(1 for v, p in zip(vrais, predits) if v == c and p == c)
        fp = sum(1 for v, p in zip(vrais, predits) if v != c and p == c)
        fn = sum(1 for v, p in zip(vrais, predits) if v == c and p != c)
        precision = vp / (vp + fp) if (vp + fp) else 0.0
        recall = vp / (vp + fn) if (vp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        resultat[c] = {"precision": precision, "recall": recall, "f1": f1, "support": vp + fn}
    return resultat


def f1_macro(scores):
    """Moyenne simple des scores F1 par classe."""
    if not scores:
        return 0.0
    return sum(d["f1"] for d in scores.values()) / len(scores)


def taux_invalides(predits, classes=CATEGORIES):
    """Part des réponses hors nomenclature."""
    if not predits:
        return 0.0
    return sum(1 for p in predits if p not in classes) / len(predits)


def matrice_confusion(vrais, predits, classes=CATEGORIES):
    """Matrice de confusion avec colonne invalide si besoin."""
    colonnes = list(classes) + ["invalide"]
    m = {v: {p: 0 for p in colonnes} for v in classes}
    for v, p in zip(vrais, predits):
        m[v][p if p in classes else "invalide"] += 1
    return m, colonnes


def afficher_rapport(nom, vrais, predits, classes=CATEGORIES):
    """Affiche un rapport détaillé avec tableau et matrice de confusion."""
    scores = scores_par_classe(vrais, predits, classes)
    acc = accuracy(vrais, predits)
    macro = f1_macro(scores)
    inval = taux_invalides(predits, classes)

    print(f"\n=== {nom} ===")
    print(f"accuracy    : {acc:.3f}")
    print(f"macro-F1    : {macro:.3f}")
    print(f"invalides   : {inval:.3f}")
    print(f"\n{'categorie':<13}{'precision':>11}{'recall':>9}{'F1':>8}{'n':>6}")
    for c, d in scores.items():
        print(f"{c:<13}{d['precision']:>11.3f}{d['recall']:>9.3f}{d['f1']:>8.3f}{d['support']:>6d}")

    m, colonnes = matrice_confusion(vrais, predits, classes)
    largeur = max(len(c) for c in colonnes) + 2
    print("\nConfusion matrix (lignes = verite, colonnes = prediction)")
    print(" " * 13 + "".join(f"{c[:6]:>{largeur}}" for c in colonnes))
    for v in classes:
        print(f"{v:<13}" + "".join(f"{m[v][p]:>{largeur}d}" for p in colonnes))

    return {
        "accuracy": acc,
        "f1_macro": macro,
        "invalides": inval,
        "par_classe": scores,
    }


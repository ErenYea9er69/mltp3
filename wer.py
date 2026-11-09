"""Mesure des erreurs de transcription : le WER (Word Error Rate) (Étape 2).

On aligne la transcription sur la référence, mot à mot, et on compte :
  S : substitutions (un mot remplacé par un autre)
  D : suppressions / délétions (un mot oublié)
  I : insertions (un mot en trop)

  WER = (S + D + I) / N    où N = nombre de mots de la référence.
"""
import re
import jiwer


def normaliser(texte):
    """Met référence et transcription dans la même forme normalisée avant comparaison.

    Règles :
      - Minuscules
      - Conversion de € en ' euros '
      - Remplacement des apostrophes (' et ’) et tirets (-) par des espaces
      - Suppression de la ponctuation (conservation des accents et chiffres)
      - Réduction des espaces multiples
    """
    if not texte:
        return ""

    texte = texte.lower()
    # Remplacer le symbole euro
    texte = texte.replace("€", " euros ")
    # Remplacer apostrophes (droite et courbe) et tirets par des espaces
    texte = re.sub(r"['’\-]", " ", texte)
    # Supprimer la ponctuation restante tout en gardant lettres (avec accents) et chiffres
    texte = re.sub(r"[^\w\s]", " ", texte)
    # Réduire les espaces multiples
    return " ".join(texte.split())


def wer(references, hypotheses):
    """Calcule le WER global d'un corpus (total erreurs / total mots de référence)."""
    return jiwer.wer(list(references), list(hypotheses))


def detail(reference, hypothese):
    """Renvoie les compteurs S, D, I, N et le WER d'un seul message."""
    m = jiwer.process_words(reference, hypothese)
    return {
        "S": m.substitutions,
        "D": m.deletions,
        "I": m.insertions,
        "N": len(reference.split()),
        "wer": m.wer,
    }


def afficher_alignement(reference, hypothese):
    """Affiche l'alignement mot à mot entre référence et hypothèse."""
    m = jiwer.process_words(reference, hypothese)
    lignes = jiwer.visualize_alignment(m, show_measures=False).splitlines()
    print("\n".join(l for l in lignes if l.strip() and not l.startswith("===")))


if __name__ == "__main__":
    ref = "Ma facture de mars est de 89 euros. Pouvez-vous m'expliquer ?"
    hyp = "ma facture de mars et de 89 € pouvez vous m'expliquer"
    print("brut       :", round(wer([ref], [hyp]), 3))
    print("normalisé  :", round(wer([normaliser(ref)], [normaliser(hyp)]), 3))
    afficher_alignement(normaliser(ref), normaliser(hyp))


"""Analyse de propagation des erreurs entre texte et voix (Étape 4).

Usage :
    python propagation.py whisper-large-v3-turbo
    python propagation.py whisper-large-v3-turbo auto
"""
import json
import sys
from pathlib import Path


def propagation(vrais, predits_texte, predits_voix):
    """Range chaque ticket dans une des quatre cases selon l'exactitude des prédictions.

      "conserve"  : juste sur le texte, juste sur la voix
      "perdu"     : juste sur le texte, faux sur la voix   (erreur introduite par l'audio / ASR)
      "gagne"     : faux sur le texte, juste sur la voix   (effet du hasard)
      "deux_faux" : faux sur le texte, faux sur la voix    (erreur imputable au classifieur)
    """
    cases = {"conserve": [], "perdu": [], "gagne": [], "deux_faux": []}
    for i, (v, t, a) in enumerate(zip(vrais, predits_texte, predits_voix)):
        t_ok = (t == v)
        a_ok = (a == v)
        if t_ok and a_ok:
            cases["conserve"].append(i)
        elif t_ok and not a_ok:
            cases["perdu"].append(i)
        elif not t_ok and a_ok:
            cases["gagne"].append(i)
        else:
            cases["deux_faux"].append(i)
    return cases


def charger(nom):
    """Charge un fichier de résultat JSON depuis le dossier resultats/."""
    chemin = Path("resultats") / f"{nom}.json"
    if not chemin.exists():
        raise FileNotFoundError(f"Fichier de résultat introuvable : {chemin}. Exécutez d'abord evaluer_voix.py.")
    with open(chemin, encoding="utf-8") as f:
        return json.load(f)


def afficher_propagation(texte, voix):
    """Affiche le tableau de contingence de propagation et le détail des tickets perdus."""
    if texte["ids"] != voix["ids"]:
        sys.exit("Les deux évaluations ne portent pas sur les mêmes tickets.")
    vrais = texte["vrais"]
    cases = propagation(vrais, texte["predits"], voix["predits"])
    n = len(vrais)

    print(f"\n{'':<14}{'voix juste':>12}{'voix fausse':>13}")
    print(f"{'texte juste':<14}{len(cases['conserve']):>12}{len(cases['perdu']):>13}")
    print(f"{'texte faux':<14}{len(cases['gagne']):>12}{len(cases['deux_faux']):>13}")
    print(f"\n{len(cases['perdu'])} tickets perdus à cause de l'audio, "
          f"{len(cases['gagne'])} gagnés par hasard, sur {n}.")

    print("\n--- Détail des tickets perdus à cause de l'audio ---")
    for i in cases["perdu"]:
        wer_msg = voix.get("wer_par_message", [0.0] * n)[i]
        print(f"\nTicket {voix['ids'][i]}  [Vérité: {vrais[i]}]  -> Prédit voix : {voix['predits'][i]} (WER: {wer_msg:.2f})")
        print(f"   Lu de référence : {texte['textes'][i]}")
        print(f"   Entendu Whisper : {voix['textes'][i]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage : python propagation.py <modele_whisper> [fr|auto]")
    modele = sys.argv[1]
    langue = sys.argv[2] if len(sys.argv) > 2 else "fr"
    nom_voix = f"voix__{modele}__{langue}"
    afficher_propagation(charger("texte"), charger(nom_voix))


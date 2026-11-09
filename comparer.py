"""Comparaison des trois canaux : texte, voix, image (Étape 6).

Usage :
    python comparer.py
"""
import glob
import json
from pathlib import Path


def charger_tous():
    """Charge l'ensemble des fichiers de résultats dans resultats/."""
    resumes = []
    for chemin in sorted(glob.glob("resultats/*.json")):
        with open(chemin, encoding="utf-8") as f:
            resumes.append(json.load(f))
    return resumes


def tableau(resumes):
    """Affiche le tableau comparatif des canaux évalués."""
    print(f"{'Canal':<10}{'Modèle':<26}{'Langue':<8}{'WER':>8}{'Macro-F1':>11}{'Invalides':>11}{'N':>5}{'Latence':>10}")
    print("-" * 92)
    for r in sorted(resumes, key=lambda x: (x.get("canal", ""), -x.get("f1_macro", 0))):
        wer_str = f"{r['wer']:.3f}" if "wer" in r else "-"
        n = len(r.get("ids", []))
        nom_mod = str(r.get("modele", "-"))[:25]
        print(f"{r.get('canal', '-'):<10}{nom_mod:<26}{r.get('langue', '-'):<8}{wer_str:>8}"
              f"{r.get('f1_macro', 0.0):>11.3f}{r.get('invalides', 0.0):>11.3f}{n:>5}{r.get('latence_s', 0.0):>9.2f}s")


def tableau_markdown(resumes):
    """Génère le tableau au format Markdown pour le rapport."""
    lignes = [
        "| Canal | Modèle | Qualité (Macro-F1 / WER) | Latence moy. | N tickets |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for r in sorted(resumes, key=lambda x: (x.get("canal", ""), -x.get("f1_macro", 0))):
        qualite = f"Macro-F1 : {r.get('f1_macro', 0.0):.3f}"
        if "wer" in r:
            qualite += f" (WER : {r['wer']:.3f})"
        nom_mod = str(r.get("modele", "-"))
        lignes.append(f"| **{r.get('canal', '-')}** | `{nom_mod}` | {qualite} | {r.get('latence_s', 0.0):.2f} s | {len(r.get('ids', []))} |")
    return "\n".join(lignes)


if __name__ == "__main__":
    tous = charger_tous()
    if not tous:
        raise SystemExit("Aucun résultat dans resultats/. Lancez d'abord les évaluations.")
    tableau(tous)
    print("\n--- Tableau au format Markdown ---")
    print(tableau_markdown(tous))
    print("\nRappel : 1 ticket sur 72 = ~1,4 point de macro-F1 ; 1 image sur 20 = 5 points.")


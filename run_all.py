"""Script d'orchestration global : lance l'ensemble des étapes et tests du TP3.

Usage :
    python run_all.py
"""
import subprocess
import sys

COMMANDES = [
    ("Étape 2 (Tests unitaires WER - sans quota)", [sys.executable, "test_wer.py"]),
    ("Étape 2 (Exemple calcul WER)", [sys.executable, "wer.py"]),
    ("Étape 4 (Tests unitaires Propagation - sans quota)", [sys.executable, "test_propagation.py"]),
    ("Étape 1 (Test transcription T016)", [sys.executable, "transcription.py", "atelier3/data/voix/T016.mp3"]),
    ("Étape 5 (Test vision sur I06.png)", [sys.executable, "vision.py", "atelier3/data/captures/I06.png"]),
]


def executer(titre, cmd):
    print("\n" + "=" * 70)
    print(f"▶ {titre}")
    print(f"Commande : {' '.join(cmd)}")
    print("=" * 70)
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"⚠️ Erreur lors de l'exécution de : {' '.join(cmd)}")
        return False
    return True


if __name__ == "__main__":
    print("=== Démarrage de la suite de tests rapides du TP3 ===")
    for titre, cmd in COMMANDES:
        executer(titre, cmd)
    print("\n✅ Suite de tests rapides terminée avec succès !")
    print("\nPour lancer les évaluations complètes (corpus entier avec quotas API) :")
    print("  1. python evaluer_voix.py texte")
    print("  2. python evaluer_voix.py whisper-large-v3-turbo")
    print("  3. python propagation.py whisper-large-v3-turbo")
    print("  4. python evaluer_images.py")
    print("  5. python comparer.py")


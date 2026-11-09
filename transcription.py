"""Transcription audio avec Whisper sur l'API Groq (Étape 1).

Usage :
    python transcription.py atelier3/data/voix/T016.mp3
    python transcription.py atelier3/data/voix/T006.mp3 whisper-large-v3-turbo auto
"""
import json
import os
import sys
import time
from pathlib import Path

from groq import Groq, RateLimitError

client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

# Formats acceptés par l'API de transcription de Groq (25 Mo max)
FORMATS = {".flac", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".ogg", ".wav", ".webm"}


def trouver_dossier_data():
    """Détecte l'emplacement du dossier data."""
    for p in [Path("atelier3/data"), Path("data"), Path("../atelier3/data"), Path("../data")]:
        if p.exists() and (p / "messages.csv").exists():
            return p
    return Path("atelier3/data")


DOSSIER_DATA = trouver_dossier_data()
DOSSIER_CACHE = DOSSIER_DATA / "transcriptions"


def transcrire(chemin, modele="whisper-large-v3-turbo", langue="fr"):
    """Envoie un fichier audio à Whisper et renvoie le texte transcrit.

    langue="fr" impose le français ; langue=None laisse Whisper deviner la langue.
    """
    chemin = Path(chemin)
    if not chemin.exists():
        raise FileNotFoundError(f"Fichier audio introuvable : {chemin}")

    with open(chemin, "rb") as f:
        octets = f.read()

    parametres = {
        "file": (chemin.name, octets),
        "model": modele,
        "temperature": 0.0,
    }
    if langue is not None:
        parametres["language"] = langue

    resultat = client.audio.transcriptions.create(**parametres)
    return resultat.text


def transcription_en_cache(chemin, modele="whisper-large-v3-turbo", langue="fr"):
    """Transcrit un fichier UNE fois : le résultat est gardé en cache dans data/transcriptions/.

    Renvoie (texte, latence en secondes, deja_en_cache).
    """
    chemin = Path(chemin)
    if chemin.suffix.lower() not in FORMATS:
        raise ValueError(f"{chemin.name} : format refusé par Groq. Acceptés : {' '.join(sorted(FORMATS))}")

    dossier = DOSSIER_CACHE / f"{chemin.parent.name}__{modele}__{langue or 'auto'}"
    cache = dossier / f"{chemin.stem}.json"
    if cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
        return d["texte"], d["latence_s"], True

    for tentative in range(4):
        try:
            debut = time.time()
            texte = transcrire(chemin, modele, langue)
            latence = time.time() - debut
            break
        except RateLimitError:
            attente = 15 * (tentative + 1)
            print(f"\n  quota audio atteint, pause de {attente} s")
            time.sleep(attente)
    else:
        raise RuntimeError("quota toujours atteint après 4 tentatives")

    dossier.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"texte": texte, "latence_s": round(latence, 2)}, ensure_ascii=False), encoding="utf-8")
    return texte, latence, False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage : python transcription.py <fichier audio> [modele] [fr|auto]")
    modele = sys.argv[2] if len(sys.argv) > 2 else "whisper-large-v3-turbo"
    langue = sys.argv[3] if len(sys.argv) > 3 else "fr"
    langue_param = None if langue == "auto" else langue

    debut = time.time()
    texte = transcrire(sys.argv[1], modele, langue_param)
    print(texte)
    print(f"({time.time() - debut:.1f} s)")


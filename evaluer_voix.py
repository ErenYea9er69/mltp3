"""Chaîne vocale complète : audio -> Whisper -> texte -> Classifieur -> catégorie (Étape 3).

Usage :
    python evaluer_voix.py texte                            # Point de référence : le texte lu
    python evaluer_voix.py whisper-large-v3-turbo           # Voix avec français imposé
    python evaluer_voix.py whisper-large-v3-turbo auto      # Voix avec détection auto
"""
import csv
import json
import sys
import time
from pathlib import Path

import mutagen

from classifieur import classer
from evaluation import afficher_rapport
from transcription import transcription_en_cache
from wer import detail, normaliser, wer

MODELE_TEXTE = "qwen/qwen3.8-27b"
PAUSE_WHISPER = 3.5  # 20 requêtes / min sur le palier gratuit
PAUSE_TEXTE = 2.5    # 30 requêtes / min


def trouver_dossier_data():
    for p in [Path("atelier3/data"), Path("data"), Path("../atelier3/data"), Path("../data")]:
        if p.exists() and (p / "messages.csv").exists():
            return p
    return Path("atelier3/data")


DATA = trouver_dossier_data()


def charger_messages():
    with open(DATA / "messages.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def nom_fichier_clean(texte):
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in texte)


def classer_avec_cache(textes):
    """Classe chaque texte avec cache persistant dans data/classifications/."""
    chemin = DATA / "classifications" / f"{nom_fichier_clean(MODELE_TEXTE)}.json"
    cache = json.loads(chemin.read_text(encoding="utf-8")) if chemin.exists() else {}
    predits, latences = [], []

    for i, texte in enumerate(textes, 1):
        if texte not in cache:
            debut = time.time()
            categorie = classer(texte, MODELE_TEXTE)
            latence = round(time.time() - debut, 2)
            cache[texte] = {"categorie": categorie, "latence_s": latence}
            chemin.parent.mkdir(parents=True, exist_ok=True)
            chemin.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
            time.sleep(PAUSE_TEXTE)
        predits.append(cache[texte]["categorie"])
        latences.append(cache[texte]["latence_s"])
        print(f"\r  classification {i}/{len(textes)}", end="", flush=True)
    print()
    return predits, latences


def transcrire_tout(fichiers, modele, langue):
    """Transcrit l'ensemble des fichiers audio avec gestion du cache et calcul des secondes facturées."""
    textes, latences, secondes = [], [], 0.0
    for i, fichier in enumerate(fichiers, 1):
        texte, latence, en_cache = transcription_en_cache(fichier, modele, langue)
        textes.append(texte)
        latences.append(latence)
        duree = mutagen.File(fichier).info.length if mutagen.File(fichier) else 10.0
        secondes += max(10.0, duree)  # 10s facturées au minimum par fichier audio
        print(f"\r  transcription {i}/{len(fichiers)}", end="", flush=True)
        if not en_cache:
            time.sleep(PAUSE_WHISPER)
    print()
    return textes, latences, secondes


def mesurer_wer(references, transcriptions):
    """Calcule le WER brut et le WER normalisé."""
    refs_n = [normaliser(r) for r in references]
    hyps_n = [normaliser(t) for t in transcriptions]
    return {
        "wer_brut": wer(references, transcriptions),
        "wer": wer(refs_n, hyps_n),
        "wer_par_message": [detail(r, h)["wer"] for r, h in zip(refs_n, hyps_n)],
    }


def enregistrer(resume, nom):
    Path("resultats").mkdir(exist_ok=True)
    chemin = Path("resultats") / f"{nom_fichier_clean(nom)}.json"
    chemin.write_text(json.dumps(resume, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRésultat enregistré dans {chemin}")


def evaluer_texte():
    """Évaluation du texte de référence directement sur le classifieur."""
    messages = charger_messages()
    textes = [m["texte_oral"] for m in messages]
    vrais = [m["categorie"] for m in messages]
    predits, latences = classer_avec_cache(textes)
    resume = afficher_rapport("texte lu -> " + MODELE_TEXTE, vrais, predits)
    lat_moy = round(sum(latences) / len(latences), 2)
    resume.update(
        etiquette="texte",
        canal="texte",
        modele="-",
        langue="-",
        ids=[m["id"] for m in messages],
        vrais=vrais,
        predits=predits,
        textes=textes,
        latence_s=lat_moy,
    )
    print(f"\nLatence moyenne par ticket : {lat_moy:.2f} s")
    enregistrer(resume, "texte")


def evaluer_voix(modele, langue):
    """Évaluation de la chaîne vocale complète : voix -> Whisper -> classifieur."""
    messages = charger_messages()
    references = [m["texte_oral"] for m in messages]
    vrais = [m["categorie"] for m in messages]
    fichiers = [DATA / m["fichier"] for m in messages]

    transcriptions, lat_asr, secondes = transcrire_tout(fichiers, modele, langue)
    mesures = mesurer_wer(references, transcriptions)

    print(f"\nWER sans normalisation : {mesures['wer_brut']:.3f}")
    print(f"WER normalisé          : {mesures['wer']:.3f}")

    par_voix = {}
    for v in sorted({m["voix"] for m in messages}):
        idx = [i for i, m in enumerate(messages) if m["voix"] == v]
        par_voix[v] = wer(
            [normaliser(references[i]) for i in idx],
            [normaliser(transcriptions[i]) for i in idx],
        )
        print(f"   voix {v:<8}: {par_voix[v]:.3f}  ({len(idx)} messages)")
    print(f"Secondes audio facturées : {secondes:.0f}")

    predits, lat_texte = classer_avec_cache(transcriptions)
    etiquette = f"voix__{modele}__{langue or 'auto'}"
    resume = afficher_rapport(f"{modele} ({langue or 'auto'}) -> {MODELE_TEXTE}", vrais, predits)
    resume.update(mesures)

    lat_totale = round((sum(lat_asr) + sum(lat_texte)) / len(messages), 2)
    resume.update(
        etiquette=etiquette,
        canal="voix",
        modele=modele,
        langue=langue or "auto",
        ids=[m["id"] for m in messages],
        vrais=vrais,
        predits=predits,
        textes=transcriptions,
        wer_par_voix=par_voix,
        secondes_audio=round(secondes),
        latence_s=lat_totale,
    )
    print(f"\nLatence moyenne par message : {lat_totale:.2f} s "
          f"(Whisper {sum(lat_asr) / len(messages):.2f} s + classification {sum(lat_texte) / len(messages):.2f} s)")
    enregistrer(resume, etiquette)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage : python evaluer_voix.py [texte | <modele_whisper>] [fr|auto]")
    if sys.argv[1] == "texte":
        evaluer_texte()
    else:
        lang = sys.argv[2] if len(sys.argv) > 2 else "fr"
        lang_param = None if lang == "auto" else lang
        evaluer_voix(sys.argv[1], lang_param)


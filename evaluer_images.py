"""Évaluation du modèle de vision sur les 20 captures d'écran (Étape 5).

Usage :
    python evaluer_images.py
"""
import csv
import json
import time
from pathlib import Path

from groq import RateLimitError
from evaluation import afficher_rapport
from vision import MODELE_VISION, classer_image

PAUSE = 16  # Une image + consigne ~ 2000 tokens ; quota gratuit : 8000 tokens / min


def trouver_dossier_data():
    for p in [Path("atelier3/data"), Path("data"), Path("../atelier3/data"), Path("../data")]:
        if p.exists() and (p / "images.csv").exists():
            return p
    return Path("atelier3/data")


DATA = trouver_dossier_data()


def classer_avec_cache(chemin):
    """Effectue la classification ou renvoie le résultat stocké en cache."""
    fichier_cache = DATA / "classifications" / "images.json"
    cache = json.loads(fichier_cache.read_text(encoding="utf-8")) if fichier_cache.exists() else {}
    cle = f"{MODELE_VISION}|{chemin.name}"

    if cle in cache:
        return cache[cle]["categorie"], cache[cle]["latence_s"]

    for tentative in range(4):
        try:
            debut = time.time()
            categorie = classer_image(chemin)
            latence = round(time.time() - debut, 2)
            break
        except RateLimitError:
            attente = 20 * (tentative + 1)
            print(f"\n  quota atteint, pause de {attente} s")
            time.sleep(attente)
    else:
        raise RuntimeError("quota toujours atteint après 4 tentatives")

    cache[cle] = {"categorie": categorie, "latence_s": latence}
    fichier_cache.parent.mkdir(parents=True, exist_ok=True)
    fichier_cache.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    time.sleep(PAUSE)
    return categorie, latence


def main():
    with open(DATA / "images.csv", encoding="utf-8") as f:
        images = list(csv.DictReader(f))

    vrais, predits, latences = [], [], []
    print(f"Évaluation de {len(images)} captures d'écran avec {MODELE_VISION}...")
    for i, img in enumerate(images, 1):
        chemin_img = DATA / img["fichier"]
        categorie, latence = classer_avec_cache(chemin_img)
        vrais.append(img["categorie"])
        predits.append(categorie)
        latences.append(latence)
        print(f"\r  capture {i}/{len(images)} : {img['id']} -> {categorie} ({latence:.2f}s)", end="", flush=True)
    print()

    resume = afficher_rapport(f"captures -> {MODELE_VISION}", vrais, predits)

    print("\n--- Captures mal classées ---")
    mal_classees = 0
    for img, p in zip(images, predits):
        if p != img["categorie"]:
            mal_classees += 1
            marque = " (ambiguë)" if img.get("ambigu") == "1" else ""
            print(f"   {img['id']}  attendu: {img['categorie']:<12}  prédit: {p}{marque}")
    if mal_classees == 0:
        print("   Aucune erreur !")

    lat_moy = round(sum(latences) / len(latences), 2) if latences else 0.0
    resume.update(
        etiquette=f"image__{MODELE_VISION}",
        canal="image",
        modele=MODELE_VISION,
        langue="-",
        ids=[img["id"] for img in images],
        vrais=vrais,
        predits=predits,
        latence_s=lat_moy,
    )
    print(f"\nLatence moyenne par capture : {lat_moy:.2f} s")

    Path("resultats").mkdir(exist_ok=True)
    nom_modele_clean = MODELE_VISION.replace("/", "_")
    chemin_res = Path("resultats") / f"image__{nom_modele_clean}.json"
    chemin_res.write_text(json.dumps(resume, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Résultats enregistrés dans {chemin_res}")


if __name__ == "__main__":
    main()


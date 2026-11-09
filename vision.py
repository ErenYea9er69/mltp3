"""Classification d'image de bout en bout avec un modèle de vision (Étape 5).

Usage :
    python vision.py atelier3/data/captures/I06.png
"""
import base64
import json
import os
import sys
import time
from pathlib import Path

from groq import Groq, RateLimitError
from evaluation import CATEGORIES

client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
MODELE_VISION = "qwen/qwen3.8-27b"

CONSIGNE_IMAGE = """Un client du service client a envoye cette capture d'ecran pour illustrer son probleme.
Classe-la dans exactement une categorie parmi : facturation, technique, compte, commercial.

Definitions :
- facturation : un montant, une facture, un paiement, un remboursement.
- technique : un service qui ne fonctionne pas comme prevu.
- compte : l'acces au compte ou les donnees personnelles.
- commercial : le choix d'une offre, un abonnement, une resiliation.

Si la capture montre une erreur technique (code d'erreur, page qui ne charge pas,
application qui plante), choisis technique, meme sur une page de paiement ou d'offre.

Reponds uniquement par un objet JSON de la forme {"categorie": "<une des quatre>"}."""


def data_url(chemin):
    """Transforme une image en data URL base64 insérable dans une requête multimodale."""
    chemin = Path(chemin)
    if not chemin.exists():
        raise FileNotFoundError(f"Image introuvable : {chemin}")
    octets = chemin.read_bytes()
    type_mime = "image/png" if chemin.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{type_mime};base64,{base64.b64encode(octets).decode('ascii')}"


def classer_image(chemin, modele=MODELE_VISION, consigne=CONSIGNE_IMAGE):
    """Envoie la capture au modèle de vision et renvoie une catégorie, ou 'invalide'."""
    url = data_url(chemin)

    for tentative in range(4):
        try:
            reponse = client.chat.completions.create(
                model=modele,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": consigne},
                            {"type": "image_url", "image_url": {"url": url}},
                        ],
                    }
                ],
            )
            break
        except RateLimitError:
            attente = 15 * (tentative + 1)
            print(f"  quota vision atteint, pause de {attente} s")
            time.sleep(attente)
    else:
        raise RuntimeError("quota toujours atteint après 4 tentatives")

    brut = reponse.choices[0].message.content
    try:
        data = json.loads(brut)
        categorie = data.get("categorie", "")
    except (json.JSONDecodeError, AttributeError):
        return "invalide"

    categorie = str(categorie).strip().lower()
    return categorie if categorie in CATEGORIES else "invalide"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage : python vision.py <chemin_vers_capture.png>")
    chemin_img = sys.argv[1]
    debut = time.time()
    cat = classer_image(chemin_img)
    print(f"Catégorie prédite : {cat} ({time.time() - debut:.2f} s)")


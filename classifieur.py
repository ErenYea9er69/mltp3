"""Classifieur de tickets textuels via le modèle Qwen sur Groq (hérité de l'Atelier 2)."""
import json
import os
import time

from groq import Groq, RateLimitError
from evaluation import CATEGORIES

client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

MODELE_TEXTE_DEFAUT = "qwen/qwen3.8-27b"

CONSIGNE = """Tu es un agent de tri du service client.
Classe le ticket dans exactement une categorie parmi : facturation, technique, compte, commercial.

Definitions :
- facturation : le ticket porte sur un montant, une facture, un paiement, un remboursement.
- technique : le ticket porte sur un service qui ne fonctionne pas comme prevu.
- compte : le ticket porte sur l'acces au compte ou les donnees personnelles.
- commercial : le ticket porte sur le choix d'une offre, un abonnement, une resiliation.

Si plusieurs categories semblent possibles, retiens l'intention principale du client :
ce qu'il veut obtenir, pas ce qu'il decrit en chemin.

Reponds uniquement par un objet JSON de la forme {"categorie": "<une des quatre>"}."""


def classer(texte, modele=MODELE_TEXTE_DEFAUT, consigne=CONSIGNE, temperature=0.0):
    """Renvoie une catégorie parmi les 4 autorisées, ou 'invalide'."""
    for tentative in range(4):
        try:
            reponse = client.chat.completions.create(
                model=modele,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": consigne},
                    {"role": "user", "content": texte},
                ],
            )
            break
        except RateLimitError:
            attente = 5 * (tentative + 1)
            print(f"  quota atteint, pause de {attente} s")
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
    exemple = "Ma facture de mars comporte un montant anormal de 89 euros."
    print(f"Ticket : {exemple}")
    print(f"Catégorie prédite : {classer(exemple)}")


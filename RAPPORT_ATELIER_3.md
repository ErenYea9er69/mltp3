# Rapport Atelier 3 — La Voix et l'Image
**Machine Learning Appliqué — 3e année DSI**

---

## 1. Vue d'ensemble de l'architecture

Le projet étend le système de classification de tickets initié à l'Atelier 2 en intégrant deux modalités supplémentaires :
1. **Le canal Voix (Pipeline en 2 étapes)** :
   $$\text{Audio (.mp3)} \xrightarrow{\text{Whisper ASR}} \text{Texte transcrit} \xrightarrow{\text{Classifieur Qwen 3.8}} \text{Catégorie}$$
   * Qualité ASR mesurée par le **WER** (*Word Error Rate*).
   * Qualité finale mesurée par le **Macro-F1**.

2. **Le canal Image (Modèle multimodal de bout en bout)** :
   $$\text{Capture d'écran (.png)} \xrightarrow{\text{Vision Qwen 3.8}} \text{Catégorie}$$
   * Qualité mesurée par le **Macro-F1**.

---

## 2. Synthèse des résultats obtenus

| Canal | Modèle | Qualité | Latence moy. | N |
| :--- | :--- | :--- | :--- | :--- |
| **Texte (Référence)** | `qwen/qwen3.8-27b` | **Macro-F1 : 0.959** (Accuracy : 0.958) | 0.49 s | 72 |
| **Voix** | `whisper-large-v3-turbo` + `qwen3.8-27b` | WER brut : ~0.545 / WER norm : ~0.083 | ~1.5 - 2.0 s | 72 |
| **Image** | `qwen/qwen3.8-27b` (Vision) | **Macro-F1 : ~0.90** | ~2.5 s | 20 |

---

## 3. Analyse détaillée par étape

### Étape 1 – Transcription audio (`transcription.py`)
* Implémentation de `transcrire()` avec ouverture binaire (`"rb"`), température à `0.0` et passage conditionnel du paramètre `language`.
* **Rôle du paramètre `language="fr"`** :
  * Sur des messages courts (ex: `T006` *"stop"*), le mode `auto` échoue dans la phase de détection de langue (transcription aberrante : *"All."*).
  * Imposer `fr` contraint le décodeur aux probabilités phonétiques du français et réduit la latence.

### Étape 2 – Normalisation & Calcul du WER (`wer.py`)
* Normalisation : conversion minuscules, `€` $\rightarrow$ `euros`, suppression des tirets/apostrophes et ponctuation résiduelle, réduction des espaces multiples, conservation des accents.
* **Résultat sur la phrase de référence** :
  * WER brut : **0.545** (pénalisé artificiellement par la casse, la ponctuation et le symbole euro).
  * WER normalisé : **0.083** (1 seule vraie substitution d'écoute : *"est"* $\rightarrow$ *"et"* sur 12 mots).

### Étape 4 – Propagation des erreurs (`propagation.py`)
Les tickets sont répartis en 4 quadrants :
* **Conservé** : Juste en texte ET juste en voix.
* **Perdu** : Juste en texte, mais FAUX en voix $\rightarrow$ *Erreur introduite par Whisper*.
* **Gagné** : Faux en texte, juste en voix $\rightarrow$ *Effet aléatoire (bruit fortuit)*.
* **Deux faux** : Faux sur les deux canaux $\rightarrow$ *Erreur intrinsèque au classifieur*.

---

## 4. Réponses aux Questions de conclusion

### Question A : Quel canal donne le meilleur résultat ? Justifiez avec une mesure.
> **Réponse** :  
> Le canal **Texte de référence** donne le meilleur résultat avec un **Macro-F1 de 0.959** et une exactitude (*accuracy*) de **95.8%**.  
> **Justification** : Le texte d'origine ne subit aucune altération phonétique ni compression audio. Les canaux vocaux et visuels introduisent une perte d'information (bruit acoustique, artefacts d'OCR / vision) qui dégrade la prédiction finale.

### Question B : Pour la voix, les erreurs viennent-elles de la transcription ou de la classification ?
> **Réponse** :  
> Les erreurs viennent principalement de la **propagation d'erreurs de la transcription (Whisper)**.  
> **Justification** : Le classifieur textuel est très performant (95.9% F1). Dès lors qu'un ticket correctement classé sur le texte brut est mal classé sur sa version audio (cas du quadrant *"perdu"*), l'erreur provient directement d'une substitution ou d'une omission de mot-clé commise par Whisper (ex. un terme technique confondu ou un montant altéré). Le classifieur ne fait que traiter fidèlement une entrée corrompue.

### Question C : Quelles vérifications faudrait-il faire avant d'utiliser ce système dans un vrai guichet ?
> **Réponse** :  
> 1. **Robustesse acoustique** : Tester sur des enregistrements réels avec bruit de fond (voiture, rue, téléphone dégradé) et des accents régionaux / non natifs variés.
> 2. **Confiance et routage vers l'humain** : Ne pas forcer une décision si le modèle hésite ; introduire un score de confiance et envoyer les tickets incertains à un téléconseiller humain.
> 3. **Confidentialité & RGPD** : Filtrer et caviarder les données sensibles (mots de passe dictés, numéros de cartes bancaires, visages sur les captures d'écran).
> 4. **Tolérance aux pannes et latence** : Mettre en place un mécanisme de repli (*fallback*) en cas de saturation de l'API de transcription ou de vision.


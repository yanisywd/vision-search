"""
Script d'évaluation complète de tous les descripteurs.
Lance ce script une fois depuis le terminal :
    python evaluate_all.py
Il génère/met à jour le fichier resultats_evaluation.csv utilisé par l'app.
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
from PIL import Image

# ── Importer les fonctions de app1 sans déclencher Streamlit ────────────────
import unittest.mock as mock
import types

# On mock streamlit pour que l'import de app1 ne crashe pas
st_mock = types.ModuleType("streamlit")
for attr in ["cache_resource", "cache_data", "session_state", "progress",
             "empty", "success", "error", "warning", "info"]:
    setattr(st_mock, attr, lambda *a, **kw: None)
st_mock.session_state = {}
sys.modules["streamlit"] = st_mock

import importlib.util
spec = importlib.util.spec_from_file_location(
    "app1", os.path.join(os.path.dirname(__file__), "app1.py")
)
app1 = importlib.util.load_from_spec(spec)  # type: ignore
# ─────────────────────────────────────────────────────────────────────────────

BASE_PATH = os.path.join(os.path.dirname(__file__), "BD_images_prepared")
CSV_OUT   = os.path.join(os.path.dirname(__file__), "resultats_evaluation.csv")

DISTANCES = ["euclidienne", "manhattan", "cosinus"]

# Descripteurs à évaluer.
# desc_ann exclu : nécessite un encodeur entraîné passé en paramètre (géré dans l'app).
# desc_cnn_dataset supprimé : c'était un histogramme RGB+HSV, redondant avec ResNet18.
# desc_ann_dataset : ConvAutoencoder entraîné sur le dataset — nécessite que le modèle
#   soit déjà entraîné (conv_autoencoder_dataset.pth présent) avant de lancer ce script.
DESCRIPTEURS = [
    "hist_rgb", "hist_pond_sat", "hist_cumule", "hist_entropie",
    "lbp", "lbp_blocs", "glcm",
    "desc_stat", "desc_stat_complet", "desc_stat_entropy",
    "cds", "dcd", "ccd",
    "desc_forme_sobel", "desc_forme_prewitt", "desc_forme_roberts",
    "hog", "hog_non_pondere", "hog_blocs",
    "desc_cnn", "desc_ann_dataset", "desc_ann_pretrained",
]


def charger_image(chemin):
    try:
        img = Image.open(chemin)
        arr = np.array(img)
        if len(arr.shape) == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        return arr
    except Exception:
        return None


def explorer_base(base):
    structure = {}
    for root, _, files in os.walk(base):
        if root != base:
            classe = os.path.basename(root)
            imgs = [f for f in files if f.lower().endswith(
                ('.jpg', '.jpeg', '.png', '.bmp'))]
            if imgs:
                structure[classe] = imgs
    return structure


def calculer_ap(resultats_tries, classe_requete, top_k=None):
    if top_k:
        resultats_tries = resultats_tries[:top_k]
    nb_corrects = 0
    precisions = []
    for rang, r in enumerate(resultats_tries, 1):
        if r["classe"] == classe_requete:
            nb_corrects += 1
            precisions.append(nb_corrects / rang)
    return float(np.mean(precisions)) if precisions else 0.0


def evaluer(db, desc_type, distance):
    """Retourne (MAP, MAP@10) pour un descripteur + distance."""
    # Extraire tous les descripteurs
    print(f"    extraction des descripteurs...", end=" ", flush=True)
    desc_db = {}
    for i, item in enumerate(db):
        img = charger_image(item["chemin"])
        if img is None:
            continue
        try:
            from app1 import extraire_descripteur_par_type  # type: ignore
            d = extraire_descripteur_par_type(img, desc_type)
        except Exception:
            d = None
        if d is not None:
            desc_db[i] = d

    if len(desc_db) < 2:
        print("aucun descripteur extrait, skip.")
        return None, None

    print(f"{len(desc_db)} ok. calcul distances...", end=" ", flush=True)

    try:
        from app1 import calculer_distance  # type: ignore
    except Exception:
        return None, None

    aps, aps10 = [], []
    keys = list(desc_db.keys())
    for idx_q in keys:
        d_q = desc_db[idx_q]
        classe_q = db[idx_q]["classe"]
        resultats = []
        for idx_db in keys:
            if idx_db == idx_q:
                continue
            dist = calculer_distance(d_q, desc_db[idx_db], distance)
            resultats.append({"classe": db[idx_db]["classe"], "distance": dist})
        resultats_tries = sorted(resultats, key=lambda x: x["distance"])
        aps.append(calculer_ap(resultats_tries, classe_q))
        aps10.append(calculer_ap(resultats_tries, classe_q, top_k=10))

    map_val  = float(np.mean(aps))
    map10_val = float(np.mean(aps10))
    print(f"MAP={map_val:.4f}  MAP@10={map10_val:.4f}")
    return map_val, map10_val


def main():
    print(f"Base : {BASE_PATH}")
    structure = explorer_base(BASE_PATH)
    db = []
    for classe, images in structure.items():
        chemin_classe = os.path.join(BASE_PATH, classe)
        for img_name in images:
            db.append({"chemin": os.path.join(chemin_classe, img_name),
                       "classe": classe})
    print(f"{len(db)} images dans {len(structure)} classes.\n")

    # Charger les résultats existants pour ne pas recalculer
    existing = set()
    if os.path.exists(CSV_OUT):
        df_exist = pd.read_csv(CSV_OUT)
        for _, row in df_exist.iterrows():
            existing.add((row["Descripteur"], row["Distance"]))
        rows = df_exist.to_dict("records")
        print(f"CSV existant : {len(rows)} lignes déjà calculées.\n")
    else:
        rows = []

    total = len(DESCRIPTEURS) * len(DISTANCES)
    done = 0
    for desc in DESCRIPTEURS:
        for dist in DISTANCES:
            done += 1
            if (desc, dist) in existing:
                print(f"[{done}/{total}] {desc} × {dist} → déjà calculé, skip.")
                continue
            print(f"[{done}/{total}] {desc} × {dist}")
            map_val, map10_val = evaluer(db, desc, dist)
            if map_val is not None:
                rows.append({
                    "Descripteur": desc,
                    "Distance": dist,
                    "MAP": map_val,
                    "MAP@10": map10_val,
                })
                # Sauvegarder après chaque calcul (en cas d'interruption)
                pd.DataFrame(rows).to_csv(CSV_OUT, index=False)

    print(f"\nTerminé. {len(rows)} lignes sauvegardées dans {CSV_OUT}")


if __name__ == "__main__":
    main()

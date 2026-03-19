





import streamlit as st
import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import pickle
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurtosis
from scipy.ndimage import label as scipy_label

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as transforms
    from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

st.set_page_config(
    page_title="VisionSearch — CBIR",
    page_icon="🔍",
    layout="wide"
)

BASE_PATH        = os.path.join(os.path.dirname(__file__), "BD_images_prepared")
CACHE_FILE       = os.path.join(os.path.dirname(__file__), "descripteurs_cache.pkl")
CONV_AE_PATH     = os.path.join(os.path.dirname(__file__), "conv_autoencoder_dataset.pth")
SAM_CHECKPOINT   = os.path.join(os.path.dirname(__file__), "sam_vit_b_01ec64.pth")
SAM_MODEL_TYPE   = "vit_b"


def lister_images(chemin_repertoire):
    if os.path.exists(chemin_repertoire):
        images = [f for f in os.listdir(chemin_repertoire) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
        return sorted(images)
    return []


def explorer_base(base):
    structure = {}
    for root, dirs, files in os.walk(base):
        if root != base:
            classe = os.path.basename(root)
            images = lister_images(root)
            structure[classe] = images
    return structure


def charger_image(chemin_image):
    try:
        img = Image.open(chemin_image)
        img_array = np.array(img)
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        return img_array
    except Exception as e:
        print(f"Erreur lors du chargement de {chemin_image}: {e}")
        return None


def convertir_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)


def convertir_gris(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def convertir_lab(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)


def convertir_ycbcr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)


def extraire_histogramme_rgb(image, bins=32):
    # Calcule l'histogramme du canal Rouge : répartition des intensités de 0 à 255 en `bins` intervalles
    hist_r = cv2.calcHist([image], [0], None, [bins], [0, 256])
    # Calcule l'histogramme du canal Vert
    hist_g = cv2.calcHist([image], [1], None, [bins], [0, 256])
    # Calcule l'histogramme du canal Bleu
    hist_b = cv2.calcHist([image], [2], None, [bins], [0, 256])
    # Aplatit en vecteur 1D et normalise par la somme → chaque valeur = proportion de pixels dans ce bin
    hist_r = hist_r.flatten() / hist_r.sum()
    hist_g = hist_g.flatten() / hist_g.sum()
    hist_b = hist_b.flatten() / hist_b.sum()
    # Concatène les 3 histogrammes → vecteur final de taille 3*bins (ex: 96 valeurs pour bins=32)
    return np.concatenate([hist_r, hist_g, hist_b])


def histogramme_pondere_saturation(image, bins=32):
    # Convertit l'image RGB en espace HSV (Teinte, Saturation, Valeur)
    hsv = convertir_hsv(image)
    # Sépare les 3 canaux HSV en composantes individuelles
    h, s, v = cv2.split(hsv)
    # Histogramme de la Teinte (H) pondéré par le masque de Saturation (s) → ignore les pixels peu saturés (quasi-gris)
    hist_h = cv2.calcHist([h], [0], s, [bins], [0, 180])
    # Histogramme de la Saturation (S) : mesure la "vivacité" des couleurs
    hist_s = cv2.calcHist([s], [0], None, [bins], [0, 256])
    # Histogramme de la Valeur/Luminosité (V)
    hist_v = cv2.calcHist([v], [0], None, [bins], [0, 256])
    # Normalise chaque histogramme (+1e-10 évite la division par zéro)
    hist_h = hist_h.flatten() / (hist_h.sum() + 1e-10)
    hist_s = hist_s.flatten() / (hist_s.sum() + 1e-10)
    hist_v = hist_v.flatten() / (hist_v.sum() + 1e-10)
    # Calcule la saturation moyenne globale normalisée entre 0 et 1
    saturation = s.mean() / 255.0
    # Amplifie l'histogramme de teinte selon la saturation moyenne → si l'image est très colorée, la teinte pèse plus
    return np.concatenate([hist_h * (1 + saturation), hist_s, hist_v])


def histogramme_cumule(image, bins=32):
    # Calcule l'histogramme brut pour chaque canal RGB
    hist_r = cv2.calcHist([image], [0], None, [bins], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [bins], [0, 256])
    hist_b = cv2.calcHist([image], [2], None, [bins], [0, 256])
    # np.cumsum transforme l'histogramme en cumulatif : chaque bin = somme de tous les bins précédents
    # Puis division par la somme totale → valeurs entre 0 et 1 (CDF = Cumulative Distribution Function)
    # Avantage : plus robuste aux changements de luminosité qu'un histogramme simple
    hist_r = np.cumsum(hist_r.flatten()) / (hist_r.sum() + 1e-10)
    hist_g = np.cumsum(hist_g.flatten()) / (hist_g.sum() + 1e-10)
    hist_b = np.cumsum(hist_b.flatten()) / (hist_b.sum() + 1e-10)
    # Vecteur final de taille 3*bins
    return np.concatenate([hist_r, hist_g, hist_b])


def histogramme_entropie(image, bins=32):
    # Extrait l'histogramme RGB normalisé par canal (vecteur de taille 3*bins)
    hist = extraire_histogramme_rgb(image, bins=bins)
    # Petit epsilon pour éviter log2(0) qui serait -infini
    eps = 1e-10
    # Calcule l'entropie de Shannon : mesure la complexité/variété des couleurs
    # Plus l'entropie est haute → plus la distribution est uniforme (image variée)
    # Plus elle est basse → peu de couleurs dominantes (image uniforme)
    ent = -np.sum(hist * np.log2(hist + eps))
    # Re-normalise l'histogramme globalement (les 3 canaux ensemble somment à 1)
    hist_norm = hist / (hist.sum() + 1e-10)
    # Ajoute la valeur scalaire d'entropie comme dernière dimension → vecteur de taille 3*bins+1
    return np.concatenate([hist_norm, [ent]])


def calculer_glcm(image, distance=1, angle=0, niveaux_gris=256):
    # Récupère les dimensions de l'image en niveaux de gris
    h, w = image.shape
    # Crée une matrice carrée 256×256 initialisée à zéro
    # glcm[i][j] = nombre de fois que le pixel de valeur i a un voisin de valeur j
    glcm = np.zeros((niveaux_gris, niveaux_gris), dtype=np.float32)
    # Parcourt chaque pixel de l'image
    for i in range(h):
        for j in range(w):
            if angle == 0:
                # Voisin à droite (direction horizontale) : (i, j) → (i, j+distance)
                if j + distance < w:
                    glcm[image[i, j], image[i, j + distance]] += 1
            elif angle == 45:
                # Voisin en haut à droite (diagonale /) : (i, j) → (i-distance, j+distance)
                if i - distance >= 0 and j + distance < w:
                    glcm[image[i, j], image[i - distance, j + distance]] += 1
            elif angle == 90:
                # Voisin en bas (direction verticale) : (i, j) → (i+distance, j)
                if i + distance < h:
                    glcm[image[i, j], image[i + distance, j]] += 1
            elif angle == 135:
                # Voisin en bas à droite (diagonale \) : (i, j) → (i+distance, j+distance)
                if i + distance < h and j + distance < w:
                    glcm[image[i, j], image[i + distance, j + distance]] += 1
    # Normalise la GLCM → chaque case devient une probabilité de co-occurrence
    glcm_norm = glcm / (np.sum(glcm) + 1e-10)
    return glcm_norm


def energie_asm(glcm):
    # Énergie / ASM (Angular Second Moment) : somme des carrés des probabilités de co-occurrence
    # Valeur haute → texture uniforme et répétitive (ex: tissu régulier)
    # Valeur basse → texture chaotique et variée
    return np.sum(glcm ** 2)


def entropie(glcm):
    # Petit epsilon pour éviter log2(0)
    eps = 1e-10
    # Entropie de la GLCM : mesure le désordre de la texture
    # Valeur haute → texture complexe et aléatoire
    # Valeur basse → texture régulière et prévisible
    return -np.sum(glcm * np.log2(glcm + eps))


def contraste(glcm):
    # Dimensions de la GLCM (256×256)
    rows, cols = glcm.shape
    resultat = 0.0
    for i in range(rows):
        for j in range(cols):
            # Pondère chaque co-occurrence par le carré de la différence entre les niveaux de gris
            # → Plus les voisins sont différents, plus le contraste est élevé
            # Valeur haute → grandes variations locales d'intensité (texture contrastée)
            resultat += ((i - j) ** 2) * glcm[i, j]
    return resultat


def inverse_difference_moment(glcm):
    # IDM / Homogénéité locale : pondère les co-occurrences proches de la diagonale plus fortement
    rows, cols = glcm.shape
    resultat = 0.0
    for i in range(rows):
        for j in range(cols):
            # Division par (1 + différence²) → les paires similaires (i≈j) contribuent le plus
            # Valeur haute → pixels voisins très similaires (texture homogène)
            resultat += glcm[i, j] / (1 + (i - j) ** 2)
    return resultat


def dissimilarite(glcm):
    # Dissimilarité : comme le contraste mais pondération linéaire (|i-j|) au lieu de quadratique
    rows, cols = glcm.shape
    resultat = 0.0
    for i in range(rows):
        for j in range(cols):
            # Poids = différence absolue entre les niveaux de gris voisins
            # Valeur haute → les voisins ont des intensités très différentes
            resultat += np.abs(i - j) * glcm[i, j]
    return resultat


def homogeneite(glcm):
    # Homogénéité : mesure à quel point les co-occurrences sont proches de la diagonale principale
    rows, cols = glcm.shape
    resultat = 0.0
    for i in range(rows):
        for j in range(cols):
            # Division par (1 + |i-j|) : décroissance linéaire (moins sévère que IDM)
            # Valeur haute → texture lisse et homogène
            resultat += glcm[i, j] / (1 + np.abs(i - j))
    return resultat


def extraire_descripteurs_texture(image, distance=1, angles=[0, 45, 90, 135]):
    # Convertit en niveaux de gris si l'image est en couleur (GLCM travaille sur 1 canal)
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image
    # Liste pour stocker les 4 GLCM calculées selon 4 directions
    glcms = []
    for angle in angles:
        # Calcule une GLCM pour chaque direction : 0°, 45°, 90°, 135°
        # → capture la texture dans toutes les orientations (descripteur invariant à la rotation si on moyenne)
        glcm = calculer_glcm(img_gray, distance=distance, angle=angle)
        glcms.append(glcm)
    # Calcule chaque mesure de texture sur les 4 GLCM directionnelles
    energie_vals = [energie_asm(g) for g in glcms]       # 4 valeurs d'énergie
    entropie_vals = [entropie(g) for g in glcms]         # 4 valeurs d'entropie
    contraste_vals = [contraste(g) for g in glcms]       # 4 valeurs de contraste
    idm_vals = [inverse_difference_moment(g) for g in glcms]  # 4 valeurs d'IDM
    dissimilarite_vals = [dissimilarite(g) for g in glcms]    # 4 valeurs de dissimilarité
    homogeneite_vals = [homogeneite(g) for g in glcms]   # 4 valeurs d'homogénéité
    # Moyenne des 4 directions pour chaque mesure → 1 scalaire par mesure, invariant à l'orientation
    # Résultat : vecteur compact de 6 valeurs résumant la texture globale de l'image
    descripteurs = np.array([
        np.mean(energie_vals),
        np.mean(entropie_vals),
        np.mean(contraste_vals),
        np.mean(idm_vals),
        np.mean(dissimilarite_vals),
        np.mean(homogeneite_vals)
    ])
    return descripteurs


def calculer_lbp(image, radius=3, n_points=24):
    # Convertit en niveaux de gris (LBP travaille sur 1 canal)
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image
    # Calcule le LBP (Local Binary Pattern) :
    # Pour chaque pixel, compare ses n_points voisins sur un cercle de rayon `radius`
    # → si voisin >= pixel central : bit=1, sinon bit=0 → code binaire décrivant la texture locale
    # method='uniform' : garde uniquement les patterns avec au max 2 transitions 0→1 (plus robuste)
    lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
    # Nombre de bins = n_points + 2 (n_points patterns uniformes + 1 pattern non-uniforme + 1 bord)
    bins = n_points + 2
    # Construit l'histogramme des codes LBP → distribution des micro-textures dans l'image
    # density=True normalise automatiquement l'histogramme (somme ≈ 1)
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins), density=True)
    # Vecteur de taille n_points+2 = 26 valeurs (pour radius=3, n_points=24)
    return hist


def calculer_lbp_blocs(image, n_blocs=4, radius=3, n_points=24):
    # Convertit en niveaux de gris
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image
    # Récupère les dimensions de l'image
    h, w = img_gray.shape
    # Calcule la hauteur et largeur de chaque bloc (division en grille n_blocs × n_blocs)
    bh = max(h // n_blocs, 1)  # hauteur d'un bloc (min 1 pixel)
    bw = max(w // n_blocs, 1)  # largeur d'un bloc (min 1 pixel)
    # Nombre de bins LBP par bloc
    bins = n_points + 2
    descripteur = []  # liste qui va accumuler les histogrammes de chaque bloc
    for i in range(n_blocs):          # ligne de la grille
        for j in range(n_blocs):      # colonne de la grille
            # Calcule les coordonnées du bloc courant dans l'image
            y_start = i * bh
            y_end = min((i + 1) * bh, h)   # min() pour ne pas dépasser les bords
            x_start = j * bw
            x_end = min((j + 1) * bw, w)
            # Extrait le sous-bloc correspondant
            bloc = img_gray[y_start:y_end, x_start:x_end]
            if bloc.size == 0:
                # Si le bloc est vide (cas limite), ajoute des zéros pour garder la taille constante
                descripteur.extend(np.zeros(bins))
                continue
            # Calcule le LBP sur ce bloc uniquement → capture la texture à cet endroit spatial
            lbp = local_binary_pattern(bloc, n_points, radius, method='uniform')
            # Histogramme LBP du bloc, normalisé
            hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins), density=True)
            # Ajoute au descripteur global → préserve l'information spatiale (où est la texture)
            descripteur.extend(hist)
    # Vecteur final : n_blocs² × (n_points+2) = 4×4×26 = 416 valeurs (pour n_blocs=4)
    return np.array(descripteur)


def desc_statistiques(image):
    # Convertit en niveaux de gris pour calculer les stats globales de l'image
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image
    # Calcule 8 statistiques globales sur le canal gris
    stats = [
        np.mean(img_gray),               # Luminosité moyenne (0-255)
        np.median(img_gray),             # Valeur centrale (moins sensible aux extrêmes que la moyenne)
        np.std(img_gray),                # Écart-type : mesure la dispersion → contraste global
        np.min(img_gray),                # Pixel le plus sombre
        np.max(img_gray),                # Pixel le plus lumineux
        np.percentile(img_gray, 25),     # 1er quartile : 25% des pixels sont en dessous
        np.percentile(img_gray, 75),     # 3ème quartile : 75% des pixels sont en dessous
        np.var(img_gray)                 # Variance : dispersion au carré (redondant avec std mais utile)
    ]
    stats_rgb = []
    if len(image.shape) == 3:
        # Pour chaque canal couleur (R, G, B), calcule 2 statistiques supplémentaires
        for c in range(3):
            channel = image[:, :, c]
            stats_rgb.extend([
                np.mean(channel),   # Intensité moyenne du canal (ex: fort rouge → image rougeâtre)
                np.std(channel)     # Variabilité du canal
            ])
    # Vecteur final : 8 stats grises + 3×2 stats RGB = 14 valeurs
    return np.array(stats + stats_rgb)


def desc_statistiques_complet(image):
    # Convertit en niveaux de gris
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image
    # Aplatit en vecteur 1D et convertit en float64 pour précision des calculs statistiques
    flat = img_gray.flatten().astype(np.float64)
    # 10 statistiques sur le canal gris (les 8 de base + skewness + kurtosis)
    stats = [
        np.mean(flat),                   # Moyenne
        np.median(flat),                 # Médiane
        np.std(flat),                    # Écart-type
        np.min(flat),                    # Minimum
        np.max(flat),                    # Maximum
        np.percentile(flat, 25),         # 1er quartile
        np.percentile(flat, 75),         # 3ème quartile
        np.var(flat),                    # Variance
        float(scipy_skew(flat)),         # Asymétrie : >0 → queue à droite, <0 → queue à gauche
        float(scipy_kurtosis(flat))      # Aplatissement : >0 → pic pointu, <0 → distribution plate
    ]
    stats_rgb = []
    if len(image.shape) == 3:
        # Pour chaque canal RGB : 4 statistiques enrichies avec skewness et kurtosis
        for c in range(3):
            channel = image[:, :, c].flatten().astype(np.float64)
            stats_rgb.extend([
                np.mean(channel),              # Intensité moyenne du canal
                np.std(channel),               # Variabilité du canal
                float(scipy_skew(channel)),    # Asymétrie de la distribution du canal
                float(scipy_kurtosis(channel)) # Forme des pics de la distribution du canal
            ])
    # Vecteur final : 10 stats grises + 3×4 stats RGB = 22 valeurs
    return np.array(stats + stats_rgb)


def desc_stat_entropy(image):
    # Récupère les 14 statistiques de base (moyenne, std, min, max, etc.)
    stats = desc_statistiques(image)
    # Convertit en niveaux de gris pour calculer l'entropie sur 1 canal
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image
    # Histogramme sur 256 niveaux (pleine résolution) pour le calcul de l'entropie
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    # Normalise pour obtenir des probabilités
    hist_norm = hist.flatten() / (hist.sum() + 1e-10)
    eps = 1e-10
    # Entropie de Shannon : mesure la complexité de la distribution des intensités
    # Proche de 8 bits (max) → image avec beaucoup de variété de niveaux de gris
    entropie_val = -np.sum(hist_norm * np.log2(hist_norm + eps))
    # Combine les 14 stats + 1 entropie → vecteur de 15 valeurs
    return np.concatenate([stats, [entropie_val]])


def calculer_cds(image):
    # CDS = Color Distribution Statistics : statistiques de distribution sur 2 espaces couleur
    # S'assure que l'image est en RGB (convertit le gris si nécessaire)
    if len(image.shape) != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    stats = []
    # Partie 1 : statistiques sur l'espace RGB (3 canaux × 4 stats = 12 valeurs)
    for c in range(3):
        channel = image[:, :, c].flatten().astype(np.float64)
        stats.extend([
            np.mean(channel),              # Intensité moyenne du canal R, G ou B
            np.std(channel),               # Variabilité du canal
            float(scipy_skew(channel)),    # Asymétrie de la distribution du canal
            float(scipy_kurtosis(channel)) # Aplatissement de la distribution du canal
        ])
    # Convertit en HSV pour capturer la couleur sous un autre angle perceptuel
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Partie 2 : statistiques sur l'espace HSV (3 canaux × 4 stats = 12 valeurs)
    for c in range(3):
        channel = hsv[:, :, c].flatten().astype(np.float64)
        stats.extend([
            np.mean(channel),              # Teinte/Saturation/Valeur moyenne
            np.std(channel),               # Variabilité
            float(scipy_skew(channel)),    # Asymétrie
            float(scipy_kurtosis(channel)) # Aplatissement
        ])
    # Vecteur final : 12 (RGB) + 12 (HSV) = 24 valeurs
    return np.array(stats)


def calculer_dcd(image, k=5):
    # DCD = Dominant Color Descriptor : extrait les k couleurs dominantes via K-Means
    # S'assure que l'image est en RGB
    if len(image.shape) != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # Reformate l'image en liste de pixels : shape (H×W, 3) → chaque ligne est un pixel RGB
    pixels = image.reshape(-1, 3).astype(np.float32)
    # Critère d'arrêt du K-Means : s'arrête si epsilon < 0.2 OU après 100 itérations
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # Applique K-Means sur les pixels pour trouver k couleurs dominantes
    # 10 tentatives avec centres aléatoires → garde le meilleur résultat
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Aplatit les labels d'attribution de cluster (1 label par pixel)
    labels = labels.flatten()
    # Calcule le pourcentage de pixels appartenant à chaque cluster
    percentages = np.array([np.sum(labels == i) / len(labels) for i in range(k)])
    # Trie les clusters par ordre décroissant de pourcentage (couleur la + présente en premier)
    order = np.argsort(-percentages)
    # Normalise les centres (couleurs RGB) entre 0 et 1 + réordonne
    centers_sorted = centers[order].flatten() / 255.0
    # Pourcentages triés (proportion de chaque couleur dominante)
    percentages_sorted = percentages[order]
    # Vecteur final : k×3 (couleurs RGB) + k (proportions) = k*3 + k = 20 valeurs (pour k=5)
    return np.concatenate([centers_sorted, percentages_sorted])


def calculer_ccd(image, n_bins=32, tau=50):
    # CCD = Color Coherence Vector : distingue les pixels "cohérents" (grande zone uniforme)
    # des pixels "incohérents" (pixels isolés ou petites taches) pour chaque niveau de couleur
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image
    # Lissage gaussien pour réduire le bruit avant la quantification
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    # Quantifie les intensités en n_bins niveaux (ex: 256 → 32 niveaux)
    # Chaque pixel reçoit un numéro de bin entre 0 et n_bins-1
    quantized = (img_blur.astype(np.float32) * n_bins / 256.0).astype(np.uint8)
    quantized = np.clip(quantized, 0, n_bins - 1)  # sécurité pour ne pas dépasser les bornes
    # Initialise les accumulateurs pour pixels cohérents et incohérents par bin
    coherent = np.zeros(n_bins, dtype=np.float64)
    incoherent = np.zeros(n_bins, dtype=np.float64)
    for b in range(n_bins):
        # Crée un masque binaire pour le bin courant (pixels avec ce niveau d'intensité)
        mask = (quantized == b).astype(np.uint8)
        total_pixels = np.sum(mask)
        if total_pixels == 0:
            continue  # aucun pixel de ce niveau → on saute
        # Étiquette les composantes connexes : chaque groupe de pixels adjacents reçoit un ID unique
        labeled, num_features = scipy_label(mask)
        for comp_id in range(1, num_features + 1):
            # Compte le nombre de pixels dans cette composante connexe
            comp_size = np.sum(labeled == comp_id)
            if comp_size >= tau:
                # Grande composante (≥ tau pixels) → pixels "cohérents" (font partie d'une grande zone)
                coherent[b] += comp_size
            else:
                # Petite composante (< tau pixels) → pixels "incohérents" (isolés ou petites taches)
                incoherent[b] += comp_size
    # Normalise les deux vecteurs par le total de pixels traités
    total = coherent.sum() + incoherent.sum() + 1e-10
    coherent /= total
    incoherent /= total
    # Vecteur final : n_bins cohérents + n_bins incohérents = 64 valeurs (pour n_bins=32)
    return np.concatenate([coherent, incoherent])


def histogramme_forme(image, gradient_x, gradient_y, bins=32):
    # Calcule la magnitude du gradient : force du contour en chaque pixel
    # magnitude = sqrt(Gx² + Gy²) → 0 = zone uniforme, grand = bord fort
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    # Calcule l'orientation du gradient : direction perpendiculaire au contour
    # arctan2(Gy, Gx) donne l'angle en radians entre -π et +π
    angle = np.arctan2(gradient_y, gradient_x)
    angle = np.degrees(angle)           # Convertit en degrés (-180 à +180)
    angle[angle < 0] += 360            # Ramène les angles négatifs dans [0, 360]
    # Histogramme des magnitudes : répartition des forces de contour (plage 0-100 empirique)
    hist_magnitude = cv2.calcHist([magnitude.astype(np.float32)], [0], None, [bins], [0, 100])
    hist_magnitude = hist_magnitude.flatten() / (hist_magnitude.sum() + 1e-10)  # normalise
    # Histogramme des orientations : répartition des directions de contour (0 à 360°)
    # Fort pic à 90° → beaucoup de contours horizontaux, etc.
    hist_angle = cv2.calcHist([angle.astype(np.float32)], [0], None, [bins], [0, 360])
    hist_angle = hist_angle.flatten() / (hist_angle.sum() + 1e-10)  # normalise
    # Vecteur final : bins magnitudes + bins orientations = 64 valeurs (pour bins=32)
    return np.concatenate([hist_magnitude, hist_angle])


def desc_forme_sobel(image):
    # Convertit en niveaux de gris (les gradients se calculent sur 1 canal)
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image
    # Opérateur Sobel 3×3 : calcule la dérivée selon X (bords verticaux)
    # cv2.CV_64F → résultat en float64 pour garder les valeurs négatives
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    # Opérateur Sobel 3×3 : calcule la dérivée selon Y (bords horizontaux)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    # Construit le descripteur de forme à partir des gradients Sobel
    return histogramme_forme(img_gray, grad_x, grad_y)


def desc_forme_prewitt(image):
    # Convertit en niveaux de gris
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image
    # Noyau Prewitt horizontal : détecte les bords verticaux (moyenne des lignes adjacentes)
    # Différence avec Sobel : pas de pondération centrale → moins de lissage
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    # Noyau Prewitt vertical : détecte les bords horizontaux
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    # Applique la convolution avec chaque noyau
    grad_x = cv2.filter2D(img_gray, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(img_gray, cv2.CV_64F, kernel_y)
    # Construit le descripteur de forme à partir des gradients Prewitt
    return histogramme_forme(img_gray, grad_x, grad_y)


def desc_forme_roberts(image):
    # Convertit en niveaux de gris
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image
    # Noyau Roberts croisé X : détecte les bords diagonaux (45°)
    # Compare le pixel courant avec son voisin en bas à droite
    kernel_x = np.array([[1, 0], [0, -1]])
    # Noyau Roberts croisé Y : détecte les bords diagonaux (135°)
    kernel_y = np.array([[0, 1], [-1, 0]])
    # Applique la convolution avec les noyaux 2×2 de Roberts
    grad_x = cv2.filter2D(img_gray, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(img_gray, cv2.CV_64F, kernel_y)
    # Construit le descripteur de forme à partir des gradients Roberts
    return histogramme_forme(img_gray, grad_x, grad_y)


def calculer_hog(image, cell_size=8, n_bins=9):
    # HOG = Histogram of Oriented Gradients : décrit la forme par la distribution des orientations de contours
    # Convertit en niveaux de gris
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image
    # Redimensionne à 128×128 pour avoir un descripteur de taille fixe quelle que soit l'image source
    img_resized = cv2.resize(img_gray, (128, 128))
    # Calcule les gradients selon X et Y avec Sobel
    grad_x = cv2.Sobel(img_resized, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_resized, cv2.CV_64F, 0, 1, ksize=3)
    # Magnitude du gradient : force du contour en chaque pixel
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    # Orientation du gradient : ramène dans [0°, 180°] (gradient non signé)
    direction = np.degrees(np.arctan2(grad_y, grad_x))
    direction[direction < 0] += 180.0  # angles négatifs → équivalent positif
    h, w = img_resized.shape
    # Nombre de cellules dans chaque dimension (128/8 = 16 cellules)
    n_cells_y = h // cell_size
    n_cells_x = w // cell_size
    # Initialise le tableau d'histogrammes : 1 histogramme de 9 bins par cellule
    cell_hists = np.zeros((n_cells_y, n_cells_x, n_bins))
    # Largeur angulaire d'un bin = 180°/9 = 20° par bin
    bin_width = 180.0 / n_bins
    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            # Coordonnées pixel du coin supérieur gauche de cette cellule
            y0 = cy * cell_size
            x0 = cx * cell_size
            # Extrait la magnitude et la direction pour les pixels de cette cellule (8×8 pixels)
            mag_cell = magnitude[y0:y0 + cell_size, x0:x0 + cell_size]
            dir_cell = direction[y0:y0 + cell_size, x0:x0 + cell_size]
            for i in range(cell_size):
                for j in range(cell_size):
                    ang = dir_cell[i, j]       # orientation du pixel (i,j) de la cellule
                    mag_val = mag_cell[i, j]   # magnitude (poids) du pixel
                    # Détermine dans quel bin angulaire tombe ce pixel
                    bin_idx = int(ang / bin_width)
                    if bin_idx >= n_bins:
                        bin_idx = n_bins - 1   # sécurité pour le dernier bin
                    # Ajoute la magnitude comme vote dans le bin correspondant
                    # → pondération par gradient : les contours forts comptent plus
                    cell_hists[cy, cx, bin_idx] += mag_val
    # Normalisation par blocs (2×2 cellules) : rend le descripteur invariant aux changements de luminosité
    block_size = 2  # un bloc = 2×2 cellules
    descripteur = []
    for by in range(n_cells_y - block_size + 1):  # fenêtre glissante sur les lignes
        for bx in range(n_cells_x - block_size + 1):  # fenêtre glissante sur les colonnes
            # Regroupe les histogrammes des 4 cellules du bloc en 1 vecteur (2×2×9 = 36 valeurs)
            block = cell_hists[by:by + block_size, bx:bx + block_size, :].flatten()
            # Normalise L2 le bloc : divise par sa norme → invariant à l'éclairage local
            norm = np.sqrt(np.sum(block**2) + 1e-10)
            block = block / norm
            descripteur.extend(block)
    # Taille finale : (n_cells-1)² × block_size² × n_bins = 15×15×4×9 = 8100 valeurs
    return np.array(descripteur)


def calculer_hog_non_pondere(image, cell_size=8, n_bins=9):
    # HOG non pondéré : identique au HOG classique SAUF que chaque pixel vote avec +1
    # au lieu de sa magnitude → tous les contours comptent pareil, peu importe leur force
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image
    img_resized = cv2.resize(img_gray, (128, 128))
    grad_x = cv2.Sobel(img_resized, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_resized, cv2.CV_64F, 0, 1, ksize=3)
    # Calcule uniquement la direction (pas la magnitude, on ne s'en sert pas ici)
    direction = np.degrees(np.arctan2(grad_y, grad_x))
    direction[direction < 0] += 180.0  # ramène dans [0°, 180°]
    h, w = img_resized.shape
    n_cells_y = h // cell_size  # nombre de cellules en hauteur
    n_cells_x = w // cell_size  # nombre de cellules en largeur
    cell_hists = np.zeros((n_cells_y, n_cells_x, n_bins))
    bin_width = 180.0 / n_bins  # largeur d'un bin angulaire (20° pour 9 bins)
    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            y0 = cy * cell_size
            x0 = cx * cell_size
            dir_cell = direction[y0:y0 + cell_size, x0:x0 + cell_size]
            for i in range(cell_size):
                for j in range(cell_size):
                    ang = dir_cell[i, j]
                    bin_idx = int(ang / bin_width)
                    if bin_idx >= n_bins:
                        bin_idx = n_bins - 1
                    # Vote binaire : +1 au lieu de la magnitude → comptage simple des orientations
                    cell_hists[cy, cx, bin_idx] += 1
    # Normalisation par blocs identique au HOG classique
    block_size = 2
    descripteur = []
    for by in range(n_cells_y - block_size + 1):
        for bx in range(n_cells_x - block_size + 1):
            block = cell_hists[by:by + block_size, bx:bx + block_size, :].flatten()
            norm = np.sqrt(np.sum(block**2) + 1e-10)
            block = block / norm
            descripteur.extend(block)
    return np.array(descripteur)


def calculer_hog_blocs(image, n_blocs=4, cell_size=8, n_bins=9):
    # HOG par blocs spatiaux : divise d'abord l'image en n_blocs×n_blocs régions
    # puis calcule un HOG indépendant sur chaque région → préserve l'information spatiale
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image
    # Taille cible = n_blocs × cell_size × 2 → garantit que chaque bloc a un nombre entier de cellules
    target_size = n_blocs * cell_size * 2
    img_resized = cv2.resize(img_gray, (target_size, target_size))
    # Taille d'un bloc spatial en pixels
    bloc_h = target_size // n_blocs
    bloc_w = target_size // n_blocs
    descripteur = []
    bin_width = 180.0 / n_bins  # largeur d'un bin angulaire
    for bi in range(n_blocs):        # parcourt les lignes de la grille
        for bj in range(n_blocs):   # parcourt les colonnes de la grille
            # Extrait le sous-bloc spatial courant
            bloc = img_resized[bi * bloc_h:(bi + 1) * bloc_h, bj * bloc_w:(bj + 1) * bloc_w]
            # Calcule les gradients Sobel sur ce bloc uniquement
            gx = cv2.Sobel(bloc, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(bloc, cv2.CV_64F, 0, 1, ksize=3)
            # Magnitude et direction des gradients dans ce bloc
            mag = np.sqrt(gx**2 + gy**2)
            direction = np.degrees(np.arctan2(gy, gx))
            direction[direction < 0] += 180.0  # ramène dans [0°, 180°]
            # Construit l'histogramme d'orientations pondéré par la magnitude pour ce bloc
            hist = np.zeros(n_bins)
            for i in range(bloc.shape[0]):
                for j in range(bloc.shape[1]):
                    bin_idx = int(direction[i, j] / bin_width)
                    if bin_idx >= n_bins:
                        bin_idx = n_bins - 1
                    hist[bin_idx] += mag[i, j]  # vote pondéré par la magnitude
            # Normalise L2 l'histogramme de ce bloc
            norm = np.sqrt(np.sum(hist**2) + 1e-10)
            hist = hist / norm
            # Ajoute au descripteur global (concaténation des blocs)
            descripteur.extend(hist)
    # Vecteur final : n_blocs² × n_bins = 4×4×9 = 144 valeurs (pour n_blocs=4)
    return np.array(descripteur)


# ===== CNN (ResNet18 PyTorch) =====

@st.cache_resource  # garde le modèle en mémoire entre les requêtes Streamlit (ne recharge pas à chaque clic)
def get_cnn_model():
    # Retourne None immédiatement si PyTorch n'est pas installé
    if not TORCH_AVAILABLE:
        return None
    try:
        try:
            # Charge ResNet18 pré-entraîné sur ImageNet (1000 classes, 1.2M images)
            # weights='IMAGENET1K_V1' → syntaxe moderne torchvision ≥ 0.13
            model = models.resnet18(weights='IMAGENET1K_V1')
        except TypeError:
            # Fallback pour les anciennes versions de torchvision qui n'acceptent pas `weights`
            model = models.resnet18(pretrained=True)
        # Supprime la dernière couche (classificateur fc de 1000 neurones)
        # → garde uniquement le feature extractor (512 neurones de sortie = vecteur de features)
        model = nn.Sequential(*list(model.children())[:-1])
        # Passe en mode inférence : désactive Dropout et BatchNorm en mode entraînement
        model.eval()
        return model
    except Exception:
        return None  # échec silencieux si le modèle ne peut pas être chargé


def desc_cnn(image):
    # Récupère le modèle ResNet18 (chargé une seule fois grâce au cache)
    model = get_cnn_model()
    if model is None:
        return None  # PyTorch indisponible ou modèle non chargé
    # Pipeline de transformation de l'image pour correspondre aux attentes de ResNet18
    transform = transforms.Compose([
        transforms.ToPILImage(),                          # numpy array → PIL Image
        transforms.Resize((224, 224)),                    # ResNet18 attend du 224×224
        transforms.ToTensor(),                            # PIL → tensor float32 en [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # normalisation ImageNet : soustrait la moyenne
                             std=[0.229, 0.224, 0.225])  # normalisation ImageNet : divise par l'écart-type
    ])
    # Applique les transformations et ajoute une dimension batch : (3,224,224) → (1,3,224,224)
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():  # désactive le calcul de gradient (on fait seulement de l'inférence)
        features = model(img_tensor)  # passe l'image dans ResNet18 → shape (1, 512, 1, 1)
    # squeeze() supprime les dimensions de taille 1 → vecteur de 512 valeurs
    return features.squeeze().numpy()


# ===== ANN Autoencoder (PyTorch) =====

class Autoencoder(nn.Module):
    # Autoencodeur : réseau qui apprend à compresser puis reconstruire l'image
    # Objectif : la couche "bottleneck" (goulot d'étranglement) force le réseau à
    # apprendre une représentation compacte et informative de l'image
    def __init__(self, input_dim=12288, bottleneck_dim=128):
        super(Autoencoder, self).__init__()
        # ENCODEUR : compresse progressivement l'image vers une représentation compacte
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),   # couche 1 : réduit de input_dim → 512 neurones
            nn.ReLU(),                   # activation non-linéaire : évite la saturation
            nn.Linear(512, 256),         # couche 2 : réduit 512 → 256 neurones
            nn.ReLU(),
            nn.Linear(256, bottleneck_dim),  # couche 3 : compression finale → 128 valeurs
            nn.ReLU()                    # le vecteur de 128 valeurs = descripteur de l'image
        )
        # DÉCODEUR : reconstruit l'image depuis la représentation compacte
        # (utilisé uniquement pendant l'entraînement, pas pour la recherche)
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 256),  # décompresse 128 → 256
            nn.ReLU(),
            nn.Linear(256, 512),             # décompresse 256 → 512
            nn.ReLU(),
            nn.Linear(512, input_dim),       # reconstruit la taille originale
            nn.Sigmoid()                     # ramène les valeurs entre 0 et 1 (pixels normalisés)
        )

    def forward(self, x):
        # Passe avant complet : encode puis décode → utilisé pendant l'entraînement
        return self.decoder(self.encoder(x))

    def encode(self, x):
        # Encode uniquement → retourne le vecteur de 128 valeurs (le descripteur)
        with torch.no_grad():  # pas de gradient calculé (inférence seulement)
            return self.encoder(x)


def train_ann_autoencoder(db):
    # Entraîne l'autoencodeur sur toutes les images de la base de données
    if not TORCH_AVAILABLE:
        return None
    X = []
    for item in db:
        img = charger_image(item['chemin'])  # charge chaque image de la base
        if img is not None:
            img_resized = cv2.resize(img, (64, 64))  # redimensionne à 64×64 pour taille fixe
            # Aplatit (64×64×3 = 12288 valeurs) et normalise entre 0 et 1
            X.append(img_resized.flatten().astype(np.float32) / 255.0)
    if len(X) == 0:
        return None  # aucune image chargeable → abandon
    X = np.array(X)                              # shape : (N_images, 12288)
    X_tensor = torch.FloatTensor(X)              # convertit en tenseur PyTorch
    # Crée un dataset où input = output = image (autoencodeur non supervisé)
    dataset = torch.utils.data.TensorDataset(X_tensor, X_tensor)
    # DataLoader : fournit des mini-batchs de 16 images mélangées aléatoirement
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    # Instancie l'autoencodeur : entrée = 12288 pixels aplatis, bottleneck = 128 valeurs
    model = Autoencoder(input_dim=64 * 64 * 3, bottleneck_dim=128)
    # Optimiseur Adam : adapte le taux d'apprentissage automatiquement
    optimizer = torch.optim.Adam(model.parameters())
    # Critère de perte : MSE → minimise la différence pixel à pixel entre image originale et reconstruite
    criterion = nn.MSELoss()
    model.train()  # active le mode entraînement (Dropout actif si présent)
    for epoch in range(30):  # 30 passages complets sur le dataset
        for batch_x, batch_y in loader:
            optimizer.zero_grad()        # remet les gradients à zéro avant chaque batch
            output = model(batch_x)      # passe avant : encode + décode les images du batch
            loss = criterion(output, batch_y)  # calcule l'erreur de reconstruction
            loss.backward()              # rétropropagation : calcule les gradients
            optimizer.step()             # mise à jour des poids selon les gradients
    model.eval()  # passe en mode inférence pour la suite
    return model


def get_ann_encoder(db):
    if 'ann_encoder' not in st.session_state:
        with st.spinner("Entraînement de l'autoencodeur ANN..."):
            encoder = train_ann_autoencoder(db)
            st.session_state.ann_encoder = encoder
    return st.session_state.get('ann_encoder', None)


# ===== desc_cnn_dataset — SUPPRIMÉ =====
# Ce descripteur a été retiré car il n'était PAS un vrai CNN entraîné sur le dataset :
# il calculait uniquement une concaténation d'histogrammes RGB + HSV (384 valeurs),
# ce qui est redondant avec desc_cnn (ResNet18) qui existe déjà et est bien plus puissant.
# Pour le rapport : desc_cnn_dataset = histogramme RGB+HSV classique, aucun apprentissage.


# ===== CHOIX D'ARCHITECTURE : Pourquoi des autoencoders et pas un CNN supervisé ? =====
#
# Notre dataset contient 200 images réparties en 40 classes, soit seulement 5 images par classe.
#
# Option envisagée : entraîner un CNN classifieur supervisé (from scratch) sur nos 40 classes.
# Pourquoi on ne l'a PAS fait :
#   - Un CNN supervisé nécessite des centaines à milliers d'images PAR classe pour apprendre.
#   - Avec 5 images/classe, le réseau mémoriserait les exemples sans rien généraliser
#     (overfitting total) → les descripteurs obtenus n'auraient aucune valeur pour la recherche.
#   - Le fine-tuning d'un ResNet pré-entraîné aurait été possible mais redondant :
#     desc_cnn (ResNet18) fait déjà exactement ça, et bien mieux.
#
# Ce qu'on a fait à la place : deux autoencoders entraînés en NON-SUPERVISÉ.
#   - Pas de labels nécessaires → l'objectif est uniquement de reconstruire l'image.
#   - Le bottleneck (128 dimensions) force le réseau à apprendre une représentation compacte.
#   - Avec 200 images, c'est faisable : le réseau apprend les structures visuelles du dataset
#     sans avoir besoin de beaucoup d'exemples par classe.
#   - desc_ann  : autoencodeur FC (couches linéaires) — référence simple
#   - desc_ann_dataset : autoencodeur convolutif — exploite la structure spatiale des images,
#     plus adapté aux images que le FC, entraîné from scratch sur nos 200 images.


# ===== ANN Autoencodeur CONVOLUTIF entraîné sur notre dataset =====

class ConvAutoencoder(nn.Module):
    """
    Autoencodeur convolutif entraîné sur notre dataset (200 images, 40 classes).
    Encodeur : 3 blocs Conv+ReLU+MaxPool → flatten → Linear → vecteur latent 128 dims.
    Décodeur : Linear → reshape → 3 blocs ConvTranspose → image reconstruite 64×64×3.
    Seul l'encodeur est utilisé en inférence pour extraire le descripteur.
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        # Encodeur convolutif : exploite la structure spatiale locale des images
        self.enc_conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # → 32×32×16
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # → 16×16×32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # →  8×8×64
        )
        self.enc_fc = nn.Linear(8 * 8 * 64, latent_dim)  # compression finale → 128 valeurs

        # Décodeur (utilisé uniquement pendant l'entraînement)
        self.dec_fc = nn.Linear(latent_dim, 8 * 8 * 64)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(),  # → 16×16×32
            nn.ConvTranspose2d(32, 16, 2, stride=2), nn.ReLU(),  # → 32×32×16
            nn.ConvTranspose2d(16,  3, 2, stride=2), nn.Sigmoid(),  # → 64×64×3
        )

    def encode(self, x):
        # x : (B, 3, 64, 64) → vecteur (B, 128)
        h = self.enc_conv(x)
        h = h.view(h.size(0), -1)
        return self.enc_fc(h)

    def forward(self, x):
        z = self.encode(x)
        h = self.dec_fc(z).view(-1, 64, 8, 8)
        return self.dec_conv(h)


def train_conv_autoencoder_dataset(db, epochs=50, progress_callback=None):
    """Entraîne le ConvAutoencoder sur toutes les images du dataset et sauvegarde le modèle."""
    if not TORCH_AVAILABLE:
        return None
    X = []
    for item in db:
        img = charger_image(item['chemin'])
        if img is not None:
            img_r = cv2.resize(img, (64, 64)).astype(np.float32) / 255.0
            X.append(np.transpose(img_r, (2, 0, 1)))  # HWC → CHW
    if not X:
        return None
    X_tensor = torch.FloatTensor(np.array(X))  # (N, 3, 64, 64)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tensor, X_tensor),
        batch_size=16, shuffle=True
    )
    model = ConvAutoencoder(latent_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        if progress_callback:
            progress_callback((epoch + 1) / epochs)
    model.eval()
    torch.save(model.state_dict(), CONV_AE_PATH)
    return model


@st.cache_resource
def load_conv_autoencoder_cached():
    """Charge le ConvAutoencoder depuis le disque (None si pas encore entraîné)."""
    if not TORCH_AVAILABLE or not os.path.exists(CONV_AE_PATH):
        return None
    try:
        model = ConvAutoencoder(latent_dim=128)
        model.load_state_dict(torch.load(CONV_AE_PATH, map_location='cpu'))
        model.eval()
        return model
    except Exception:
        return None


if TORCH_AVAILABLE:

    # ===== POURQUOI CES MODÈLES NE SONT PAS ENTRAÎNÉS SUR NOTRE DATASET =====
    #
    # SegNet, UNet et PSPNet sont des réseaux de SEGMENTATION SÉMANTIQUE.
    # Pour les entraîner, il faut pour chaque image une annotation pixel par pixel :
    # un masque où chaque pixel porte un label de classe (ex: pixel (x,y) = "chat").
    #
    # Notre dataset contient uniquement des labels au niveau de l'image entière
    # (ex: "cette image appartient à la classe Chats") — pas de masques de segmentation.
    #
    # Il serait donc impossible de calculer la loss de segmentation (cross-entropy pixel
    # par pixel) sans ces masques annotés. Annoter 200 images pixel par pixel
    # nécessiterait des heures de travail manuel (outils : LabelMe, CVAT...).
    #
    # Ces architectures sont donc présentées avec :
    #   - SegNet   : encodeur initialisé avec les poids VGG16 (ImageNet), décodeur aléatoire
    #   - UNet     : poids aléatoires (architecture démontrée)
    #   - PSPNet   : poids aléatoires (architecture démontrée)
    #
    # L'objectif est de montrer la compréhension et l'implémentation de ces architectures,
    # pas de produire une segmentation précise sur notre dataset.

    # ===== ARCHITECTURE SEGNET =====
    # Particularité SegNet : le décodeur utilise les indices de max-pooling sauvegardés
    # par l'encodeur (MaxUnpool2d), ce qui permet une reconstruction précise des frontières.
    # L'encodeur est basé sur VGG16 (5 blocs Conv+BN+ReLU suivis d'un MaxPool).
    # Le décodeur est le miroir exact de l'encodeur, mais remplace MaxPool par MaxUnpool
    # en réutilisant les indices sauvegardés — ainsi les frontières des objets sont
    # reconstruites pixel-par-pixel avec une grande précision.
    class SegNet(nn.Module):
        """
        SegNet — Architecture encodeur-décodeur pour la segmentation sémantique.
        Référence : Badrinarayanan et al., 2017.
        Particularité : MaxUnpooling avec indices sauvegardés par l'encodeur.
        """
        def __init__(self, num_classes=21):
            super().__init__()

            # ── Encodeur : 5 blocs Conv+BN+ReLU×2 + MaxPool(return_indices=True) ──
            # Bloc 1 : 3 → 64 → 64, sortie 128×128
            self.enc1 = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            )
            self.pool1 = nn.MaxPool2d(2, stride=2, return_indices=True)

            # Bloc 2 : 64 → 128 → 128, sortie 64×64
            self.enc2 = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            )
            self.pool2 = nn.MaxPool2d(2, stride=2, return_indices=True)

            # Bloc 3 : 128 → 256 → 256, sortie 32×32
            self.enc3 = nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            )
            self.pool3 = nn.MaxPool2d(2, stride=2, return_indices=True)

            # Bloc 4 : 256 → 512 → 512, sortie 16×16
            self.enc4 = nn.Sequential(
                nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            )
            self.pool4 = nn.MaxPool2d(2, stride=2, return_indices=True)

            # Bloc 5 : 512 → 512 → 512, sortie 8×8
            self.enc5 = nn.Sequential(
                nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            )
            self.pool5 = nn.MaxPool2d(2, stride=2, return_indices=True)

            # ── Décodeur : 5 blocs MaxUnpool + Conv+BN+ReLU×2 (ordre inversé) ──
            # MaxUnpool2d utilise les indices sauvegardés pour replacer les maxima
            # aux bonnes positions → reconstruction précise des frontières d'objets
            self.unpool5 = nn.MaxUnpool2d(2, stride=2)
            self.dec5 = nn.Sequential(
                nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            )

            self.unpool4 = nn.MaxUnpool2d(2, stride=2)
            self.dec4 = nn.Sequential(
                nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            )

            self.unpool3 = nn.MaxUnpool2d(2, stride=2)
            self.dec3 = nn.Sequential(
                nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            )

            self.unpool2 = nn.MaxUnpool2d(2, stride=2)
            self.dec2 = nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            )

            self.unpool1 = nn.MaxUnpool2d(2, stride=2)
            self.dec1 = nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            )

            # Couche de classification finale : 64 → num_classes
            self.classifier = nn.Conv2d(64, num_classes, 1)

            # Initialisation des poids encodeur depuis VGG16 pré-entraîné
            self._init_vgg16_weights()

        def _init_vgg16_weights(self):
            """Charge les poids des couches convolutives VGG16 dans l'encodeur SegNet."""
            try:
                vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
                vgg_features = list(vgg16.features.children())
                # Correspondance VGG16 → blocs SegNet
                # VGG16 features : [Conv,ReLU,Conv,ReLU,MaxPool, ...] × 5
                vgg_conv_blocks = [
                    [0, 2],    # bloc 1 : couches 0,2 de vgg16.features
                    [5, 7],    # bloc 2 : couches 5,7
                    [10, 12],  # bloc 3 : couches 10,12
                    [17, 19],  # bloc 4 : couches 17,19
                    [24, 26],  # bloc 5 : couches 24,26
                ]
                seg_encs = [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5]
                for seg_enc, vgg_idxs in zip(seg_encs, vgg_conv_blocks):
                    seg_convs = [m for m in seg_enc if isinstance(m, nn.Conv2d)]
                    for seg_conv, vgg_idx in zip(seg_convs, vgg_idxs):
                        vgg_layer = vgg_features[vgg_idx]
                        if isinstance(vgg_layer, nn.Conv2d):
                            if seg_conv.weight.shape == vgg_layer.weight.shape:
                                seg_conv.weight.data.copy_(vgg_layer.weight.data)
                                if vgg_layer.bias is not None and seg_conv.bias is not None:
                                    seg_conv.bias.data.copy_(vgg_layer.bias.data)
            except Exception:
                pass  # Si VGG16 non disponible, on continue avec l'initialisation par défaut

        def forward(self, x):
            # ── Encodeur (passage avant avec sauvegarde des indices de pooling) ──
            x1 = self.enc1(x)
            x1p, idx1 = self.pool1(x1)

            x2 = self.enc2(x1p)
            x2p, idx2 = self.pool2(x2)

            x3 = self.enc3(x2p)
            x3p, idx3 = self.pool3(x3)

            x4 = self.enc4(x3p)
            x4p, idx4 = self.pool4(x4)

            x5 = self.enc5(x4p)
            x5p, idx5 = self.pool5(x5)

            # ── Décodeur (unpooling avec indices sauvegardés → reconstruction précise) ──
            d5 = self.unpool5(x5p, idx5, output_size=x5.size())
            d5 = self.dec5(d5)

            d4 = self.unpool4(d5, idx4, output_size=x4.size())
            d4 = self.dec4(d4)

            d3 = self.unpool3(d4, idx3, output_size=x3.size())
            d3 = self.dec3(d3)

            d2 = self.unpool2(d3, idx2, output_size=x2.size())
            d2 = self.dec2(d2)

            d1 = self.unpool1(d2, idx1, output_size=x1.size())
            d1 = self.dec1(d1)

            return self.classifier(d1)

    @st.cache_resource
    def get_segnet_model():
        """Instancie et retourne SegNet en mode évaluation (poids VGG16 encodeur)."""
        if not TORCH_AVAILABLE:
            return None
        try:
            model = SegNet(num_classes=21)
            model.eval()
            return model
        except Exception:
            return None

    def segmentation_segnet(image):
        """
        Segmentation sémantique avec SegNet.
        Prétraitement : redimensionnement 256×256, normalisation ImageNet.
        Retourne (mask_color, classes_detectees).
        """
        model = get_segnet_model()
        if model is None:
            return None, []

        # Prétraitement : resize 256×256 + normalisation ImageNet
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(image).unsqueeze(0)  # (1, 3, 256, 256)

        with torch.no_grad():
            output = model(input_tensor)  # (1, 21, 256, 256)

        # Classe prédite par pixel (argmax sur la dimension classes)
        mask_labels = output.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Redimensionner au format original de l'image
        mask_labels = cv2.resize(mask_labels, (image.shape[1], image.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        # Coloriser le masque avec la palette PASCAL VOC
        mask_color = PASCAL_VOC_COLORS[mask_labels]

        # Identifier les classes présentes
        classes_uniques = np.unique(mask_labels)
        classes_detectees = [PASCAL_VOC_CLASSES[c] for c in classes_uniques
                             if c > 0 and c < len(PASCAL_VOC_CLASSES)]

        return mask_color, classes_detectees


    # ===== ARCHITECTURE UNET =====
    # Particularité UNet : connexions de saut (skip connections) entre encodeur et décodeur
    # — concaténation des feature maps à chaque niveau.
    # Cette architecture a été conçue pour la segmentation médicale (Ronneberger et al., 2015).
    # Les skip connections permettent de combiner les features de haut niveau (sémantique)
    # avec les features de bas niveau (détails fins, contours) → excellente précision des bords.
    class UNet(nn.Module):
        """
        UNet — Architecture encodeur-décodeur avec connexions de saut.
        Référence : Ronneberger et al., 2015 (segmentation médicale).
        Particularité : skip connections par concaténation des feature maps.
        """
        def __init__(self, num_classes=21):
            super().__init__()

            # ── Encodeur : 4 blocs (Conv+ReLU)×2 + MaxPool ──
            # Les feature maps sont sauvegardées pour les skip connections
            self.enc1 = self._bloc(3, 64)       # 64 canaux
            self.enc2 = self._bloc(64, 128)     # 128 canaux
            self.enc3 = self._bloc(128, 256)    # 256 canaux
            self.enc4 = self._bloc(256, 512)    # 512 canaux
            self.pool = nn.MaxPool2d(2)

            # ── Goulot d'étranglement (bottleneck) : 512 → 1024 ──
            self.bottleneck = self._bloc(512, 1024)

            # ── Décodeur : 4 blocs ConvTranspose2d(stride=2) + concat skip + (Conv+ReLU)×2 ──
            # La concaténation double le nombre de canaux d'entrée de chaque décodeur
            self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
            self.dec4 = self._bloc(1024, 512)   # 512+512 en entrée (après concat)

            self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
            self.dec3 = self._bloc(512, 256)    # 256+256 en entrée

            self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.dec2 = self._bloc(256, 128)    # 128+128 en entrée

            self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dec1 = self._bloc(128, 64)     # 64+64 en entrée

            # Couche de classification finale 1×1 : 64 → num_classes
            self.final_conv = nn.Conv2d(64, num_classes, 1)

        @staticmethod
        def _bloc(in_channels, out_channels):
            """Bloc Conv+ReLU × 2 (bloc de base UNet)."""
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            # ── Encodeur (sauvegarde des feature maps pour les skip connections) ──
            e1 = self.enc1(x)          # feature map niveau 1 → skip vers déc1
            e2 = self.enc2(self.pool(e1))   # feature map niveau 2 → skip vers déc2
            e3 = self.enc3(self.pool(e2))   # feature map niveau 3 → skip vers déc3
            e4 = self.enc4(self.pool(e3))   # feature map niveau 4 → skip vers déc4

            # ── Goulot d'étranglement ──
            b = self.bottleneck(self.pool(e4))

            # ── Décodeur avec skip connections (concaténation) ──
            # La concaténation fusionne les features sémantiques (haut niveau)
            # avec les features spatiales (bas niveau) → meilleure localisation
            d4 = self.upconv4(b)
            d4 = torch.cat([d4, e4], dim=1)   # skip connection niveau 4
            d4 = self.dec4(d4)

            d3 = self.upconv3(d4)
            d3 = torch.cat([d3, e3], dim=1)   # skip connection niveau 3
            d3 = self.dec3(d3)

            d2 = self.upconv2(d3)
            d2 = torch.cat([d2, e2], dim=1)   # skip connection niveau 2
            d2 = self.dec2(d2)

            d1 = self.upconv1(d2)
            d1 = torch.cat([d1, e1], dim=1)   # skip connection niveau 1
            d1 = self.dec1(d1)

            return self.final_conv(d1)

    @st.cache_resource
    def get_unet_model():
        """Instancie et retourne UNet en mode évaluation."""
        if not TORCH_AVAILABLE:
            return None
        try:
            model = UNet(num_classes=21)
            model.eval()
            return model
        except Exception:
            return None

    def segmentation_unet(image):
        """
        Segmentation sémantique avec UNet.
        Prétraitement : redimensionnement 256×256, normalisation ImageNet.
        Retourne (mask_color, classes_detectees).
        """
        model = get_unet_model()
        if model is None:
            return None, []

        # Prétraitement : resize 256×256 + normalisation ImageNet
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(image).unsqueeze(0)  # (1, 3, 256, 256)

        with torch.no_grad():
            output = model(input_tensor)  # (1, 21, 256, 256)

        # Classe prédite par pixel
        mask_labels = output.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Redimensionner au format original
        mask_labels = cv2.resize(mask_labels, (image.shape[1], image.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        # Coloriser avec la palette PASCAL VOC
        mask_color = PASCAL_VOC_COLORS[mask_labels]

        # Classes détectées (on ignore l'arrière-plan = classe 0)
        classes_uniques = np.unique(mask_labels)
        classes_detectees = [PASCAL_VOC_CLASSES[c] for c in classes_uniques
                             if c > 0 and c < len(PASCAL_VOC_CLASSES)]

        return mask_color, classes_detectees


    # ===== ARCHITECTURE PSPNET (réduite) =====
    # Particularité PSPNet : module de pooling pyramidal — capture le contexte à plusieurs
    # échelles (1×1, 2×2, 3×3, 6×6).
    # PSPNet (Pyramid Scene Parsing Network) utilise un backbone convolutif pour extraire
    # les features, puis un module PPM (Pyramid Pooling Module) qui effectue un pooling
    # à 4 résolutions différentes pour capturer le contexte local ET global.
    # Les features multi-échelles sont concaténées puis classifiées.
    class PyramidPoolingModule(nn.Module):
        """
        Module de Pooling Pyramidal (PPM) de PSPNet.
        Effectue un pooling adaptatif à 4 échelles : 1×1, 2×2, 3×3, 6×6.
        Chaque échelle capture un niveau de contexte différent.
        """
        def __init__(self, in_channels=256, pool_channels=64):
            super().__init__()
            # 4 branches de pooling à différentes échelles
            self.scales = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size),   # pooling vers taille fixe
                    nn.Conv2d(in_channels, pool_channels, 1),  # réduction de dimension
                    nn.ReLU(inplace=True),
                )
                for output_size in [1, 2, 3, 6]  # 4 résolutions pyramidales
            ])

        def forward(self, x):
            h, w = x.shape[2], x.shape[3]
            features = [x]  # feature map originale
            for scale in self.scales:
                # Pool → Conv → Upsample vers la taille originale
                pooled = scale(x)
                # Remonter à la résolution de x pour la concaténation
                upsampled = nn.functional.interpolate(
                    pooled, size=(h, w), mode='bilinear', align_corners=True
                )
                features.append(upsampled)
            # Concatène : feature originale + 4 contextes multi-échelles
            return torch.cat(features, dim=1)  # 256 + 4×64 = 512 canaux

    class PSPNet(nn.Module):
        """
        PSPNet réduit — Pyramid Scene Parsing Network.
        Référence : Zhao et al., 2017.
        Particularité : module de pooling pyramidal pour le contexte multi-échelles.
        Backbone simplifié (3 blocs conv) au lieu de ResNet complet.
        """
        def __init__(self, num_classes=21):
            super().__init__()

            # ── Backbone simplifié : 3 blocs Conv+BN+ReLU ──
            # (Simplifié par rapport au ResNet original de PSPNet)
            self.backbone = nn.Sequential(
                # Bloc 1 : 3 → 64, résolution originale
                nn.Conv2d(3, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                # Bloc 2 : 64 → 128, stride=2 → résolution /2
                nn.Conv2d(64, 128, 3, padding=1, stride=2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                # Bloc 3 : 128 → 256, stride=2 → résolution /4
                nn.Conv2d(128, 256, 3, padding=1, stride=2),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )

            # ── Module de Pooling Pyramidal ──
            # 4 échelles : 1×1 (contexte global), 2×2, 3×3, 6×6 (contexte local)
            self.ppm = PyramidPoolingModule(in_channels=256, pool_channels=64)

            # ── Tête de classification ──
            # Entrée : 256 (backbone) + 4×64 (PPM) = 512 canaux
            self.head = nn.Sequential(
                nn.Conv2d(256 + 4 * 64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, num_classes, 1),  # classification finale
            )

            self.num_classes = num_classes

        def forward(self, x):
            input_size = (x.shape[2], x.shape[3])

            # ── Backbone : extraction des features (résolution /4) ──
            features = self.backbone(x)  # (B, 256, H/4, W/4)

            # ── Module PPM : contexte multi-échelles ──
            ppm_out = self.ppm(features)  # (B, 512, H/4, W/4)

            # ── Tête de classification ──
            logits = self.head(ppm_out)   # (B, num_classes, H/4, W/4)

            # Upsample vers la taille originale de l'entrée
            output = nn.functional.interpolate(
                logits, size=input_size, mode='bilinear', align_corners=True
            )
            return output  # (B, num_classes, H, W)

    @st.cache_resource
    def get_pspnet_model():
        """Instancie et retourne PSPNet en mode évaluation."""
        if not TORCH_AVAILABLE:
            return None
        try:
            model = PSPNet(num_classes=21)
            model.eval()
            return model
        except Exception:
            return None

    def segmentation_pspnet(image):
        """
        Segmentation sémantique avec PSPNet (réduit).
        Prétraitement : redimensionnement 256×256, normalisation ImageNet.
        Retourne (mask_color, classes_detectees).
        """
        model = get_pspnet_model()
        if model is None:
            return None, []

        # Prétraitement : resize 256×256 + normalisation ImageNet
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(image).unsqueeze(0)  # (1, 3, 256, 256)

        with torch.no_grad():
            output = model(input_tensor)  # (1, 21, 256, 256)

        # Classe prédite par pixel
        mask_labels = output.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Redimensionner au format original
        mask_labels = cv2.resize(mask_labels, (image.shape[1], image.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        # Coloriser avec la palette PASCAL VOC
        mask_color = PASCAL_VOC_COLORS[mask_labels]

        # Classes détectées (on ignore l'arrière-plan = classe 0)
        classes_uniques = np.unique(mask_labels)
        classes_detectees = [PASCAL_VOC_CLASSES[c] for c in classes_uniques
                             if c > 0 and c < len(PASCAL_VOC_CLASSES)]

        return mask_color, classes_detectees


def desc_ann_dataset(image):
    """
    Descripteur ANN entraîné sur notre dataset.
    Utilise l'encodeur du ConvAutoencoder entraîné sur nos 200 images.
    Retourne un vecteur de 128 valeurs appris de façon non-supervisée.
    Retourne None si le modèle n'a pas encore été entraîné.
    """
    model = load_conv_autoencoder_cached()
    if model is None:
        return None
    img = cv2.resize(image, (64, 64)).astype(np.float32) / 255.0
    x = torch.FloatTensor(np.transpose(img, (2, 0, 1))).unsqueeze(0)  # (1, 3, 64, 64)
    with torch.no_grad():
        z = model.encode(x)
    return z.squeeze().numpy()  # vecteur de 128 valeurs

def desc_ann(image, encoder=None):
    # Extrait le descripteur via l'autoencodeur entraîné (encodeur uniquement)
    if encoder is None:
        return None  # autoencodeur non entraîné → impossible d'extraire un descripteur
    # Redimensionne à 64×64 (taille utilisée lors de l'entraînement)
    img = cv2.resize(image, (64, 64))
    # Aplatit et normalise : 64×64×3 = 12288 valeurs entre 0 et 1
    x = img.flatten().astype(np.float32) / 255.0
    # Ajoute la dimension batch : (12288,) → (1, 12288) pour l'encodeur
    x_tensor = torch.FloatTensor(x).unsqueeze(0)
    # Passe dans l'encodeur : compresse (1, 12288) → (1, 128)
    features = encoder.encode(x_tensor)
    # squeeze() enlève la dimension batch → vecteur de 128 valeurs (le descripteur appris)
    return features.squeeze().numpy()

# ===== ANN Pretrained (MobileNetV2) =====

@st.cache_resource  # garde le modèle en mémoire entre les requêtes (évite de le recharger à chaque fois)
def get_ann_pretrained_model():
    if not TORCH_AVAILABLE:
        return None
    try:
        try:
            # Charge MobileNetV2 pré-entraîné sur ImageNet
            # MobileNetV2 = architecture légère avec dépthwise separable convolutions → plus rapide que ResNet
            model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        except TypeError:
            # Fallback pour les anciennes versions de torchvision
            model = models.mobilenet_v2(pretrained=True)
        # Remplace le classificateur final par une couche identité (ne fait rien)
        # → retourne directement les features avant la classification (1280 valeurs)
        model.classifier = nn.Identity()
        model.eval()  # mode inférence : désactive les couches de régularisation (Dropout, BatchNorm entraînement)
        return model
    except Exception:
        return None


def desc_ann_pretrained(image):
    # Extrait un descripteur de 1280 valeurs via MobileNetV2 pré-entraîné sur ImageNet
    model = get_ann_pretrained_model()
    if model is None:
        return None  # PyTorch indisponible ou modèle non chargé
    # Pipeline de prétraitement identique à celui utilisé lors de l'entraînement de MobileNetV2
    transform = transforms.Compose([
        transforms.ToPILImage(),                          # numpy array → PIL Image
        transforms.Resize((224, 224)),                    # MobileNetV2 attend du 224×224
        transforms.ToTensor(),                            # PIL → tensor float32 en [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # normalisation ImageNet
                             std=[0.229, 0.224, 0.225])
    ])
    # Ajoute la dimension batch : (3,224,224) → (1,3,224,224)
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():  # inférence sans calcul de gradient
        features = model(img_tensor)  # shape : (1, 1280) après remplacement du classificateur
    # squeeze() → vecteur de 1280 valeurs (features visuelles apprises sur ImageNet)
    return features.squeeze().numpy()
# ===== Fin CNN / ANN =====


def extraire_descripteurs_complets(image):
    hist_rgb = extraire_histogramme_rgb(image)
    desc_texture = extraire_descripteurs_texture(image)
    return np.concatenate([hist_rgb, desc_texture])


DESCRIPTEUR_LABELS = {
    'hist_rgb': 'Histogramme RGB',
    'hist_pond_sat': 'Histogramme pondéré saturation',
    'hist_cumule': 'Histogramme cumulé',
    'hist_entropie': 'Histogramme entropie',
    'lbp': 'LBP',
    'lbp_blocs': 'LBP par blocs',
    'glcm': 'GLCM',
    'desc_stat': 'Statistiques de base',
    'desc_stat_complet': 'Statistiques complètes (skew+kurt)',
    'desc_stat_entropy': 'Statistiques + Entropie',
    'cds': 'CDS (Color Distribution Stats)',
    'dcd': 'DCD (Dominant Color Descriptor)',
    'ccd': 'CCD (Color Coherence Vector)',
    'desc_forme_sobel': 'Forme Sobel',
    'desc_forme_prewitt': 'Forme Prewitt',
    'desc_forme_roberts': 'Forme Roberts',
    'hog': 'HOG pondéré (classique)',
    'hog_non_pondere': 'HOG non pondéré',
    'hog_blocs': 'HOG par blocs',
    'desc_cnn': 'CNN (ResNet18)',
    'desc_ann': 'ANN (Autoencoder FC)',
    'desc_ann_dataset': 'ANN Conv (Entraîné sur notre dataset)',
    'desc_ann_pretrained': 'ANN Pretrained (MobileNetV2)',
}

DESCRIPTEUR_KEYS = list(DESCRIPTEUR_LABELS.keys())


def extraire_descripteur_par_type(image, type_descripteur, encoder_ann=None):
    # Dispatcher central : redirige vers la bonne fonction d'extraction selon le descripteur choisi
    # Reçoit toujours une image numpy RGB et retourne un vecteur numpy 1D
    if type_descripteur == 'hist_rgb':
        return extraire_histogramme_rgb(image)          # → 96 valeurs (3 canaux × 32 bins)
    elif type_descripteur == 'hist_pond_sat':
        return histogramme_pondere_saturation(image)    # → 96 valeurs (HSV pondéré par saturation)
    elif type_descripteur == 'hist_cumule':
        return histogramme_cumule(image)                # → 96 valeurs (CDF par canal RGB)
    elif type_descripteur == 'hist_entropie':
        return histogramme_entropie(image)              # → 97 valeurs (96 hist + 1 entropie)
    elif type_descripteur == 'lbp':
        return calculer_lbp(image)                      # → 26 valeurs (LBP global)
    elif type_descripteur == 'lbp_blocs':
        return calculer_lbp_blocs(image)                # → 416 valeurs (LBP spatial 4×4 blocs)
    elif type_descripteur == 'glcm':
        return extraire_descripteurs_texture(image)     # → 6 valeurs (énergie, entropie, contraste, IDM, dissim., homog.)
    elif type_descripteur == 'desc_stat':
        return desc_statistiques(image)                 # → 14 valeurs (8 stats gris + 3×2 stats RGB)
    elif type_descripteur == 'desc_stat_complet':
        return desc_statistiques_complet(image)         # → 22 valeurs (10 stats gris + 3×4 stats RGB avec skew/kurt)
    elif type_descripteur == 'desc_stat_entropy':
        return desc_stat_entropy(image)                 # → 15 valeurs (14 stats + 1 entropie)
    elif type_descripteur == 'cds':
        return calculer_cds(image)                      # → 24 valeurs (4 stats × 3 canaux RGB + 4 stats × 3 canaux HSV)
    elif type_descripteur == 'dcd':
        return calculer_dcd(image)                      # → 20 valeurs (5 couleurs dominantes × 3 RGB + 5 proportions)
    elif type_descripteur == 'ccd':
        return calculer_ccd(image)                      # → 64 valeurs (32 cohérents + 32 incohérents)
    elif type_descripteur == 'desc_forme_sobel':
        return desc_forme_sobel(image)                  # → 64 valeurs (32 magnitudes + 32 angles Sobel)
    elif type_descripteur == 'desc_forme_prewitt':
        return desc_forme_prewitt(image)                # → 64 valeurs (32 magnitudes + 32 angles Prewitt)
    elif type_descripteur == 'desc_forme_roberts':
        return desc_forme_roberts(image)                # → 64 valeurs (32 magnitudes + 32 angles Roberts)
    elif type_descripteur == 'hog':
        return calculer_hog(image)                      # → ~8100 valeurs (HOG pondéré magnitude, 128×128)
    elif type_descripteur == 'hog_non_pondere':
        return calculer_hog_non_pondere(image)          # → ~8100 valeurs (HOG vote binaire)
    elif type_descripteur == 'hog_blocs':
        return calculer_hog_blocs(image)                # → 144 valeurs (HOG 4×4 blocs spatiaux)
    elif type_descripteur == 'desc_cnn':
        return desc_cnn(image)                          # → 512 valeurs (ResNet18 features)
    elif type_descripteur == 'desc_ann':
        return desc_ann(image, encoder=encoder_ann)     # → 128 valeurs (autoencodeur bottleneck)
    elif type_descripteur == 'desc_ann_pretrained':
        return desc_ann_pretrained(image)               # → 1280 valeurs (MobileNetV2 features)
    elif type_descripteur == 'desc_ann_dataset':
        return desc_ann_dataset(image)                  # → 128 valeurs (ConvAutoencoder entraîné sur notre dataset)
    else:
        # Fallback : histogramme RGB si le type est inconnu
        return extraire_histogramme_rgb(image)


def calculer_distance_euclidienne(vec1, vec2):
    # Distance euclidienne (L2) : racine carrée de la somme des différences au carré
    # Sensible à la magnitude des valeurs → bon pour des descripteurs normalisés
    # Interprétation géométrique : distance en ligne droite dans l'espace de descripteurs
    return np.sqrt(np.sum((vec1 - vec2) ** 2))


def calculer_distance_manhattan(vec1, vec2):
    # Distance de Manhattan (L1) : somme des valeurs absolues des différences
    # Plus robuste aux valeurs aberrantes que la distance euclidienne (pas de carré)
    # Interprétation : distance si on ne peut se déplacer que horizontalement/verticalement
    return np.sum(np.abs(vec1 - vec2))


def calculer_distance_cosinus(vec1, vec2):
    # Produit scalaire : mesure la corrélation directionnelle entre les deux vecteurs
    dot_product = np.dot(vec1, vec2)
    # Normes L2 de chaque vecteur
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        # Un vecteur nul n'a pas d'orientation → distance maximale (1.0 = complètement différent)
        return 1.0
    # Similarité cosinus = cos(angle entre les vecteurs), entre -1 et 1
    # On retourne 1 - similarité pour avoir une DISTANCE (0 = identique, 2 = opposés)
    # Avantage : insensible à la magnitude → compare uniquement les proportions des valeurs
    return 1 - (dot_product / (norm1 * norm2))


def calculer_distance(vec1, vec2, methode='euclidienne'):
    # Dispatcher : sélectionne la bonne fonction de distance selon le paramètre `methode`
    if methode == 'euclidienne':
        return calculer_distance_euclidienne(vec1, vec2)
    elif methode == 'manhattan':
        return calculer_distance_manhattan(vec1, vec2)
    elif methode == 'cosinus':
        return calculer_distance_cosinus(vec1, vec2)
    # Fallback : euclidienne par défaut si méthode inconnue
    return calculer_distance_euclidienne(vec1, vec2)


def calculer_ap(resultats_tries, classe_requete):
    # AP = Average Precision : mesure la qualité du classement pour une requête donnée
    # Récompense les descripteurs qui placent les images pertinentes en tête de liste
    images_correctes_trouvees = 0  # compteur d'images de la bonne classe trouvées jusqu'ici
    precisions = []                # liste des valeurs de précision à chaque image correcte trouvée
    for rang, resultat in enumerate(resultats_tries, 1):  # rang commence à 1
        if resultat['classe'] == classe_requete:  # cette image est de la même classe que la requête
            images_correctes_trouvees += 1
            # Précision au rang `rang` : proportion d'images correctes parmi les `rang` premiers résultats
            precision = images_correctes_trouvees / rang
            precisions.append(precision)
    if not precisions:
        return 0.0  # aucune image correcte trouvée → AP = 0
    # AP = moyenne des précisions aux rangs où une image correcte a été trouvée
    # Plus les bonnes images sont haut dans le classement → AP proche de 1
    return np.mean(precisions)


def calculer_map(resultats_par_image):
    # MAP = Mean Average Precision : moyenne des AP sur toutes les images requêtes testées
    # C'est la métrique standard pour évaluer un système de recherche d'images
    # Proche de 1 → le système classe bien les images similaires en tête pour toutes les requêtes
    aps = [res['ap'] for res in resultats_par_image]  # récupère l'AP de chaque requête
    return np.mean(aps)  # moyenne globale


def traiter_image(chemin_image, classe):
    try:
        img = Image.open(chemin_image)
        img_array = np.array(img)
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        descripteurs = extraire_descripteurs_complets(img_array)
        return {
            'chemin': chemin_image,
            'classe': classe,
            'nom': os.path.basename(chemin_image),
            'descripteurs': descripteurs
        }
    except Exception as e:
        st.error(f"Erreur lors du traitement de {chemin_image}: {e}")
        return None


def indexer_base(force_reindex=False):
    cache_file_exists = os.path.exists(CACHE_FILE)
    if not force_reindex and cache_file_exists:
        try:
            with open(CACHE_FILE, 'rb') as f:
                cache_data = pickle.load(f)
            st.success(f"Cache chargé ! {len(cache_data)} images indexées.")
            return cache_data
        except:
            st.warning("Erreur lors du chargement du cache. Réindexation en cours...")
    structure = explorer_base(BASE_PATH)
    descripteurs_db = []
    total_images = sum(len(images) for images in structure.values())
    progress_bar = st.progress(0)
    status_text = st.empty()
    idx = 0
    for classe, images in structure.items():
        chemin_classe = os.path.join(BASE_PATH, classe)
        for img in images:
            chemin_img = os.path.join(chemin_classe, img)
            resultat = traiter_image(chemin_img, classe)
            if resultat:
                descripteurs_db.append(resultat)
            idx += 1
            progress = idx / total_images
            progress_bar.progress(progress)
            status_text.text(f"Indexation : {idx}/{total_images} images ({progress*100:.1f}%)")
    progress_bar.empty()
    status_text.empty()
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(descripteurs_db, f)
    st.success(f"Indexation terminée ! {len(descripteurs_db)} images indexées et sauvegardées dans le cache.")
    return descripteurs_db


def get_descripteurs_cached(db, type_descripteur, encoder_ann=None):
    cache_key = f'desc_cache_{type_descripteur}'
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    descripteurs = {}
    progress_bar = st.progress(0)
    status = st.empty()
    for i, item in enumerate(db):
        img = charger_image(item['chemin'])
        if img is not None:
            desc = extraire_descripteur_par_type(img, type_descripteur, encoder_ann=encoder_ann)
            if desc is not None:
                descripteurs[i] = desc
        progress_bar.progress((i + 1) / len(db))
        status.text(f"Extraction des descripteurs ({type_descripteur}): {i+1}/{len(db)}")
    progress_bar.empty()
    status.empty()
    st.session_state[cache_key] = descripteurs
    return descripteurs


def rechercher_images_avec_descripteur(image_requete, db, type_descripteur='hist_pond_sat', methode_distance='euclidienne', encoder_ann=None):
    desc_requete = extraire_descripteur_par_type(image_requete, type_descripteur, encoder_ann=encoder_ann)
    if desc_requete is None:
        st.error("Impossible d'extraire le descripteur de l'image requête.")
        return []
    descripteurs_db = get_descripteurs_cached(db, type_descripteur, encoder_ann=encoder_ann)
    resultats = []
    for i, desc_db in descripteurs_db.items():
        if desc_db is None:
            continue
        distance = calculer_distance(desc_requete, desc_db, methode_distance)
        resultats.append({
            'chemin': db[i]['chemin'],
            'classe': db[i]['classe'],
            'nom': db[i]['nom'],
            'distance': distance
        })
    resultats_tries = sorted(resultats, key=lambda x: x['distance'])
    return resultats_tries


def afficher_resultats(resultats, k=10):
    st.subheader(f"Top {k} images les plus similaires")
    cols = st.columns(5)
    for i in range(min(k, len(resultats))):
        idx = i % 5
        col = cols[idx]
        item = resultats[i]
        img = Image.open(item['chemin'])
        with col:
            st.image(img, caption=f"#{i+1}: {item['nom']}\nClasse: {item['classe']}\nDistance: {item['distance']:.4f}", use_container_width=True)
    if len(resultats) > k:
        st.write(f"... et {len(resultats) - k} autres images")


def binarisation_manuelle(image, seuil):
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image
    _, binary = cv2.threshold(img_gray, seuil, 255, cv2.THRESH_BINARY)
    return binary


def binarisation_automatique(image):
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def binarisation_mediane(image):
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image
    seuil = np.median(img_gray)
    _, binary = cv2.threshold(img_gray, seuil, 255, cv2.THRESH_BINARY)
    return binary


def binarisation_min_max(image):
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image
    seuil = (np.min(img_gray) + np.max(img_gray)) / 2
    _, binary = cv2.threshold(img_gray, int(seuil), 255, cv2.THRESH_BINARY)
    return binary


def binarisation_p_tile(image, p_tile=50):
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image
    seuil = np.percentile(img_gray, p_tile)
    _, binary = cv2.threshold(img_gray, int(seuil), 255, cv2.THRESH_BINARY)
    return binary


def binarisation_locale_moyenne(image, taille_bloc=15):
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image
    binary = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, taille_bloc, 2)
    return binary


def binarisation_locale_mediane(image, taille_bloc=15):
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image
    h, w = img_gray.shape
    binary = np.zeros_like(img_gray)
    pad = taille_bloc // 2
    img_padded = np.pad(img_gray, pad, mode='reflect')
    for i in range(h):
        for j in range(w):
            bloc = img_padded[i:i + taille_bloc, j:j + taille_bloc]
            seuil = np.median(bloc)
            binary[i, j] = 255 if img_gray[i, j] >= seuil else 0
    return binary


def binarisation_locale_min_max(image, taille_bloc=15):
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image
    h, w = img_gray.shape
    binary = np.zeros_like(img_gray)
    pad = taille_bloc // 2
    img_padded = np.pad(img_gray, pad, mode='reflect')
    for i in range(h):
        for j in range(w):
            bloc = img_padded[i:i + taille_bloc, j:j + taille_bloc]
            seuil = (np.min(bloc) + np.max(bloc)) / 2
            binary[i, j] = 255 if img_gray[i, j] >= seuil else 0
    return binary


def binarisation_niblack(image, k=-0.2, taille_bloc=15):
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image
    h, w = img_gray.shape
    binary = np.zeros_like(img_gray)
    pad = taille_bloc // 2
    img_padded = np.pad(img_gray, pad, mode='reflect')
    for i in range(h):
        for j in range(w):
            bloc = img_padded[i:i + taille_bloc, j:j + taille_bloc]
            mean = np.mean(bloc)
            std = np.std(bloc)
            seuil = mean + k * std
            binary[i, j] = 255 if img_gray[i, j] >= seuil else 0
    return binary


def binarisation_sauvola(image, k=0.34, r=128, taille_bloc=15):
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image
    h, w = img_gray.shape
    binary = np.zeros_like(img_gray)
    pad = taille_bloc // 2
    img_padded = np.pad(img_gray, pad, mode='reflect')
    for i in range(h):
        for j in range(w):
            bloc = img_padded[i:i + taille_bloc, j:j + taille_bloc]
            mean = np.mean(bloc)
            std = np.std(bloc)
            seuil = mean * (1 + k * (std / r - 1))
            binary[i, j] = 255 if img_gray[i, j] >= seuil else 0
    return binary


def binarisation_wolf(image, k=0.5, r=128, taille_bloc=15):
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image
    h, w = img_gray.shape
    binary = np.zeros_like(img_gray)
    pad = taille_bloc // 2
    img_padded = np.pad(img_gray, pad, mode='reflect')
    for i in range(h):
        for j in range(w):
            bloc = img_padded[i:i + taille_bloc, j:j + taille_bloc]
            mean = np.mean(bloc)
            std = np.std(bloc)
            min_val = np.min(bloc)
            seuil = mean + k * (std / r) * (mean - min_val)
            binary[i, j] = 255 if img_gray[i, j] >= seuil else 0
    return binary


def kmeans_segmentation(image, k=3, espace='rgb'):
    if espace == 'hsv':
        if len(image.shape) == 3:
            image_conv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        else:
            image_conv = image
    else:
        if len(image.shape) == 3:
            image_conv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_conv = image
    pixels = image_conv.reshape(-1, 3).astype(np.float32) if len(image_conv.shape) == 3 else image_conv.reshape(-1, 1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()
    segmented = centers[labels]
    segmented = segmented.reshape(image_conv.shape)
    if espace == 'hsv' and len(segmented.shape) == 3:
        segmented = cv2.cvtColor(segmented, cv2.COLOR_HSV2RGB)
    elif espace != 'hsv' and len(segmented.shape) == 3:
        segmented = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
    return segmented





# ===== Segmentation par Deep Learning (DeepLabV3) =====

# Palette de couleurs pour les 21 classes PASCAL VOC
PASCAL_VOC_CLASSES = [
    "arrière-plan", "avion", "vélo", "oiseau", "bateau",
    "bouteille", "bus", "voiture", "chat", "chaise",
    "vache", "table", "chien", "cheval", "moto",
    "personne", "plante", "mouton", "canapé", "train",
    "écran TV"
]

PASCAL_VOC_COLORS = np.array([
    [0, 0, 0],       # arrière-plan
    [128, 0, 0],     # avion
    [0, 128, 0],     # vélo
    [128, 128, 0],   # oiseau
    [0, 0, 128],     # bateau
    [128, 0, 128],   # bouteille
    [0, 128, 128],   # bus
    [128, 128, 128], # voiture
    [64, 0, 0],      # chat
    [192, 0, 0],     # chaise
    [64, 128, 0],    # vache
    [192, 128, 0],   # table
    [64, 0, 128],    # chien
    [192, 0, 128],   # cheval
    [64, 128, 128],  # moto
    [192, 128, 128], # personne
    [0, 64, 0],      # plante
    [128, 64, 0],    # mouton
    [0, 192, 0],     # canapé
    [128, 192, 0],   # train
    [0, 64, 128],    # écran TV
], dtype=np.uint8)


@st.cache_resource
def get_deeplab_model():
    """Charge le modèle DeepLabV3 pré-entraîné sur PASCAL VOC (21 classes)."""
    if not TORCH_AVAILABLE:
        return None
    try:
        weights = DeepLabV3_ResNet101_Weights.DEFAULT
        model = deeplabv3_resnet101(weights=weights)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle DeepLab: {e}")
        return None


def segmentation_deeplab(image):
    """
    Segmentation sémantique avec DeepLabV3-ResNet101.
    Retourne :
      - mask_color : image RGB où chaque pixel est coloré selon sa classe
      - mask_labels : matrice 2D avec l'index de classe par pixel
      - classes_detectees : liste des noms de classes détectées
    """
    model = get_deeplab_model()
    if model is None:
        return None, None, []

    # Prétraitement identique à ImageNet
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(520),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)['out'][0]  # shape: [21, H, W]

    # Classe prédite par pixel
    mask_labels = output.argmax(0).cpu().numpy().astype(np.uint8)  # [H, W]

    # Redimensionner au format original
    mask_labels = cv2.resize(mask_labels, (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

    # Coloriser le masque
    mask_color = PASCAL_VOC_COLORS[mask_labels]  # [H, W, 3]

    # Classes détectées
    classes_uniques = np.unique(mask_labels)
    classes_detectees = [PASCAL_VOC_CLASSES[c] for c in classes_uniques if c < len(PASCAL_VOC_CLASSES)]

    return mask_color, mask_labels, classes_detectees


def superposer_segmentation(image, mask_color, alpha=0.5):
    """Superpose le masque coloré sur l'image originale avec transparence."""
    overlay = cv2.addWeighted(image, 1 - alpha, mask_color, alpha, 0)
    return overlay


# ===== Segment Anything Model (SAM) — Meta AI =====
# SAM est un modèle de segmentation universel : il segmente n'importe quel objet
# dans n'importe quelle image, sans avoir besoin de classes prédéfinies.
# Contrairement à DeepLabV3 (limité à 21 classes PASCAL VOC), SAM fonctionne
# sur tous types d'images (fruits, paysages, animaux, objets...).
# On utilise le mode automatique : SAM détecte et segmente tous les objets visibles.
# Modèle utilisé : ViT-B (le plus léger, ~375 MB) — bon équilibre vitesse/qualité.

@st.cache_resource
def get_sam_model():
    """Charge le modèle SAM depuis le checkpoint local."""
    if not SAM_AVAILABLE:
        return None
    if not os.path.exists(SAM_CHECKPOINT):
        return None
    try:
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
        sam.eval()
        return sam
    except Exception:
        return None


@st.cache_resource
def get_resnet_classifier():
    """Charge ResNet50 pré-entraîné ImageNet pour classifier les crops SAM."""
    if not TORCH_AVAILABLE:
        return None, None
    try:
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model.eval()
        categories = weights.meta['categories']  # 1000 classes ImageNet
        return model, categories
    except Exception:
        return None, None


def classifier_crop_sam(crop, model, categories):
    """
    Classifie un crop d'image avec ResNet50 (ImageNet).
    Retourne le label prédit et le score de confiance.
    """
    if crop is None or crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
        return None, 0.0
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    try:
        x = preprocess(crop).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
        probs = torch.softmax(logits, dim=1)
        top_prob, top_idx = probs.topk(1)
        label = categories[top_idx.item()]
        # Simplifier le label ImageNet (prendre le premier mot avant la virgule)
        label = label.split(',')[0].strip()
        return label, round(top_prob.item(), 2)
    except Exception:
        return None, 0.0


def segmentation_sam(image, points_per_side=32, avec_labels=False):
    """
    Segmentation automatique avec SAM (Segment Anything Model).
    Si avec_labels=True, classifie chaque objet détecté via ResNet50 (ImageNet).
    Retourne :
      - fig_overlay : figure matplotlib avec masques + labels optionnels
      - mask_color  : image des masques seuls
      - nb_masques  : nombre d'objets détectés
      - labels_info : liste de (label, confiance, bbox) pour chaque masque labellisé
    """
    sam = get_sam_model()
    if sam is None:
        return None, None, 0, []

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        min_mask_region_area=100,
    )

    masks = mask_generator.generate(image)
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)

    # Construire l'image de masques colorés
    mask_color = np.zeros_like(image)
    rng = np.random.default_rng(42)
    colors = []
    for mask_data in masks:
        color = rng.integers(60, 230, size=3, dtype=np.uint8)
        mask_color[mask_data['segmentation']] = color
        colors.append(color)

    overlay = cv2.addWeighted(image, 0.4, mask_color, 0.6, 0)

    # Classifier chaque objet si demandé
    labels_info = []
    if avec_labels and TORCH_AVAILABLE:
        classifier, categories = get_resnet_classifier()
        if classifier is not None:
            # Seulement les N plus grands masques pour ne pas surcharger
            top_masks = [m for m in masks if m['area'] > image.shape[0] * image.shape[1] * 0.005]
            top_masks = top_masks[:15]  # max 15 labels
            for i, mask_data in enumerate(top_masks):
                x, y, w, h = [int(v) for v in mask_data['bbox']]
                crop = image[y:y+h, x:x+w]
                label, conf = classifier_crop_sam(crop, classifier, categories)
                if label and conf > 0.15:
                    cx = x + w // 2
                    cy = y + h // 2
                    labels_info.append((label, conf, cx, cy, colors[i]))

    # Construire la figure matplotlib avec labels
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(overlay)
    for label, conf, cx, cy, color in labels_info:
        hex_color = '#%02x%02x%02x' % (int(color[0]), int(color[1]), int(color[2]))
        ax.text(cx, cy, f"{label}\n{conf:.0%}",
                fontsize=7, ha='center', va='center', fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=hex_color,
                          alpha=0.85, edgecolor='white', linewidth=0.5))
    ax.axis('off')
    plt.tight_layout(pad=0)

    return fig, mask_color, len(masks), labels_info


def extraire_objets_segmentes(image, mask_labels):
    """Extrait chaque objet segmenté individuellement (fond noir)."""
    objets = {}
    classes_uniques = np.unique(mask_labels)
    for c in classes_uniques:
        if c == 0:  # ignorer l'arrière-plan
            continue
        if c < len(PASCAL_VOC_CLASSES):
            nom_classe = PASCAL_VOC_CLASSES[c]
            masque_binaire = (mask_labels == c).astype(np.uint8)
            objet_isole = image.copy()
            for ch in range(3):
                objet_isole[:, :, ch] = objet_isole[:, :, ch] * masque_binaire
            objets[nom_classe] = objet_isole
    return objets



def page_segmentation():
    st.header("✂️ Segmentation d'images")
    st.sidebar.header("⚙️ Configuration")
    fichier_image = st.sidebar.file_uploader("Choisir une image", type=['jpg', 'jpeg', 'png', 'bmp', 'gif'], key="seg_upload")
    if fichier_image:
        image = np.array(Image.open(fichier_image))
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        st.sidebar.markdown("---")
        categorie = st.sidebar.selectbox(
            "Catégorie",
            ["Globale", "Locale", "Adaptatif", "K-means", "Deep Learning", "SAM (Segment Anything)"]
        )
        resultat = None
        info_params = []
        if categorie == "Globale":
            st.sidebar.markdown("### Méthode Globale")
            methode_globale = st.sidebar.selectbox(
                "Méthode",
                ["Manuelle", "Automatique", "Médiane", "(min+max)/2", "P-tile"]
            )
            if methode_globale == "Manuelle":
                seuil = st.sidebar.slider("Seuil", 0, 255, 127)
                resultat = binarisation_manuelle(image, seuil)
                info_params.append(("Seuil", seuil))
            elif methode_globale == "Automatique":
                resultat = binarisation_automatique(image)
                info_params.append(("Méthode", "Otsu"))
            elif methode_globale == "Médiane":
                resultat = binarisation_mediane(image)
                if len(image.shape) == 3:
                    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    img_gray = image
                info_params.append(("Seuil", f"{np.median(img_gray):.1f}"))
            elif methode_globale == "(min+max)/2":
                resultat = binarisation_min_max(image)
                if len(image.shape) == 3:
                    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    img_gray = image
                seuil = (np.min(img_gray) + np.max(img_gray)) / 2
                info_params.append(("Seuil", f"{seuil:.1f}"))
            elif methode_globale == "P-tile":
                p_tile = st.sidebar.slider("P-tile (%)", 1, 99, 50)
                resultat = binarisation_p_tile(image, p_tile)
                info_params.append(("P-tile", f"{p_tile}%"))
        elif categorie == "Locale":
            st.sidebar.markdown("### Méthode Locale")
            methode_locale = st.sidebar.selectbox(
                "Méthode",
                ["Moyenne", "Médiane", "(min+max)/2", "Niblack"]
            )
            taille_bloc = st.sidebar.slider("Taille bloc", 3, 51, 15, 2)
            info_params.append(("Taille bloc", taille_bloc))
            if methode_locale == "Moyenne":
                resultat = binarisation_locale_moyenne(image, taille_bloc)
            elif methode_locale == "Médiane":
                resultat = binarisation_locale_mediane(image, taille_bloc)
            elif methode_locale == "(min+max)/2":
                resultat = binarisation_locale_min_max(image, taille_bloc)
            elif methode_locale == "Niblack":
                k = st.sidebar.slider("Paramètre k", -1.0, 0.0, -0.2, 0.05)
                resultat = binarisation_niblack(image, k, taille_bloc)
                info_params.append(("Paramètre k", f"{k:.2f}"))
        elif categorie == "Adaptatif":
            st.sidebar.markdown("### Méthode Adaptatif")
            methode_adaptatif = st.sidebar.selectbox(
                "Méthode",
                ["Sauvola", "Wolf"]
            )
            taille_bloc = st.sidebar.slider("Taille bloc", 3, 51, 15, 2)
            info_params.append(("Taille bloc", taille_bloc))
            if methode_adaptatif == "Sauvola":
                k = st.sidebar.slider("Paramètre k", 0.1, 1.0, 0.34, 0.01)
                r = st.sidebar.slider("Paramètre r", 50, 255, 128, 1)
                resultat = binarisation_sauvola(image, k, r, taille_bloc)
                info_params.append(("Paramètre k", f"{k:.2f}"))
                info_params.append(("Paramètre r", r))
            elif methode_adaptatif == "Wolf":
                k = st.sidebar.slider("Paramètre k", 0.1, 1.0, 0.5, 0.05)
                r = st.sidebar.slider("Paramètre r", 50, 255, 128, 1)
                resultat = binarisation_wolf(image, k, r, taille_bloc)
                info_params.append(("Paramètre k", f"{k:.2f}"))
                info_params.append(("Paramètre r", r))
        elif categorie == "K-means":
            st.sidebar.markdown("### K-means")
            espace = st.sidebar.selectbox("Espace", ["rgb", "hsv"])
            k = st.sidebar.slider("Clusters K", 2, 10, 3)
            resultat = kmeans_segmentation(image, k, espace)
            info_params.append(("Clusters", k))
            info_params.append(("Espace", espace.upper()))








        elif categorie == "Deep Learning":
            st.sidebar.markdown("### 🧠 Segmentation Deep Learning")

            if not TORCH_AVAILABLE:
                st.error("⚠️ PyTorch n'est pas installé. Installez-le avec : `pip install torch torchvision`")
            else:
                # Sélection du modèle de segmentation Deep Learning
                modele_dl = st.sidebar.selectbox(
                    "Modèle",
                    ["DeepLabV3 (ResNet101)", "SegNet", "UNet", "PSPNet"]
                )
                alpha = st.sidebar.slider("Transparence overlay", 0.0, 1.0, 0.5, 0.05)
                info_params.append(("Modèle", modele_dl))
                info_params.append(("Dataset", "PASCAL VOC"))
                info_params.append(("Transparence", f"{alpha:.2f}"))

                # Dispatch vers la bonne fonction de segmentation
                spinner_msg = f"🔄 Segmentation en cours ({modele_dl})..."
                with st.spinner(spinner_msg):
                    if modele_dl == "DeepLabV3 (ResNet101)":
                        mask_color, mask_labels, classes_detectees = segmentation_deeplab(image)
                    elif modele_dl == "SegNet":
                        mask_color_seg, classes_detectees = segmentation_segnet(image)
                        # Recalculer mask_labels à partir du mask_color pour la cohérence d'affichage
                        mask_color = mask_color_seg
                        if mask_color is not None:
                            # Reconstruction approchée des labels depuis le masque coloré
                            mask_labels = np.zeros(image.shape[:2], dtype=np.uint8)
                            for idx_c, couleur in enumerate(PASCAL_VOC_COLORS):
                                match = np.all(mask_color == couleur, axis=-1)
                                mask_labels[match] = idx_c
                        else:
                            mask_labels = None
                    elif modele_dl == "UNet":
                        mask_color_seg, classes_detectees = segmentation_unet(image)
                        mask_color = mask_color_seg
                        if mask_color is not None:
                            mask_labels = np.zeros(image.shape[:2], dtype=np.uint8)
                            for idx_c, couleur in enumerate(PASCAL_VOC_COLORS):
                                match = np.all(mask_color == couleur, axis=-1)
                                mask_labels[match] = idx_c
                        else:
                            mask_labels = None
                    elif modele_dl == "PSPNet":
                        mask_color_seg, classes_detectees = segmentation_pspnet(image)
                        mask_color = mask_color_seg
                        if mask_color is not None:
                            mask_labels = np.zeros(image.shape[:2], dtype=np.uint8)
                            for idx_c, couleur in enumerate(PASCAL_VOC_COLORS):
                                match = np.all(mask_color == couleur, axis=-1)
                                mask_labels[match] = idx_c
                        else:
                            mask_labels = None
                    else:
                        mask_color, mask_labels, classes_detectees = segmentation_deeplab(image)

                if mask_color is not None:
                    # Overlay = image + masque superposé
                    overlay = superposer_segmentation(image, mask_color, alpha)

                    # Affichage principal : 3 colonnes
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.subheader("Image originale")
                        st.image(image, use_container_width=True)
                    with col_b:
                        st.subheader("Masque sémantique")
                        st.image(mask_color, use_container_width=True)
                    with col_c:
                        st.subheader("Superposition")
                        st.image(overlay, use_container_width=True)

                    # Classes détectées
                    st.markdown("---")
                    st.markdown("### 🏷️ Classes détectées")
                    if classes_detectees:
                        cols_classes = st.columns(min(len(classes_detectees), 6))
                        for idx_c, nom_c in enumerate(classes_detectees):
                            with cols_classes[idx_c % 6]:
                                class_idx = PASCAL_VOC_CLASSES.index(nom_c) if nom_c in PASCAL_VOC_CLASSES else 0
                                color = PASCAL_VOC_COLORS[class_idx]
                                nb_pixels = np.sum(mask_labels == class_idx)
                                pourcentage = nb_pixels / mask_labels.size * 100
                                color_hex = '#%02x%02x%02x' % (color[0], color[1], color[2])
                                st.markdown(
                                    f"<div style='background-color:{color_hex};padding:8px;border-radius:5px;"
                                    f"color:white;text-align:center;font-weight:bold;'>"
                                    f"{nom_c}<br>{pourcentage:.1f}%</div>",
                                    unsafe_allow_html=True
                                )
                    else:
                        st.info("Aucune classe détectée (seulement l'arrière-plan).")

                    # Légende complète
                    with st.expander("📋 Légende des 21 classes PASCAL VOC"):
                        legend_cols = st.columns(7)
                        for idx_l, (nom_l, couleur_l) in enumerate(zip(PASCAL_VOC_CLASSES, PASCAL_VOC_COLORS)):
                            with legend_cols[idx_l % 7]:
                                c_hex = '#%02x%02x%02x' % (couleur_l[0], couleur_l[1], couleur_l[2])
                                st.markdown(
                                    f"<div style='background-color:{c_hex};padding:4px;border-radius:3px;"
                                    f"color:white;text-align:center;font-size:11px;margin:2px;'>"
                                    f"{nom_l}</div>",
                                    unsafe_allow_html=True
                                )

                    # Extraction des objets individuels
                    st.markdown("---")
                    st.markdown("### 🎯 Objets segmentés individuellement")
                    objets = extraire_objets_segmentes(image, mask_labels)
                    if objets:
                        cols_obj = st.columns(min(len(objets), 4))
                        for idx_o, (nom_obj, img_obj) in enumerate(objets.items()):
                            with cols_obj[idx_o % 4]:
                                st.image(img_obj, caption=nom_obj, use_container_width=True)
                    else:
                        st.info("Aucun objet détecté (seulement l'arrière-plan).")

                    # On empêche l'affichage en 2 colonnes par défaut pour Deep Learning
                    resultat = "DEEP_LEARNING_HANDLED"
                else:
                    st.error("Erreur lors de la segmentation.")

        elif categorie == "SAM (Segment Anything)":
            st.sidebar.markdown("### SAM — Segment Anything")
            if not SAM_AVAILABLE:
                st.error("La librairie `segment-anything` n'est pas installée. Lancez : `pip install segment-anything`")
            elif not os.path.exists(SAM_CHECKPOINT):
                st.warning("Le checkpoint SAM (ViT-B) est introuvable.")
                st.info("""
**Pour utiliser SAM :**
1. Télécharge le checkpoint ViT-B depuis le dépôt officiel de Meta (segment-anything sur GitHub)
2. Place le fichier `sam_vit_b_01ec64.pth` dans le dossier de l'application
3. Relance l'application
                """)
            else:
                points_per_side = st.sidebar.slider("Densité d'échantillonnage", 8, 64, 32, 8)
                avec_labels = st.sidebar.checkbox("Labelliser les objets (ResNet50)", value=True)
                st.sidebar.info("Plus la densité est élevée, plus la segmentation est fine (mais plus lente).")
                with st.spinner("Segmentation SAM en cours..."):
                    fig_overlay, mask_color, nb_masques, labels_info = segmentation_sam(
                        image, points_per_side, avec_labels=avec_labels
                    )
                if fig_overlay is not None:
                    st.success(f"{nb_masques} objets détectés" + (f" — {len(labels_info)} labellisés" if avec_labels else ""))
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.subheader("Image originale")
                        st.image(image, use_container_width=True)
                        st.subheader("Masques SAM")
                        st.image(mask_color, use_container_width=True)
                    with col_b:
                        st.subheader("Superposition" + (" + Labels ResNet50" if avec_labels else ""))
                        st.pyplot(fig_overlay)
                        plt.close(fig_overlay)
                else:
                    st.error("Erreur lors de la segmentation SAM.")
                resultat = "DEEP_LEARNING_HANDLED"

        # Affichage pour les méthodes non-Deep Learning
        if categorie not in ("Deep Learning", "SAM (Segment Anything)"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Image originale")
                st.image(image, use_container_width=True)
            with col2:
                st.subheader("Image segmentée")
                if resultat is not None:
                    if len(resultat.shape) == 2:
                        resultat = cv2.cvtColor(resultat, cv2.COLOR_GRAY2RGB)
                    st.image(resultat, use_container_width=True)
                else:
                    st.info("Sélectionnez une méthode pour voir le résultat")

        if info_params:
            st.markdown("---")
            st.markdown("### 📊 Paramètres utilisés")
            cols = st.columns(min(len(info_params), 4))
            for i, (label, value) in enumerate(info_params):
                with cols[i % 4]:
                    st.metric(label, value)


def page_recherche():
    st.header("🔍 Recherche d'images")
    st.sidebar.header("⚙️ Configuration")

    type_descripteur = st.sidebar.selectbox(
        "Descripteur",
        DESCRIPTEUR_KEYS,
        index=0,
        format_func=lambda x: DESCRIPTEUR_LABELS.get(x, x)
    )

    methode_distance = st.sidebar.selectbox(
        "Distance",
        ['euclidienne', 'manhattan', 'cosinus'],
        index=0
    )

    force_reindex = st.sidebar.checkbox("Réindexer", value=False)

    if type_descripteur == 'desc_cnn' and not TORCH_AVAILABLE:
        st.error("⚠️ PyTorch n'est pas installé. Installez-le avec : `pip install torch torchvision`")
        return
    if type_descripteur == 'desc_ann' and not TORCH_AVAILABLE:
        st.error("⚠️ PyTorch n'est pas installé. Installez-le avec : `pip install torch torchvision`")
        return
    if type_descripteur == 'desc_ann_pretrained' and not TORCH_AVAILABLE:
        st.error("⚠️ PyTorch n'est pas installé. Installez-le avec : `pip install torch torchvision`")
        return
    
    if 'db' not in st.session_state or force_reindex:
        with st.spinner("Indexation en cours..."):
            st.session_state.db = indexer_base(force_reindex=force_reindex)

    encoder_ann = None
    if type_descripteur == 'desc_ann':
        encoder_ann = get_ann_encoder(st.session_state.db)
        if encoder_ann is None:
            st.error("Impossible d'entraîner l'autoencodeur ANN.")
            return

    if type_descripteur == 'desc_ann_dataset' and not os.path.exists(CONV_AE_PATH):
        st.warning("Le ConvAutoencoder n'est pas encore entraîné sur le dataset.")
        if st.button("Entraîner maintenant (~1 min)"):
            if 'db' not in st.session_state:
                st.session_state.db = indexer_base()
            prog = st.progress(0)
            status_ae = st.empty()
            status_ae.text("Entraînement en cours... (50 epochs sur 200 images)")
            train_conv_autoencoder_dataset(
                st.session_state.db, epochs=50,
                progress_callback=lambda p: prog.progress(p)
            )
            st.cache_resource.clear()
            prog.empty()
            status_ae.empty()
            st.success("Modèle entraîné et sauvegardé définitivement !")
            st.rerun()
        return

    fichier_requete = st.file_uploader("Choisir une image", type=['jpg', 'jpeg', 'png', 'bmp', 'gif'], key="search_upload")

    if fichier_requete:
        img_requete = np.array(Image.open(fichier_requete))
        if len(img_requete.shape) == 2:
            img_requete = cv2.cvtColor(img_requete, cv2.COLOR_GRAY2RGB)
        elif img_requete.shape[2] == 4:
            img_requete = cv2.cvtColor(img_requete, cv2.COLOR_RGBA2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_requete, caption="Image de requête", use_container_width=True)
        with col2:
            st.markdown("### 📈 Visualisation du Vecteur")
            desc_visu = extraire_descripteur_par_type(img_requete, type_descripteur, encoder_ann=encoder_ann)
            if desc_visu is not None:
                st.info(f"Dimension du vecteur : {len(desc_visu)} valeurs")
                if type_descripteur in ['desc_stat', 'desc_stat_complet', 'glcm', 'desc_stat_entropy', 'cds', 'dcd']:
                    st.bar_chart(desc_visu)
                else:
                    st.line_chart(desc_visu)
                with st.expander("🔢 Voir les valeurs brutes"):
                    st.write(desc_visu)
            else:
                st.warning("Impossible d'extraire le descripteur pour cette image.")

        st.markdown("---")
        k = st.slider("Nombre de résultats", 1, 50, 10)

        if st.button("Rechercher", use_container_width=True):
            with st.spinner("Recherche..."):
                resultats = rechercher_images_avec_descripteur(
                    img_requete,
                    st.session_state.db,
                    type_descripteur=type_descripteur,
                    methode_distance=methode_distance,
                    encoder_ann=encoder_ann
                )
            afficher_resultats(resultats, k=k)


def _evaluer_descripteur_distance(db, desc_type, distance):
    """Calcule MAP et MAP@10 pour un descripteur + distance. Retourne (MAP, MAP@10) ou (None, None)."""
    # desc_ann nécessite un encodeur entraîné — on l'entraîne une fois ici
    encoder_ann = None
    if desc_type == 'desc_ann':
        if not TORCH_AVAILABLE:
            return None, None
        encoder_ann = train_ann_autoencoder(db)
        if encoder_ann is None:
            return None, None

    # desc_ann_dataset nécessite le ConvAutoencoder sauvegardé sur disque
    if desc_type == 'desc_ann_dataset' and not os.path.exists(CONV_AE_PATH):
        return None, None

    desc_db = {}
    for i, item in enumerate(db):
        img = charger_image(item['chemin'])
        if img is None:
            continue
        try:
            d = extraire_descripteur_par_type(img, desc_type, encoder_ann=encoder_ann)
        except Exception:
            d = None
        if d is not None:
            desc_db[i] = d
    if len(desc_db) < 2:
        return None, None
    aps, aps10 = [], []
    keys = list(desc_db.keys())
    for idx_q in keys:
        d_q = desc_db[idx_q]
        classe_q = db[idx_q]['classe']
        resultats = []
        for idx_db in keys:
            if idx_db == idx_q:
                continue
            dist_val = calculer_distance(d_q, desc_db[idx_db], distance)
            resultats.append({'classe': db[idx_db]['classe'], 'distance': dist_val})
        resultats_tries = sorted(resultats, key=lambda x: x['distance'])
        aps.append(calculer_ap(resultats_tries, classe_q))
        aps10.append(calculer_ap(resultats_tries[:10], classe_q))
    return float(np.mean(aps)), float(np.mean(aps10))


def page_evaluation():
    st.header("Résultats de l'évaluation des descripteurs")
    structure = explorer_base(BASE_PATH)
    nb_images = sum(len(imgs) for imgs in structure.values())
    st.info(f"Dataset : {nb_images} images réparties en {len(structure)} classes")
    with st.expander("Voir les classes du dataset"):
        for classe, items in sorted(structure.items()):
            st.write(f"  - {classe} : {len(items)} images")

    csv_path = os.path.join(os.path.dirname(__file__), "resultats_evaluation.csv")

    # ── Bouton pour lancer/compléter l'évaluation ──────────────────────────
    DESCRIPTEURS_EVAL = [
        'hist_rgb', 'hist_pond_sat', 'hist_cumule', 'hist_entropie',
        'lbp', 'lbp_blocs', 'glcm',
        'desc_stat', 'desc_stat_complet', 'desc_stat_entropy',
        'cds', 'dcd', 'ccd',
        'desc_forme_sobel', 'desc_forme_prewitt', 'desc_forme_roberts',
        'hog', 'hog_non_pondere', 'hog_blocs',
        'desc_cnn', 'desc_ann', 'desc_ann_dataset', 'desc_ann_pretrained',
    ]
    DISTANCES_EVAL = ['euclidienne', 'manhattan', 'cosinus']

    existing = set()
    rows_existing = []
    if os.path.exists(csv_path):
        df_exist = pd.read_csv(csv_path)
        for _, row in df_exist.iterrows():
            existing.add((row['Descripteur'], row['Distance']))
            rows_existing.append(row.to_dict())

    # ── Statut du ConvAutoencoder (nécessaire pour desc_ann_dataset) ───────────
    st.markdown("#### Modèle ConvAutoencoder (ANN entraîné sur notre dataset)")
    if os.path.exists(CONV_AE_PATH):
        st.success("Modèle entraîné et prêt à l'utilisation.")
    else:
        st.error("Modèle non entraîné — desc_ann_dataset ne sera pas évaluable.")
        if TORCH_AVAILABLE:
            if st.button("Entraîner le ConvAutoencoder maintenant (~1 min)"):
                if 'db' not in st.session_state:
                    st.session_state.db = indexer_base()
                prog = st.progress(0)
                status_ae = st.empty()
                status_ae.text("Entraînement en cours... (50 epochs sur 200 images)")
                train_conv_autoencoder_dataset(
                    st.session_state.db, epochs=50,
                    progress_callback=lambda p: prog.progress(p)
                )
                st.cache_resource.clear()
                prog.empty(); status_ae.empty()
                st.success("Entraînement terminé ! Modèle sauvegardé définitivement.")
                st.rerun()
        else:
            st.info("PyTorch non disponible — impossible d'entraîner le modèle.")
    st.markdown("---")

    todo = [(d, dist) for d in DESCRIPTEURS_EVAL for dist in DISTANCES_EVAL if (d, dist) not in existing]
    nb_todo = len(todo)
    nb_total = len(DESCRIPTEURS_EVAL) * len(DISTANCES_EVAL)

    if nb_todo > 0:
        st.warning(f"{nb_todo} combinaisons non encore évaluées sur {nb_total}.")
        if st.button(f"Lancer l'évaluation des {nb_todo} combinaisons manquantes"):
            db = []
            for classe, images in structure.items():
                chemin_classe = os.path.join(BASE_PATH, classe)
                for img_name in images:
                    db.append({'chemin': os.path.join(chemin_classe, img_name), 'classe': classe})
            rows = list(rows_existing)
            progress = st.progress(0)
            status = st.empty()
            for i, (desc, dist) in enumerate(todo):
                status.text(f"Évaluation : {DESCRIPTEUR_LABELS.get(desc, desc)} × {dist} ({i+1}/{nb_todo})")
                map_val, map10_val = _evaluer_descripteur_distance(db, desc, dist)
                if map_val is not None:
                    rows.append({'Descripteur': desc, 'Distance': dist, 'MAP': map_val, 'MAP@10': map10_val})
                    pd.DataFrame(rows).to_csv(csv_path, index=False)
                progress.progress((i + 1) / nb_todo)
            progress.empty()
            status.empty()
            st.success("Évaluation terminée ! Rechargez la page pour voir les résultats.")
            st.rerun()
    else:
        st.success(f"Tous les {nb_total} descripteurs × distances sont évalués.")

    st.markdown("---")

    # ── Affichage des résultats ─────────────────────────────────────────────
    if not os.path.exists(csv_path):
        st.error("Aucun résultat disponible. Lancez l'évaluation ci-dessus.")
        return

    df = pd.read_csv(csv_path)
    df['Nom'] = df['Descripteur'].map(lambda k: DESCRIPTEUR_LABELS.get(k, k))
    df_sorted = df.sort_values('MAP', ascending=False).reset_index(drop=True)

    # Métriques du meilleur
    best = df_sorted.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Meilleur descripteur", DESCRIPTEUR_LABELS.get(best['Descripteur'], best['Descripteur']))
    c2.metric("Meilleure distance", best['Distance'])
    c3.metric("Meilleur MAP", f"{best['MAP']:.4f}")
    c4.metric("Meilleur MAP@10", f"{best['MAP@10']:.4f}")

    st.markdown("---")

    # ── Graphique 1 : MAP par descripteur (meilleure distance) ─────────────
    st.subheader("MAP par descripteur (meilleure distance)")
    idx_best = df.groupby('Descripteur')['MAP'].idxmax()
    df_best = df.loc[idx_best].copy()
    df_best['Nom'] = df_best['Descripteur'].map(lambda k: DESCRIPTEUR_LABELS.get(k, k))
    df_best = df_best.sort_values('MAP', ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(4, len(df_best) * 0.35)))
    colors = ['#2ecc71' if v == df_best['MAP'].max() else '#3498db' for v in df_best['MAP']]
    bars = ax.barh(df_best['Nom'], df_best['MAP'], color=colors, edgecolor='white', height=0.6)
    for bar, val in zip(bars, df_best['MAP']):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va='center', fontsize=8)
    ax.set_xlabel("MAP", fontsize=10)
    ax.set_xlim(0, df_best['MAP'].max() * 1.15)
    ax.set_title("MAP par descripteur (meilleure distance parmi euclidienne/manhattan/cosinus)", fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # ── Graphique 2 : MAP vs MAP@10 pour les meilleurs descripteurs ─────────
    st.subheader("Comparaison MAP vs MAP@10 (top 10 descripteurs)")
    df_top = df_best.sort_values('MAP', ascending=False).head(10)
    x = np.arange(len(df_top))
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.bar(x - 0.2, df_top['MAP'], width=0.35, label='MAP', color='#3498db')
    ax2.bar(x + 0.2, df_top['MAP@10'], width=0.35, label='MAP@10', color='#e67e22')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_top['Nom'], rotation=30, ha='right', fontsize=8)
    ax2.set_ylabel("Score")
    ax2.set_title("MAP vs MAP@10 — Top 10 descripteurs")
    ax2.legend()
    ax2.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.markdown("---")

    # ── Graphique 3 : effet de la distance pour chaque descripteur ──────────
    st.subheader("Effet de la distance sur le MAP")
    desc_options = sorted(df['Descripteur'].unique(),
                          key=lambda k: DESCRIPTEUR_LABELS.get(k, k))
    desc_choisi = st.selectbox(
        "Choisir un descripteur",
        desc_options,
        format_func=lambda k: DESCRIPTEUR_LABELS.get(k, k)
    )
    df_dist = df[df['Descripteur'] == desc_choisi].sort_values('MAP', ascending=False)
    fig3, ax3 = plt.subplots(figsize=(5, 3))
    bar_colors = ['#2ecc71' if v == df_dist['MAP'].max() else '#95a5a6' for v in df_dist['MAP']]
    ax3.bar(df_dist['Distance'], df_dist['MAP'], color=bar_colors, edgecolor='white')
    for i, (_, row) in enumerate(df_dist.iterrows()):
        ax3.text(i, row['MAP'] + 0.003, f"{row['MAP']:.4f}", ha='center', fontsize=9)
    ax3.set_ylabel("MAP")
    ax3.set_title(f"{DESCRIPTEUR_LABELS.get(desc_choisi, desc_choisi)} — MAP selon la distance")
    ax3.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    st.markdown("---")

    # ── Tableau complet ─────────────────────────────────────────────────────
    st.subheader("Tableau complet")
    df_display = df_sorted[['Nom', 'Distance', 'MAP', 'MAP@10']].copy()
    df_display['MAP'] = df_display['MAP'].round(4)
    df_display['MAP@10'] = df_display['MAP@10'].round(4)
    st.dataframe(df_display, use_container_width=True, hide_index=True)
    st.info(f"Résultats chargés depuis `{os.path.basename(csv_path)}`")


def page_traitement():
    st.header("🛠️ Traitements d'image (Module 912)")
    st.sidebar.header("⚙️ Configuration Traitement")
    fichier_image = st.sidebar.file_uploader("Choisir une image", type=['jpg', 'jpeg', 'png', 'bmp', 'gif'], key="traitement_upload")
    if fichier_image:
        image = np.array(Image.open(fichier_image))
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        st.sidebar.markdown("---")
        mode_traitement = st.sidebar.selectbox(
            "Type de traitement",
            ["Analyse de Forme (Gradients)", "Filtrage (Convolution/Corrélation)", "Restauration & Apparence", "Espaces de couleur"]
        )

        if mode_traitement == "Espaces de couleur":
            st.subheader("🎨 Conversion d'espaces de couleur")
            espace = st.sidebar.selectbox("Espace cible", ["HSV", "Lab", "YCbCr", "Niveaux de gris"])
            if espace == "HSV":
                converted = convertir_hsv(image)
            elif espace == "Lab":
                converted = convertir_lab(image)
            elif espace == "YCbCr":
                converted = convertir_ycbcr(image)
            else:
                converted = convertir_gris(image)
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Originale (RGB)", use_container_width=True)
            with col2:
                if len(converted.shape) == 2:
                    st.image(converted, caption=f"Convertie ({espace})", use_container_width=True)
                else:
                    st.image(converted, caption=f"Convertie ({espace})", use_container_width=True)
            if len(converted.shape) == 3:
                st.markdown("### Canaux séparés")
                ch_names = {"HSV": ["H", "S", "V"], "Lab": ["L", "a", "b"], "YCbCr": ["Y", "Cb", "Cr"]}
                names = ch_names.get(espace, ["C1", "C2", "C3"])
                cols = st.columns(3)
                for i in range(3):
                    with cols[i]:
                        st.image(converted[:, :, i], caption=names[i], use_container_width=True)
            st.markdown("### 🌐 Visualisation 3D des pixels")
            sample_size = min(1000, image.shape[0] * image.shape[1])
            pixels_flat = image.reshape(-1, 3)
            indices = np.random.choice(len(pixels_flat), sample_size, replace=False)
            sampled = pixels_flat[indices]
            if espace != "Niveaux de gris" and len(converted.shape) == 3:
                conv_flat = converted.reshape(-1, 3)
                sampled_conv = conv_flat[indices]
                ch_n = ch_names.get(espace, ["C1", "C2", "C3"])
                df_scatter = pd.DataFrame(sampled_conv, columns=ch_n)
                colors = ['#%02x%02x%02x' % (r, g, b) for r, g, b in sampled]
                df_scatter['color'] = colors
                st.scatter_chart(df_scatter, x=ch_n[0], y=ch_n[1], color='color')

        elif mode_traitement == "Analyse de Forme (Gradients)":
            st.subheader("Analyse des Gradients (Dérivées, Norme, Direction)")
            operateur = st.sidebar.selectbox("Opérateur", ["Sobel", "Prewitt", "Roberts"])
            img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            grad_x, grad_y = None, None
            if operateur == "Sobel":
                grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
            elif operateur == "Prewitt":
                kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
                kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
                grad_x = cv2.filter2D(img_gray, cv2.CV_64F, kernel_x)
                grad_y = cv2.filter2D(img_gray, cv2.CV_64F, kernel_y)
            elif operateur == "Roberts":
                kernel_x = np.array([[1, 0], [0, -1]])
                kernel_y = np.array([[0, 1], [-1, 0]])
                grad_x = cv2.filter2D(img_gray, cv2.CV_64F, kernel_x)
                grad_y = cv2.filter2D(img_gray, cv2.CV_64F, kernel_y)
            if grad_x is not None and grad_y is not None:
                norme = np.sqrt(grad_x**2 + grad_y**2)
                direction = np.arctan2(grad_y, grad_x)
                norme_disp = cv2.normalize(norme, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Originale", use_container_width=True)
                    st.image(grad_x / (2 * np.max(np.abs(grad_x)) + 1e-10) + 0.5, caption=f"Dérivée X ({operateur}) - Contours verticaux (gris=0, blanc=+, noir=-)", use_container_width=True)
                with col2:
                    st.image(norme_disp, caption="Norme (Magnitude)", use_container_width=True)
                    st.image(grad_y / (2 * np.max(np.abs(grad_y)) + 1e-10) + 0.5, caption=f"Dérivée Y ({operateur}) - Contours horizontaux (gris=0, blanc=+, noir=-)", use_container_width=True)
                st.markdown("### Direction des gradients")
                step = max(min(img_gray.shape[0], img_gray.shape[1]) // 30, 8)
                y_idx = np.arange(step // 2, img_gray.shape[0], step)
                x_idx = np.arange(step // 2, img_gray.shape[1], step)
                X_grid, Y_grid = np.meshgrid(x_idx, y_idx)
                U = grad_x[np.ix_(y_idx, x_idx)]
                V = grad_y[np.ix_(y_idx, x_idx)]
                mag_grid = norme[np.ix_(y_idx, x_idx)]
                norm_uv = np.sqrt(U**2 + V**2) + 1e-10
                U_n = U / norm_uv
                V_n = V / norm_uv
                threshold = np.percentile(norme, 60)
                mask = mag_grid > threshold
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(img_gray, cmap='gray', interpolation='nearest')
                q = ax.quiver(
                    X_grid[mask], Y_grid[mask], U_n[mask], -V_n[mask],
                    mag_grid[mask],
                    cmap='hot', clim=[threshold, norme.max()],
                    scale=20, width=0.003, headwidth=4, headlength=4
                )
                cbar = plt.colorbar(q, ax=ax, fraction=0.03, pad=0.04)
                cbar.set_label("Magnitude du gradient", fontsize=10)
                ax.set_title(f"Direction des gradients — {operateur}\n(couleur = intensité du contour)", fontsize=11)
                ax.axis('off')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.error("Erreur lors du calcul des gradients.")

        elif mode_traitement == "Filtrage (Convolution/Corrélation)":
            st.subheader("Convolution, Corrélation et Padding")
            type_filtre = st.sidebar.selectbox("Filtre", ["Moyenneur (Blur)", "Gaussien", "Laplacien (Bords)", "Custom (Asymétrique)"])
            taille_k = st.sidebar.slider("Taille Noyau", 3, 15, 3, step=2)
            kernel = None
            if type_filtre == "Moyenneur (Blur)":
                kernel = np.ones((taille_k, taille_k), np.float32) / (taille_k**2)
            elif type_filtre == "Gaussien":
                kernel = cv2.getGaussianKernel(taille_k, 0)
                kernel = kernel @ kernel.T
            elif type_filtre == "Laplacien (Bords)":
                kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
            elif type_filtre == "Custom (Asymétrique)":
                kernel = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0.5]], dtype=np.float32)
                st.sidebar.info("Noyau Asymétrique utilisé pour montrer la différence Conv/Corr")
            padding_mode_str = st.sidebar.radio("Padding (Bords)", ["Zero (0)", "Replica (Copie)", "Reflect (Miroir)"])
            padding_map = {
                "Zero (0)": cv2.BORDER_CONSTANT,
                "Replica (Copie)": cv2.BORDER_REPLICATE,
                "Reflect (Miroir)": cv2.BORDER_REFLECT
            }
            border_type = padding_map[padding_mode_str]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Cross-Correlation")
                res_corr = cv2.filter2D(image, -1, kernel, borderType=border_type)
                st.image(res_corr, caption="Résultat Corrélation", use_container_width=True)
            with col2:
                st.markdown("#### Convolution")
                kernel_flipped = cv2.flip(kernel, -1)
                res_conv = cv2.filter2D(image, -1, kernel_flipped, borderType=border_type)
                st.image(res_conv, caption="Résultat Convolution (Kernel retourné)", use_container_width=True)
            if type_filtre == "Custom (Asymétrique)":
                st.warning("⚠️ Observez la différence ci-dessus ! Avec un filtre asymétrique, le résultat change.")
            else:
                st.success("ℹ️ Avec un filtre symétrique, Convolution et Corrélation donnent le même résultat.")
            with st.expander("Voir le noyau (kernel) actuel"):
                st.write(kernel)

        elif mode_traitement == "Restauration & Apparence":
            st.subheader("Restauration et Modification d'Apparence")
            option = st.sidebar.radio("Opération", ["Réhaussement (Sharpening)", "Quantification (Réduction couleurs)"])
            if option == "Réhaussement (Sharpening)":
                st.markdown("Renforce les détails et les bords en ajoutant le Laplacien à l'image originale.")
                force = st.sidebar.slider("Force", 0.0, 2.0, 1.0)
                image_blur = cv2.GaussianBlur(image, (0, 0), 3)
                image_sharp = cv2.addWeighted(image, 1.0 + force, image_blur, -force, 0)
                col1, col2 = st.columns(2)
                col1.image(image, caption="Originale", use_container_width=True)
                col2.image(image_sharp, caption=f"Réhaussée (Force {force})", use_container_width=True)
            elif option == "Quantification (Réduction couleurs)":
                st.markdown("Réduit le nombre de niveaux de gris/couleurs (Posterization).")
                n_couleurs = st.sidebar.slider("Nombre de niveaux par canal", 2, 64, 8)
                indices = np.arange(0, 256)
                divider = np.linspace(0, 255, n_couleurs + 1)[1]
                quantiz = np.int64(indices / divider) * divider
                lut = np.clip(quantiz, 0, 255).astype(np.uint8)
                if len(image.shape) == 3:
                    r, g, b = cv2.split(image)
                    r_q = cv2.LUT(r, lut)
                    g_q = cv2.LUT(g, lut)
                    b_q = cv2.LUT(b, lut)
                    img_quant = cv2.merge([r_q, g_q, b_q])
                else:
                    img_quant = cv2.LUT(image, lut)
                col1, col2 = st.columns(2)
                col1.image(image, caption="Originale", use_container_width=True)
                col2.image(img_quant, caption=f"Quantifiée ({n_couleurs} niveaux)", use_container_width=True)


def page_clustering():
    st.header("📦 Clustering d'Images (Non-supervisé)")
    st.markdown("Regroupement automatique des images par similarité visuelle (K-Means).")
    st.sidebar.header("⚙️ Configuration Clustering")

    type_descripteur = st.sidebar.selectbox(
        "Descripteur",
        DESCRIPTEUR_KEYS,
        index=0,
        key="cluster_desc",
        format_func=lambda x: DESCRIPTEUR_LABELS.get(x, x)
    )

    k_clusters = st.sidebar.slider("Nombre de clusters (K)", 2, 20, 5)

    if type_descripteur == 'desc_cnn' and not TORCH_AVAILABLE:
        st.error("⚠️ PyTorch n'est pas installé. Installez-le avec : `pip install torch torchvision`")
        return
    if type_descripteur == 'desc_ann' and not TORCH_AVAILABLE:
        st.error("⚠️ PyTorch n'est pas installé. Installez-le avec : `pip install torch torchvision`")
        return
    
    if type_descripteur == 'desc_ann_pretrained' and not TORCH_AVAILABLE:
        st.error("⚠️ PyTorch n'est pas installé. Installez-le avec : `pip install torch torchvision`")
        return
    
    if st.button("Lancer le Clustering", use_container_width=True):
        if 'db' not in st.session_state:
            st.session_state.db = indexer_base()
        db = st.session_state.db

        encoder_ann = None
        if type_descripteur == 'desc_ann':
            encoder_ann = get_ann_encoder(db)
            if encoder_ann is None:
                st.error("Impossible d'entraîner l'autoencodeur ANN.")
                return

        if type_descripteur == 'desc_ann_dataset' and not os.path.exists(CONV_AE_PATH):
            st.warning("Le ConvAutoencoder n'est pas encore entraîné. Allez dans Recherche d'images pour l'entraîner d'abord.")
            return

        vectors = []
        valid_indices = []
        status = st.empty()
        bar = st.progress(0)
        status.text("Extraction des descripteurs pour le clustering...")
        for i, item in enumerate(db):
            img = charger_image(item['chemin'])
            if img is not None:
                vec = extraire_descripteur_par_type(img, type_descripteur, encoder_ann=encoder_ann)
                if vec is not None:
                    vectors.append(vec)
                    valid_indices.append(i)
            bar.progress((i + 1) / len(db))
        status.text("Calcul du K-Means...")
        if not vectors:
            st.error("Aucun vecteur extrait. Vérifiez les images.")
            return
        X = np.array(vectors, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        attempts = 10
        flags = cv2.KMEANS_PP_CENTERS
        _, labels, _ = cv2.kmeans(X, k_clusters, None, criteria, attempts, flags)
        labels = labels.flatten()
        status.empty()
        bar.empty()
        st.success(f"Clustering terminé en {k_clusters} groupes !")
        tabs_clusters = st.tabs([f"Groupe {i+1}" for i in range(k_clusters)])
        for i in range(k_clusters):
            with tabs_clusters[i]:
                cluster_idxs = np.where(labels == i)[0]
                nb_imgs = len(cluster_idxs)
                st.markdown(f"**{nb_imgs} images** dans ce groupe.")
                cols = st.columns(5)
                for j, idx_val in enumerate(cluster_idxs):
                    original_idx = valid_indices[idx_val]
                    item = db[original_idx]
                    if j < 20:
                        with cols[j % 5]:
                            st.image(item['chemin'], caption=item['nom'], use_container_width=True)
                if nb_imgs > 20:
                    st.info(f"... et {nb_imgs - 20} autres images.")


# ===== PIPELINE R-CNN (Region-based Convolutional Neural Network) =====
#
# Le pipeline R-CNN (Girshick et al., 2014) se décompose en 4 étapes :
#   1. Extraction de propositions de régions (ici : fenêtre glissante)
#      → On fait glisser des fenêtres de différentes tailles sur l'image
#        pour générer ~2000 régions candidates ("region proposals")
#   2. Extraction de features CNN pour chaque région
#      → Chaque région est redimensionnée en 224×224 et passée dans le backbone VGG16
#        → vecteur de 25 088 valeurs (features visuelles profondes)
#   3. Classification SVM par région
#      → Un SVM (Support Vector Machine) entraîné sur notre dataset
#        prédit la classe de chaque région (ou "fond")
#   4. Régression de boîtes englobantes (bounding box regression)
#      → Un régresseur Ridge ajuste les coordonnées (x, y, w, h) de chaque boîte
#        pour affiner la localisation de l'objet détecté
# Enfin, une suppression des non-maxima (NMS) élimine les détections redondantes.
#
# Note : Cette implémentation utilise la fenêtre glissante au lieu de la
#        recherche sélective (selective search) pour simplifier le pipeline.


RCNN_SVM_PATH = {
    "VGG16":   os.path.join(os.path.dirname(__file__), "rcnn_svm_vgg16.pkl"),
    "AlexNet": os.path.join(os.path.dirname(__file__), "rcnn_svm_alexnet.pkl"),
}
RCNN_REG_PATH = {
    "VGG16":   os.path.join(os.path.dirname(__file__), "rcnn_reg_vgg16.pkl"),
    "AlexNet": os.path.join(os.path.dirname(__file__), "rcnn_reg_alexnet.pkl"),
}


def extraire_propositions(image, tailles=None, stride=32):
    """
    Génère des propositions de régions par fenêtre glissante.
    Fait glisser des fenêtres de plusieurs tailles sur l'image avec un pas `stride`.
    Retourne une liste de tuples (x, y, w, h) — limité à 2000 propositions max.
    """
    if tailles is None:
        tailles = [(64, 64), (96, 96), (128, 128), (192, 192)]
    h_img, w_img = image.shape[:2]
    propositions = []
    for (ww, hh) in tailles:
        # Faire glisser la fenêtre (ww×hh) sur l'image avec un pas `stride`
        for y in range(0, h_img - hh + 1, stride):
            for x in range(0, w_img - ww + 1, stride):
                propositions.append((x, y, ww, hh))
    # Limiter à 2000 propositions pour la performance
    if len(propositions) > 2000:
        # Échantillonnage uniforme parmi toutes les propositions
        indices = np.linspace(0, len(propositions) - 1, 2000, dtype=int)
        propositions = [propositions[i] for i in indices]
    return propositions


if TORCH_AVAILABLE:
    @st.cache_resource
    def get_vgg_backbone():
        """
        Charge VGG16 pré-entraîné et supprime le classificateur final.
        On utilise un GlobalAveragePooling 1×1 sur les features conv → vecteur 512 dims.
        512 dims (au lieu de 25088) est bien plus adapté pour un SVM entraîné sur 200 images :
        moins d'overfitting, meilleure calibration des probabilités.
        """
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            # GlobalAveragePooling 1×1 → 512 features au lieu de 25088
            # Bien plus approprié pour SVM avec peu de données
            class VGGBackbone(nn.Module):
                def __init__(self, features):
                    super().__init__()
                    self.features = features
                    self.pool = nn.AdaptiveAvgPool2d((1, 1))
                def forward(self, x):
                    x = self.features(x)
                    x = self.pool(x)
                    return x.view(x.size(0), -1)  # (B, 512)
            backbone = VGGBackbone(vgg.features)
            backbone.eval()
            return backbone  # sortie : vecteur 512 dimensions
        except Exception:
            return None

    @st.cache_resource
    def get_alexnet_backbone():
        """
        Charge AlexNet pré-entraîné sur ImageNet et supprime le classificateur final.
        On garde uniquement les couches convolutives (features) + AdaptiveAvgPool → 256 dims.
        AlexNet est plus léger que VGG16 : plus rapide, utile avec peu de données.
        """
        try:
            alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
            class AlexNetBackbone(nn.Module):
                def __init__(self, features):
                    super().__init__()
                    self.features = features
                    self.pool = nn.AdaptiveAvgPool2d((1, 1))
                def forward(self, x):
                    x = self.features(x)
                    x = self.pool(x)
                    return x.view(x.size(0), -1)  # (B, 256)
            backbone = AlexNetBackbone(alexnet.features)
            backbone.eval()
            return backbone  # sortie : vecteur 256 dimensions
        except Exception:
            return None

    def extraire_features_region(crop, backbone):
        """
        Extrait le vecteur de features d'un crop d'image via le backbone (VGG16 ou AlexNet).
        Redimensionne le crop en 224×224, normalise (ImageNet), passe dans le backbone.
        """
        if backbone is None or crop is None or crop.size == 0:
            return None
        if crop.shape[0] < 4 or crop.shape[1] < 4:
            return None
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        try:
            x = preprocess(crop).unsqueeze(0)  # (1, 3, 224, 224)
            with torch.no_grad():
                features = backbone(x)  # (1, 512) pour VGG16 ou (1, 256) pour AlexNet
            return features.squeeze().numpy()
        except Exception:
            return None


def entrainer_svm_rcnn(db, backbone, backbone_name="VGG16"):
    """
    Entraîne un SVM linéaire sur les features du backbone (VGG16 → 512 dims, AlexNet → 256 dims).
    Pour chaque image : extrait features de l'image complète + 3 crops aléatoires.
    Cela permet au SVM de reconnaître des objets partiels (comme lors de l'inférence).
    Applique un StandardScaler avant le SVM pour normaliser les features.
    Sauvegarde le modèle dans rcnn_svm_<backbone>.pkl.
    """
    if not TORCH_AVAILABLE or backbone is None:
        return None
    try:
        from sklearn.svm import LinearSVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.pipeline import Pipeline
    except ImportError:
        return None

    X, y = [], []
    status = st.empty()
    bar = st.progress(0)
    rng = np.random.default_rng(42)
    for i, item in enumerate(db):
        img = charger_image(item['chemin'])
        if img is not None:
            # Image complète
            feat = extraire_features_region(img, backbone)
            if feat is not None:
                X.append(feat)
                y.append(item['classe'])
            # 3 crops aléatoires pour simuler les propositions de régions
            h, w = img.shape[:2]
            for _ in range(3):
                cw = rng.integers(w // 3, w)
                ch = rng.integers(h // 3, h)
                cx = rng.integers(0, max(1, w - cw))
                cy = rng.integers(0, max(1, h - ch))
                crop = img[cy:cy+ch, cx:cx+cw]
                feat_crop = extraire_features_region(crop, backbone)
                if feat_crop is not None:
                    X.append(feat_crop)
                    y.append(item['classe'])
        bar.progress((i + 1) / len(db))
        status.text(f"Extraction features SVM R-CNN : {i+1}/{len(db)}")
    bar.empty()
    status.empty()

    if len(X) < 2:
        return None

    # Pipeline : StandardScaler + LinearSVC calibré avec probabilités
    # LinearSVC est bien plus rapide et généralise mieux que RBF avec peu de données
    # CalibratedClassifierCV ajoute la calibration des probabilités (nécessaire pour les scores)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', CalibratedClassifierCV(LinearSVC(C=0.1, max_iter=2000), cv=3))
    ])
    pipeline.fit(np.array(X), y)

    svm_path = RCNN_SVM_PATH.get(backbone_name, RCNN_SVM_PATH["VGG16"])
    with open(svm_path, 'wb') as f:
        pickle.dump(pipeline, f)

    return pipeline


def charger_svm_rcnn(backbone_name="VGG16"):
    """Charge le SVM R-CNN depuis le disque (None si pas encore entraîné)."""
    svm_path = RCNN_SVM_PATH.get(backbone_name, RCNN_SVM_PATH["VGG16"])
    if not os.path.exists(svm_path):
        return None
    try:
        with open(svm_path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def entrainer_regression_bbox(db, backbone, svm, backbone_name="VGG16"):
    """
    Entraîne une régression Ridge pour affiner les boîtes englobantes (bounding box regression).
    Pour chaque image bien classifiée, génère une région centrale et entraîne
    la régression à prédire les corrections (dx, dy, dw, dh).
    Sauvegarde le modèle dans rcnn_reg_<backbone>.pkl.
    Retourne le régresseur entraîné (ou None si erreur).
    """
    if not TORCH_AVAILABLE or backbone is None or svm is None:
        return None
    try:
        from sklearn.linear_model import Ridge
    except ImportError:
        return None

    X_reg, y_reg = [], []
    for item in db:
        img = charger_image(item['chemin'])
        if img is None:
            continue
        h, w = img.shape[:2]
        # Région centrale (simulation d'une proposition correcte)
        cx, cy = w // 2, h // 2
        rw, rh = min(w, 96), min(h, 96)
        x1 = max(0, cx - rw // 2)
        y1 = max(0, cy - rh // 2)
        crop = img[y1:y1+rh, x1:x1+rw]
        feat = extraire_features_region(crop, backbone)
        if feat is None:
            continue
        # Prédit la classe
        pred = svm.predict([feat])[0]
        if pred == item['classe']:
            # Cible : corrections nulles (région déjà bien localisée)
            # dx, dy, dw, dh = 0, 0, 0, 0 (pas de correction nécessaire)
            X_reg.append(feat)
            y_reg.append([0.0, 0.0, 0.0, 0.0])

    if len(X_reg) < 2:
        return None

    # Régression Ridge multi-sortie : prédit (dx, dy, dw, dh)
    reg = Ridge(alpha=1.0)
    reg.fit(np.array(X_reg), np.array(y_reg))

    # Sauvegarde sur disque
    reg_path = RCNN_REG_PATH.get(backbone_name, RCNN_REG_PATH["VGG16"])
    with open(reg_path, 'wb') as f:
        pickle.dump(reg, f)

    return reg


def charger_regression_bbox(backbone_name="VGG16"):
    """Charge le régresseur de boîtes depuis le disque (None si pas encore entraîné)."""
    reg_path = RCNN_REG_PATH.get(backbone_name, RCNN_REG_PATH["VGG16"])
    if not os.path.exists(reg_path):
        return None
    try:
        with open(reg_path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def nms(boxes, scores, iou_threshold=0.3):
    """
    Suppression des non-maxima (Non-Maximum Suppression).
    Élimine les boîtes englobantes redondantes en ne gardant que
    celle avec le score le plus élevé parmi les boîtes qui se chevauchent trop.

    Paramètres :
      boxes          : liste de (x, y, w, h)
      scores         : liste de scores de confiance (float)
      iou_threshold  : seuil d'IoU au-dessus duquel deux boîtes sont considérées redondantes

    Retourne les indices des boîtes conservées.
    """
    if len(boxes) == 0:
        return []

    boxes_arr = np.array(boxes, dtype=np.float32)
    scores_arr = np.array(scores, dtype=np.float32)

    # Convertit (x, y, w, h) → (x1, y1, x2, y2) pour le calcul d'IoU
    x1 = boxes_arr[:, 0]
    y1 = boxes_arr[:, 1]
    x2 = boxes_arr[:, 0] + boxes_arr[:, 2]
    y2 = boxes_arr[:, 1] + boxes_arr[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    # Trier par score décroissant
    order = scores_arr.argsort()[::-1]
    kept = []

    while len(order) > 0:
        i = order[0]
        kept.append(int(i))
        if len(order) == 1:
            break
        # Calcule l'IoU entre la boîte i et toutes les boîtes restantes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / (union + 1e-10)
        # Garder uniquement les boîtes dont l'IoU avec i est inférieur au seuil
        order = order[1:][iou <= iou_threshold]

    return kept


def page_rcnn():
    """
    Page dédiée au pipeline R-CNN (Region-based CNN) pour la détection d'objets.
    Implémente les 4 étapes classiques : propositions → features → SVM → régression bbox.
    """
    st.header("🎯 Détection d'objets — Pipeline R-CNN")

    # ── Explication du pipeline R-CNN ──────────────────────────────────────────
    st.markdown("""
## Pipeline R-CNN — Vue d'ensemble

Le **R-CNN** (Region-based CNN, Girshick et al. 2014) est une des premières architectures
combinant CNN et détection d'objets. Il fonctionne en **4 étapes successives** :

| Étape | Composant | Description |
|-------|-----------|-------------|
| 1 | **Propositions de régions** | Fenêtre glissante → ~2000 régions candidates |
| 2 | **Extraction de features** | VGG16 (512 dims) ou AlexNet (256 dims) pré-entraîné sur ImageNet |
| 3 | **Classification SVM** | LinearSVC calibré, entraîné sur notre dataset → classe + score |
| 4 | **Régression bbox** | Ridge regression → ajustement (dx, dy, dw, dh) |

Enfin, la **NMS** (Non-Maximum Suppression) élimine les détections redondantes.

> **Note** : Cette implémentation utilise la **fenêtre glissante** (sliding window)
> au lieu de la recherche sélective (selective search) pour la simplicité.
> Le SVM et le régresseur sont entraînés sur **notre propre dataset**.
    """)

    st.markdown("---")

    if not TORCH_AVAILABLE:
        st.error("⚠️ PyTorch n'est pas installé. Installez-le avec : `pip install torch torchvision`")
        return

    try:
        from sklearn.svm import SVC
        from sklearn.linear_model import Ridge
        sklearn_ok = True
    except ImportError:
        sklearn_ok = False
        st.error("⚠️ scikit-learn n'est pas installé. Installez-le avec : `pip install scikit-learn`")
        return

    # ── Choix du backbone ────────────────────────────────────────────────────────
    backbone_name = st.sidebar.selectbox(
        "Backbone CNN",
        ["VGG16", "AlexNet"],
        help="VGG16 : 512 dims, plus précis. AlexNet : 256 dims, plus rapide."
    )

    # ── Étape 1 : Entraînement (si modèles non sauvegardés) ────────────────────
    st.markdown("### Étape 1 — Entraînement du SVM et du régresseur")
    svm_existe = os.path.exists(RCNN_SVM_PATH[backbone_name])
    reg_existe = os.path.exists(RCNN_REG_PATH[backbone_name])

    if svm_existe and reg_existe:
        st.success(f"Modèles R-CNN ({backbone_name}) déjà entraînés et sauvegardés.")
    else:
        st.warning(f"Les modèles R-CNN ({backbone_name}) ne sont pas encore entraînés.")
        if st.button(f"Entraîner le pipeline R-CNN avec {backbone_name} sur notre dataset"):
            if 'db' not in st.session_state:
                st.session_state.db = indexer_base()
            db = st.session_state.db

            if backbone_name == "AlexNet":
                backbone = get_alexnet_backbone()
            else:
                backbone = get_vgg_backbone()
            if backbone is None:
                st.error(f"Impossible de charger le backbone {backbone_name}.")
                return

            st.info(f"Entraînement du SVM avec features {backbone_name}...")
            svm = entrainer_svm_rcnn(db, backbone, backbone_name)
            if svm is None:
                st.error("Échec de l'entraînement du SVM.")
                return

            st.info("Entraînement du régresseur de boîtes englobantes...")
            reg = entrainer_regression_bbox(db, backbone, svm, backbone_name)
            if reg is None:
                st.warning("Régresseur non entraîné (pas assez de données correctement classifiées).")

            st.success(f"Pipeline R-CNN ({backbone_name}) entraîné et sauvegardé !")
            st.rerun()
        return

    # ── Étape 2 : Upload de l'image ─────────────────────────────────────────────
    st.markdown("### Étape 2 — Choisir une image à analyser")
    fichier_rcnn = st.file_uploader(
        "Charger une image",
        type=['jpg', 'jpeg', 'png', 'bmp', 'gif'],
        key="rcnn_upload"
    )
    if fichier_rcnn is None:
        st.info("Veuillez charger une image pour lancer la détection R-CNN.")
        return

    image = np.array(Image.open(fichier_rcnn))
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    st.image(image, caption="Image chargée", use_container_width=True)

    # ── Étape 3 : Propositions de régions ───────────────────────────────────────
    st.markdown("### Étape 3 — Propositions de régions (fenêtre glissante)")
    stride = st.sidebar.slider("Stride de la fenêtre", 16, 64, 32, 16)
    min_confidence = st.sidebar.slider("Confiance minimale", 0.05, 0.99, 0.10, 0.05)

    propositions = extraire_propositions(image, stride=stride)
    st.info(f"{len(propositions)} propositions générées avec stride={stride}.")

    # Afficher l'image avec quelques boîtes de propositions
    img_props = image.copy()
    # N'afficher qu'un échantillon de 50 boîtes pour la lisibilité
    sample_props = propositions[::max(1, len(propositions) // 50)]
    for (x, y, w, h) in sample_props:
        cv2.rectangle(img_props, (x, y), (x + w, y + h), (0, 200, 255), 1)
    st.image(img_props, caption=f"Aperçu des propositions ({len(sample_props)} sur {len(propositions)} affichées)",
             use_container_width=True)

    # ── Étape 4 : Classification et régression ──────────────────────────────────
    st.markdown("### Étape 4 — Classification SVM + Régression bbox + NMS")
    if st.button("Lancer la détection R-CNN", use_container_width=True):
        if backbone_name == "AlexNet":
            backbone = get_alexnet_backbone()
        else:
            backbone = get_vgg_backbone()
        svm = charger_svm_rcnn(backbone_name)
        reg = charger_regression_bbox(backbone_name)

        if backbone is None or svm is None:
            st.error("Impossible de charger les modèles R-CNN.")
            return

        # Classification de chaque proposition
        detections = []   # (x, y, w, h, classe, score)
        features_list = []  # pour la visualisation

        bar = st.progress(0)
        status = st.empty()
        n = len(propositions)
        for i, (x, y, w, h) in enumerate(propositions):
            crop = image[y:y + h, x:x + w]
            feat = extraire_features_region(crop, backbone)
            if feat is None:
                bar.progress((i + 1) / n)
                continue

            # Prédiction SVM avec probabilités
            proba = svm.predict_proba([feat])[0]
            classe_idx = np.argmax(proba)
            score = proba[classe_idx]
            classe = svm.classes_[classe_idx]

            if score >= min_confidence:
                # Régression bbox (ajustement de la boîte si régresseur disponible)
                bx, by, bw, bh = x, y, w, h
                if reg is not None:
                    try:
                        delta = reg.predict([feat])[0]
                        bx = int(x + delta[0] * w)
                        by = int(y + delta[1] * h)
                        bw = int(w * np.exp(delta[2]))
                        bh = int(h * np.exp(delta[3]))
                        # Clamp aux dimensions de l'image
                        bx = max(0, min(bx, image.shape[1] - 1))
                        by = max(0, min(by, image.shape[0] - 1))
                        bw = min(bw, image.shape[1] - bx)
                        bh = min(bh, image.shape[0] - by)
                    except Exception:
                        pass

                detections.append((bx, by, bw, bh, classe, score))
                if len(features_list) < 5:
                    features_list.append((feat, classe, score))

            bar.progress((i + 1) / n)
            if (i + 1) % 100 == 0:
                status.text(f"Analyse des propositions : {i+1}/{n} ({len(detections)} détections)")

        bar.empty()
        status.empty()
        st.info(f"{len(detections)} détections avant NMS (confiance ≥ {min_confidence:.0%})")

        if not detections:
            st.warning("Aucun objet détecté avec la confiance minimale choisie. Essayez de réduire le seuil.")
            return

        # Application de la NMS
        boxes_det = [(d[0], d[1], d[2], d[3]) for d in detections]
        scores_det = [d[5] for d in detections]
        kept_idx = nms(boxes_det, scores_det, iou_threshold=0.3)
        detections_finales = [detections[i] for i in kept_idx]
        st.success(f"{len(detections_finales)} objets détectés après NMS.")

        # ── Affichage des détections finales ────────────────────────────────────
        # Palette de couleurs pour les classes
        np.random.seed(42)
        classes_uniques = list(set(d[4] for d in detections_finales))
        palette = {cls: tuple(np.random.randint(50, 220, 3).tolist()) for cls in classes_uniques}

        img_result = image.copy()
        for (bx, by, bw, bh, classe, score) in detections_finales:
            color = palette.get(classe, (255, 100, 0))
            cv2.rectangle(img_result, (bx, by), (bx + bw, by + bh), color, 2)
            label_txt = f"{classe} {score:.0%}"
            # Fond du texte pour la lisibilité
            (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img_result, (bx, by - th - 4), (bx + tw + 2, by), color, -1)
            cv2.putText(img_result, label_txt, (bx + 1, by - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Image originale")
            st.image(image, use_container_width=True)
        with col_b:
            st.subheader("Détections R-CNN (après NMS)")
            st.image(img_result, use_container_width=True)

        # ── Tableau des détections ───────────────────────────────────────────────
        st.markdown("### Récapitulatif des détections")
        rows = []
        for i, (bx, by, bw, bh, classe, score) in enumerate(detections_finales):
            rows.append({
                "N°": i + 1,
                "Classe": classe,
                "Confiance": f"{score:.1%}",
                "Position (x,y)": f"({bx}, {by})",
                "Taille (w×h)": f"{bw}×{bh}"
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── Visualisation du vecteur de features de la meilleure détection ──────
        st.markdown("### Visualisation du vecteur de features (meilleure région)")
        if features_list:
            feat_top, classe_top, score_top = max(features_list, key=lambda x: x[2])
            st.info(f"Classe : **{classe_top}** — Confiance : **{score_top:.1%}** — Dimension : {len(feat_top)} valeurs")
            # Affiche un sous-ensemble du vecteur (les 256 premières valeurs)
            feat_display = feat_top[:256]
            fig_feat, ax_feat = plt.subplots(figsize=(10, 2))
            ax_feat.plot(feat_display, linewidth=0.5, color='steelblue')
            ax_feat.fill_between(range(len(feat_display)), feat_display, alpha=0.3, color='steelblue')
            ax_feat.set_title(f"Features VGG16 — {classe_top} (256 premières valeurs sur {len(feat_top)})")
            ax_feat.set_xlabel("Dimension")
            ax_feat.set_ylabel("Valeur")
            ax_feat.spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_feat)
            plt.close(fig_feat)


def main():
    st.sidebar.title("Navigation")
    choix_page = st.sidebar.radio(
        "Aller vers",
        ["🔍 Recherche d'images", "📦 Clustering d'Images", "🛠️ Traitements (Module 912)", "✂️ Segmentation", "📊 Évaluation", "🎯 Détection R-CNN"]
    )
    st.sidebar.markdown("---")

    if choix_page == "🔍 Recherche d'images":
        st.title("🔍 Mini-moteur de recherche d'images")
        page_recherche()
    elif choix_page == "📦 Clustering d'Images":
        st.title("📦 Clustering d'Images (Non-supervisé)")
        page_clustering()
    elif choix_page == "🛠️ Traitements (Module 912)":
        st.title("🛠️ Traitements & Analyse d'image")
        page_traitement()
    elif choix_page == "✂️ Segmentation":
        st.title("✂️ Segmentation d'images")
        page_segmentation()
    elif choix_page == "📊 Évaluation":
        st.title("📊 Évaluation des performances")
        page_evaluation()
    elif choix_page == "🎯 Détection R-CNN":
        st.title("🎯 Détection d'objets — Pipeline R-CNN")
        page_rcnn()


if __name__ == "__main__":
    main()





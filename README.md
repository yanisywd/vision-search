# VisionSearch — Application CBIR

Application complète de **recherche d'images par le contenu** (Content-Based Image Retrieval), développée avec Streamlit.

**Dataset** : 200 images, 40 classes, 5 images/classe.

---

## Lancement

```bash
pip install -r requirements.txt
streamlit run app1.py
```

> **SAM** : si `segment-anything` échoue via pip :
> ```bash
> pip install git+https://github.com/facebookresearch/segment-anything.git
> ```
> Télécharger le checkpoint ViT-B (~375 MB) depuis Meta et le placer à la racine :
> `sam_vit_b_01ec64.pth`

---

## Ce qui est implémenté

### 1. Descripteurs de couleur

| Descripteur | Vecteur | Description |
|-------------|---------|-------------|
| **Histogramme RGB** | 96 dims | 3 canaux × 32 bins, normalisé |
| **Histogramme Pondéré Saturation** | 96 dims | Canal H pondéré par la saturation (HSV), pixels peu saturés masqués |
| **Histogramme Cumulé** | 96 dims | CDF du RGB — invariant aux changements de luminosité |
| **Histogramme + Entropie** | 97 dims | RGB hist + entropie de Shannon (–∑ p·log₂p) |
| **CDS** | 24 dims | Moyenne, std, skewness, kurtosis sur RGB + HSV |
| **DCD** | 20 dims | K-Means k=5 sur les pixels → 5 couleurs dominantes + proportions |
| **CCD** | 64 dims | Cohérence spatiale : pixels en grandes régions (coherent) vs isolés (incoherent) |

---

### 2. Descripteurs de texture

| Descripteur | Vecteur | Description |
|-------------|---------|-------------|
| **GLCM** | 6 dims | Matrice de co-occurrence (4 angles) → énergie, entropie, contraste, IDM, dissimilarité, homogénéité |
| **LBP** | 26 dims | Patterns binaires locaux, cercle rayon=3, n=24 voisins, patterns uniformes |
| **LBP par blocs** | 416 dims | Grille 4×4 spatiale — LBP indépendant par bloc |
| **Statistiques de base** | 14 dims | 8 stats niveaux de gris + moyenne/std par canal RGB |
| **Statistiques complètes** | 22 dims | Idem + skewness + kurtosis par canal |
| **Statistiques + Entropie** | 15 dims | Stats de base + entropie de Shannon |

---

### 3. Descripteurs de forme (gradient)

Tous → histogramme 64 valeurs (32 magnitudes + 32 angles orientés)

| Descripteur | Noyaux | Particularité |
|-------------|--------|---------------|
| **Sobel** | 3×3 | Standard, détection horizontale/verticale |
| **Prewitt** | 3×3 | Moins de lissage que Sobel |
| **Roberts** | 2×2 diagonaux | Plus rapide, noyaux croisés 45°/135° |
| **HOG pondéré** | ~8100 dims | Chaque pixel vote avec sa magnitude, blocs 16×16, normalisation L2 |
| **HOG non-pondéré** | ~8100 dims | Vote binaire (magnitude ignorée), robuste aux variations d'intensité |
| **HOG par blocs** | 144 dims | Grille 4×4 × 9 bins directionnels |

Visualisation de la direction du gradient : **quiver plot matplotlib** (flèches colorées par magnitude).

---

### 4. Descripteurs CNN / ANN

| Descripteur | Vecteur | Architecture |
|-------------|---------|--------------|
| **ResNet18** | 512 dims | Pré-entraîné ImageNet, classificateur retiré → AdaptiveAvgPool |
| **MobileNetV2** | 1280 dims | Backbone léger (dépthwise separable convolutions), pré-entraîné ImageNet |
| **Autoencoder FC** | 128 dims | FC : 12288→512→256→**128**→256→512→12288 — entraîné sur le dataset (30 epochs, MSE) |
| **ConvAutoencoder** | 128 dims | Conv encoder (3→16→32→64, MaxPool) → **128** → ConvTranspose decoder — 50 epochs sur le dataset |

Les autoencoders sont utilisés (non supervisé) car 5 images/classe est insuffisant pour entraîner un CNN classifieur.

---

### 5. Métriques de distance

| Distance | Formule |
|----------|---------|
| **Euclidienne** | √(∑(v₁−v₂)²) |
| **Manhattan** | ∑\|v₁−v₂\| |
| **Cosinus** | 1 − (v₁·v₂) / (‖v₁‖·‖v₂‖) |

---

### 6. Segmentation

#### Seuillage global
Manuel, Otsu, Médiane, Moyenne (min+max)/2, P-tile

#### Seuillage local
Moyenne locale, Médiane locale, Min-Max local, Niblack, Sauvola, Wolf

#### K-Means
K couleurs dominantes (RGB ou HSV), K-Means++, 100 itérations

#### Deep Learning

| Modèle | Architecture | Particularité |
|--------|-------------|---------------|
| **DeepLabV3** | ResNet101 + Atrous convolution | Pré-entraîné PASCAL VOC, 21 classes |
| **SegNet** | Encoder VGG16-like + MaxUnpool avec indices | Frontières nettes grâce aux indices de pooling |
| **UNet** | Encoder-Decoder + **skip connections** | Concatène features sémantiques + spatiales |
| **PSPNet** | Backbone + **Pyramid Pooling Module** | 4 échelles (1×1, 2×2, 3×3, 6×6) → contexte multi-échelle |

> SegNet, UNet et PSPNet ne sont pas entraînés sur ce dataset : ils nécessitent des annotations pixel-level (masques de segmentation) qui ne sont pas disponibles. Ils utilisent leurs poids pré-entraînés.

#### SAM — Segment Anything Model (Meta)
- Modèle Vision Transformer ViT-B
- Segmentation universelle, sans classes fixes
- `SamAutomaticMaskGenerator` : grille 32×32 points, iou_thresh=0.88, stability=0.95
- Labeling optionnel des régions par **ResNet50** (catégories ImageNet)

---

### 7. Détection — Pipeline R-CNN

Implémentation complète du pipeline R-CNN (Girshick et al., 2014) en 5 étapes :

```
Image → Sliding Window → CNN Backbone → SVM → Régression bbox → NMS → Détections
```

| Étape | Détail |
|-------|--------|
| **Propositions** | Fenêtre glissante, 4 tailles (64→192 px), stride configurable, max 2000 régions |
| **Backbone VGG16** | Features 512 dims via AdaptiveAvgPool (au lieu de 25088) |
| **Backbone AlexNet** | Features 256 dims via AdaptiveAvgPool — plus rapide |
| **SVM** | StandardScaler → LinearSVC(C=0.1) → CalibratedClassifierCV pour les probabilités |
| **Entraînement SVM** | Image complète + 3 crops aléatoires par image → ~800 échantillons |
| **Régression bbox** | Ridge(α=1.0), prédit (dx, dy, dw, dh) en proportions |
| **NMS** | Suppression si IoU > 0.3, garde le score le plus élevé |

Sélection du backbone (VGG16 ou AlexNet) via la sidebar — modèles sauvegardés séparément.

---

### 8. Évaluation des descripteurs

- **AP** : précision moyenne aux rangs où un résultat correct apparaît
- **MAP** : moyenne des AP sur toutes les images requêtes
- **MAP@10** : MAP limité aux 10 premiers résultats
- **69 combinaisons** : 23 descripteurs × 3 distances
- Résultats sauvegardés dans `resultats_evaluation.csv`, calcul incrémental
- Visualisations : barres horizontales MAP, MAP vs MAP@10, effet de la distance par descripteur

---

### 9. Pages de l'application

| Page | Contenu |
|------|---------|
| **Recherche** | Upload image → top-k résultats similaires, visualisation du vecteur descripteur |
| **Clustering** | K-Means sur tous les descripteurs → groupes visuels |
| **Traitements** | Espaces couleur, analyse de gradient (quiver plot), convolution, restauration |
| **Segmentation** | Toutes les méthodes ci-dessus |
| **Évaluation** | MAP / MAP@10 avec graphiques comparatifs pour tous les descripteurs |
| **Détection R-CNN** | Pipeline complet VGG16 / AlexNet |

---

## Structure du projet

```
├── app1.py                        # Application principale
├── evaluate_all.py                # Script d'évaluation autonome
├── requirements.txt
├── BD_images_prepared/            # Dataset (40 classes, 200 images)
├── conv_autoencoder_dataset.pth   # ConvAutoencoder entraîné sur le dataset
├── rcnn_svm_alexnet.pkl           # SVM R-CNN AlexNet (pré-entraîné)
├── rcnn_reg_alexnet.pkl           # Régresseur bbox AlexNet
└── resultats_evaluation.csv       # MAP / MAP@10 de tous les descripteurs
```

> `sam_vit_b_01ec64.pth` (375 MB) non inclus — à télécharger depuis Meta et placer à la racine.

# Application — Recherche d'Images par le Contenu (CBIR)

Application Streamlit complète de traitement et d'analyse d'images.

---

## Installation

### 1. Créer un environnement virtuel (recommandé)
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

> **SAM** : si `segment-anything` échoue via pip, utiliser :
> ```bash
> pip install git+https://github.com/facebookresearch/segment-anything.git
> ```

---

## Lancement

```bash
streamlit run app1.py
```

---

## Structure du projet

```
app_yahiaoui_prof/
├── app1.py                        # Application principale
├── evaluate_all.py                # Script d'évaluation autonome
├── requirements.txt               # Dépendances Python
│
├── BD_images_prepared/            # Dataset (40 classes, 5 images/classe = 200 images)
│   ├── Ananas_8/
│   ├── Chats_13/
│   └── ...
│
├── conv_autoencoder_dataset.pth   # ConvAutoencoder entraîné sur le dataset
├── sam_vit_b_01ec64.pth           # Checkpoint SAM ViT-B (Meta)
│
├── rcnn_svm_alexnet.pkl           # SVM R-CNN backbone AlexNet (pré-entraîné)
├── rcnn_reg_alexnet.pkl           # Régresseur bbox AlexNet (pré-entraîné)
└── resultats_evaluation.csv       # Résultats MAP / MAP@10 de tous les descripteurs
```

---

## Première utilisation — À faire une seule fois

### Pipeline R-CNN VGG16
Le modèle AlexNet est déjà prêt. Pour entraîner **VGG16** :
1. Aller dans **"Détection R-CNN"**
2. Sélectionner **VGG16** dans la sidebar
3. Cliquer **"Entraîner le pipeline R-CNN"**
(L'entraînement prend ~5 min, le modèle est ensuite sauvegardé)

---

## Fonctionnalités

| Section | Description |
|---------|-------------|
| Recherche par similarité | 23 descripteurs (couleur, texture, forme, CNN, ANN) × 3 distances |
| Analyse de gradient | Sobel, Prewitt, Roberts — quiver plot de direction |
| Segmentation | K-Means, Watershed, DeepLabV3, SegNet, UNet, PSPNet, **SAM** |
| Détection R-CNN | Pipeline complet : sliding window → VGG16/AlexNet → SVM → NMS |
| Évaluation | MAP et MAP@10 pour tous les descripteurs, graphiques comparatifs |

---

## Notes

- **Dataset** : 200 images, 40 classes, 5 images par classe
- **SAM** utilise le checkpoint ViT-B (~375 MB) — déjà inclus
- Les modèles CNN (ResNet18, VGG16, AlexNet) sont téléchargés automatiquement depuis PyTorch Hub au premier lancement (~500 MB, nécessite internet)
- CPU suffit pour toutes les fonctionnalités (GPU accélère le R-CNN et SAM)

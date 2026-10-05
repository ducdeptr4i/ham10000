# SkinScope — HAM10000 Dermoscopy Demo

A Streamlit research demo that classifies dermoscopy images into the seven HAM10000 classes using ResNet50, ViT-B/16, and a weighted hybrid classifier.

## Features

- Checks image dimensions and flags likely dark or blank borders before analysis. It never crops the uploaded image automatically.
- Shows the two highest-ranked classes. The second result is an alternative class ranking, not proof of a second coexisting disease.
- Displays a Grad-CAM overlay from the ResNet50 branch to illustrate image regions that influence the top-class score. It is not a lesion segmentation mask.
- Shows the model's raw softmax scores. They are not calibrated clinical probabilities.

## Run locally

Use Python 3.11 or 3.12 and create an environment inside this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run src\demo_app.py
```

The application expects these checkpoints in `outputs/`:

- `best_resnet50.pth`
- `best_vit_b16.pth`
- `best_hybrid_weighted.pth`

The model checkpoints are tracked with Git LFS. The original `data/images/` collection and cached hybrid features are excluded from Git because they are large and are not needed to run inference.

## Deploy with Streamlit Community Cloud

1. Push this repository, including its Git LFS objects, to GitHub.
2. In Streamlit Community Cloud, create an app from that GitHub repository.
3. Set the main file path to `src/demo_app.py` and select Python 3.11 or 3.12.
4. Deploy. The checkpoints must be available in `outputs/` after the cloud clone; verify that the host retrieves Git LFS files rather than pointer files.

The ViT checkpoint is large, and CPU-only hosts may take longer to load and analyze an image. Hosting limits and LFS transfer quotas may affect public deployment.

## Retrain

Training scripts are in `src/`. To train the hybrid classifier, first train the CNN and ViT, then regenerate cached features, and finally train `train_hybrid_weighted.py`. Do not reuse cached features after replacing either backbone checkpoint.

## Intended use

This is an educational/research demonstration. It is not a medical device and must not be used to diagnose or treat a person.

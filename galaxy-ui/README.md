# Galaxy Morphology AI Classifier

An end-to-end computer vision project that classifies galaxy images as **Elliptical** or **Spiral** using a ResNet-18 CNN and Grad-CAM interpretability. The web app combines a Gradio backend on Hugging Face Spaces with a React frontend deployed on Vercel.

**Live Demo:** [https://computer-vision-galaxy-classificati.vercel.app/](https://computer-vision-galaxy-classificati.vercel.app/)

**Hugging Face:** [https://huggingface.co/spaces/SharmilNK/Galaxy-Morphology-Classification/tree/main](https://huggingface.co/spaces/SharmilNK/Galaxy-Morphology-Classification/tree/main)

---

## Overview

Galaxy morphology classification is a foundational task in observational astronomy
that supports large-scale studies of galaxy formation and evolution. In this work, we study binary galaxy morphology classification (elliptical vs. spiral) using images from the Galaxy Zoo dataset, with an emphasis on robustness, calibration, and methodological
transparency.

We evaluate three modeling paradigms: a naive majority-class baseline, a classical
machine learning pipeline using hand-crafted image features with hyperparameter
tuning, and deep learning models based on transfer learning with convolutional
neural networks.

## Features

- **Binary Classification** — Elliptical vs Spiral galaxies
- **Grad-CAM Visualization** — Highlights regions the model focuses on for each prediction
- **Transfer Learning** — Pre-trained ResNet-18 fine-tuned on Galaxy Zoo data
- **Modern Stack** — Hugging Face Spaces (backend), React + Vite (UI), Vercel (hosting)

---

## Architecture

```

| Component | Technology |
|-----------|------------|
| Backend   | Python, Gradio, PyTorch, ResNet-18 |
| Frontend  | React, Vite, Gradio Client |
| Model Host | Hugging Face Spaces |
| UI Host   | Vercel |

---

## Project Structure

```
ComputerVision-Galaxy-Classification/
galaxy-ui/ 
├── main.py                         # main script to run user interface
|──  models
|    |── baseline_galaxy_model.pth  # Baseline model 
     ├── robust_galaxy_model.pth    # Trained ResNet-18 weights 
├── setup.ipynb                     # Training notebook to set up project (get data, build features, train model)
├── requirements.txt                # Python dependencies
│── src/                            # React UI
│   │   ├── App.jsx                 # Main UI + Gradio API integration
│   │   └── App.css                 # Styles
│   ├── public/                     # Static assets (landing.jpg, astro.jpg)
│   ├── vercel.json                 # Vercel config
│   └── package.json
└── README.md
```

---

## How It Works

### Model

- **Architecture:** ResNet-18 (ImageNet pretrained)
- **Output:** 2 classes — Elliptical (0), Spiral (1)
- **Input:** 224×224 RGB images
- **Fine-tuning:** Last residual block (`layer4`) unfrozen; rest frozen

### Grad-CAM

- Uses the last convolutional layer (`layer4`) of ResNet-18
- Overlays a heatmap on the input image to show which regions drive the prediction

---

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js 18+ (for the React UI)
- PyTorch (CPU or CUDA)

### 1. Clone & Backend Setup

```bash
cd ComputerVision-Galaxy-Classification
pip install -r requirements.txt
```

Place your trained model `robust_galaxy_model.pth` in this directory.

### 2. Run Gradio Locally

```bash
python main.py
```

The Gradio API will be at `http://localhost:7860`. The `/predict` endpoint accepts an image and returns Grad-CAM output plus the classification result.

### 3. Run the React UI

```bash
cd galaxy-ui
npm install
npm run dev
```

Set `VITE_GRADIO_URL` in `.env` to your Gradio URL (e.g. `http://localhost:7860` for local, or your Hugging Face Space URL for production):

```
VITE_GRADIO_URL=http://localhost:7860
```

---

## Training (setup.ipynb)

The model is trained in `setup.ipynb` using:

- **Dataset:** Galaxy Zoo – The Galaxy Challenge (Kaggle)
- **Labels:** Training solutions (`training_solutions_rev1.csv`)
- **Morphology mapping:** Galaxy Zoo responses → Elliptical vs Spiral
- **Augmentations:** Resize, random crop, horizontal flip
- **Split:** Train / Val / Test with stratification

Run the notebook on Google Colab or a local Jupyter environment with a GPU for faster training.

---

## Deployment

### Backend (Hugging Face Spaces)

1. Create a new Gradio Space on [Hugging Face Spaces](https://huggingface.co/spaces)
2. Upload `main.py`, `requirements.txt`, `robust_galaxy_model.pth`, `landing.jpg`, `astro.jpg`
3. Configure the Space to run `main.py`

### Frontend (Vercel)

1. Set `Root Directory` to `galaxy-ui`
2. Build command: `npm run build`
3. Output directory: `dist`
4. Add `VITE_GRADIO_URL` pointing to your Hugging Face Space

See `galaxy-ui/DEPLOYMENT.md` for step-by-step instructions.

---

## API

The Gradio app exposes a `/predict` endpoint.

**Input:** Image (file or base64)

**Output:** `[gradcam_image, result_text]`

- `gradcam_image` — Heatmap overlay
- `result_text` — `Predicted Class: Elliptical|Spiral\nProbability: X.XX%`

---

## Science Context

Galaxy morphology classification helps astrophysicists:

- Analyze large catalogs from Hubble and James Webb Space Telescopes
- Understand galaxy formation and evolution
- Map large-scale structure and constrain dark energy
- Automate classification for surveys (e.g. Vera C. Rubin Observatory LSST)

---

## License

MIT (or as specified for Galaxy Zoo data and pre-trained models).

---

## Acknowledgments

- [Galaxy Zoo – The Galaxy Challenge](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge)
- [PyTorch](https://pytorch.org/) and [TorchVision](https://pytorch.org/vision/)
- [Gradio](https://gradio.app/)
- [Hugging Face Spaces](https://huggingface.co/spaces)



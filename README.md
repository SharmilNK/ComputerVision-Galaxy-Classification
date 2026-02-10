# Galaxy Morphology AI Classifier

An end-to-end computer vision project that classifies galaxy images as **Elliptical** or **Spiral** using a ResNet-18 CNN and Grad-CAM. The app has two parts: a **Gradio backend** (Python + model) and a **React frontend** (web UI). The backend runs on Hugging Face Spaces; the frontend runs on Vercel.

**Live demo:** [https://computer-vision-galaxy-classificati.vercel.app/](https://computer-vision-galaxy-classificati.vercel.app/)

**Backend (Hugging Face):** [https://huggingface.co/spaces/SharmilNK/Galaxy-Morphology-Classification](https://huggingface.co/spaces/SharmilNK/Galaxy-Morphology-Classification)


---

## Requirements

- **Python 3.8+** (for the backend and training).
- **Node.js 18+** (for the React frontend).
- **A Kaggle account** (to download the Galaxy Zoo dataset).
- **A Hugging Face account** (to host the Gradio backend).
- **A GitHub account** (to store your code).
- **A Vercel account** (to host the frontend).

---

## High-Level Overview

1. **Train the model** using the Galaxy Zoo dataset in the Jupyter notebook (`setup.ipynb`). This produces a file `robust_galaxy_model.pth`.
2. **Run the backend locally** with `main.py` and the trained model. This starts a Gradio API (e.g. at `http://localhost:7860`).
3. **Run the frontend locally** with the React app in `galaxy-ui`. It talks to the Gradio API via a URL you set.
4. **Deploy the backend** to Hugging Face Spaces so the API is available on the internet.
5. **Deploy the frontend** to Vercel so the UI is available on the internet and points to your Hugging Face API.

The frontend (Vercel) only sends images to the backend (Hugging Face) and shows the result. The model and `main.py` stay on the backend; they are not uploaded to Vercel.

---

## Project Structure

```
ComputerVision-Galaxy-Classification/             
|             
├── main.py                        # Gradio backend (loads model, runs prediction + Grad-CAM)
├── requirements.txt               # Python dependencies for main.py
├── setup.py                       # download data, train model, save .pth
├── models/
|   ├── svm_model.joblib           # Optional classical ml model -SVM
│   ├── baseline_galaxy_model.pth  # Optional baseline model
│   └── robust_galaxy_model.pth    # Trained ResNet-18 weights (you create this)
├── galaxy-ui/                     # React frontend (what you deploy to Vercel)
│   ├── src/
│   │   ├── App.jsx                # Main UI + calls Gradio API
│   │   └── App.css
│   ├── public/                    # Images (e.g. landing.jpg, astro.jpg)
│   ├── package.json
│   ├── vercel.json                # Vercel build config
│   └── ...
└── README.md
|── notebooks
│   ├── setup.ipynb               # Notebook: exploratory analysis        
```

---

## Step 1: Get the Data and Train the Model

The model is trained in the Jupyter notebook `setup.ipynb` using the **Galaxy Zoo – The Galaxy Challenge** dataset from Kaggle.

1. **Create a Kaggle account** at [kaggle.com](https://www.kaggle.com) if you do not have one.
2. **Get your Kaggle API credentials:**
   - Go to Kaggle → Profile → **Account**.
   - In the "API" section, click **Create New Token**. This downloads a file `kaggle.json`.
3. **Open `setup.ipynb`** in Jupyter or Google Colab.
4. **Set up Kaggle in the notebook:**
   - In the notebook there is a cell that writes `kaggle.json` (e.g. under `/root/.kaggle` on Colab). Replace the `username` and `key` in that cell with your own from `kaggle.json`. Do not share this file or commit it to Git.
5. **Run the notebook from top to bottom.** It will:
   - Install `kaggle` and download the competition data.
   - Unzip and prepare the data.
   - Build labels (e.g. Elliptical vs Spiral from Galaxy Zoo responses).
   - Train a ResNet-18 model and save it (e.g. as `robust_galaxy_model.pth`).
6. **Download the trained model** from Colab and place it in your project:
   - Put `robust_galaxy_model.pth` in the **project root** or inside the **`models/`** folder. The script `main.py` looks in both places.

---

## Step 2: Run the Backend Locally

The backend is the Gradio app in `main.py`. It loads the model and exposes an API that accepts an image and returns the class (Elliptical/Spiral) and a Grad-CAM visualization.

1. **Open a terminal** and go to the project root
2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   ```
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   If your `requirements.txt` is very large (e.g. from a full environment export), you can install only what is needed for `main.py`, for example:
   ```bash
   pip install gradio torch torchvision pillow numpy opencv-python
   ```
4. **Start the Gradio app:**
   ```bash
   python main.py
   ```
5. When it finishes loading, open a browser and go to **http://localhost:7860**. You should see the Gradio interface. You can upload a galaxy image and get a prediction and Grad-CAM heatmap.

Keep this terminal open while you use the app. Press `Ctrl+C` to stop the server.

---

## Step 3: Run the Frontend Locally

The frontend is a React app in the `galaxy-ui` folder. It uses the Gradio API to send images and show results. For local use, point it to your local backend.

1. **Open a new terminal** and go to the frontend folder:
   ```bash
   cd galaxy-ui
   ```
2. **Install Node dependencies:**
   ```bash
   npm install
   ```
3. **Tell the frontend where the backend is.** Create a file named `.env` inside `galaxy-ui` with:
   ```
   VITE_GRADIO_URL=http://localhost:7860
   ```
   This makes the React app call your local Gradio server. If your backend runs on another machine, use that URL and port instead.
4. **Start the development server:**
   ```bash
   npm run dev
   ```
5. Open the URL shown in the terminal (e.g. **http://localhost:5173**). You should see the Galaxy Morphology AI Classifier UI. Upload an image; the frontend will send it to the Gradio backend and display the result.

Make sure the backend (`python main.py`) is still running in the other terminal.

---

## Step 4: Deploy the Backend to Hugging Face Spaces

So that the frontend can use your model from the internet (and so you do not need to run `main.py` on your own machine), you host the Gradio app on Hugging Face Spaces.

1. **Create a Hugging Face account** at [huggingface.co](https://huggingface.co) if you do not have one.
2. **Create a new Space:**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces).
   - Click **Create new Space**.
   - Choose a name (e.g. `Galaxy-Morphology-Classification`).
   - Select **Gradio** as the SDK.
   - Choose **Public** if you want the app to be reachable by anyone. Click **Create Space**.
3. **Upload the files the Space needs:**
   - **`main.py`** — Your Gradio script. On Hugging Face the app entrypoint is often `app.py`. If the Space expects `app.py`, you can rename `main.py` to `app.py` before uploading, or configure the Space to run `main.py` if the UI allows it.
   - **`requirements.txt`** — A file that lists only what is needed for `main.py` (e.g. `gradio`, `torch`, `torchvision`, `pillow`, `numpy`, `opencv-python`). Do not upload a huge environment dump; keep it minimal so the Space can build quickly.
   - **`robust_galaxy_model.pth`** — The trained model. Upload it to the root of the Space (same level as `app.py` or `main.py`). If your script looks for the model in a `models/` folder, create that folder in the Space and put the `.pth` file there.
   - **Optional:** `landing.jpg`, `astro.jpg` or other images if your app or README use them.
4. **Configure the Space** so it runs your app:
   - If the file is named `app.py`, Hugging Face usually runs it by default.
   - If you keep the name `main.py`, set the Space’s "App file" (or similar) to `main.py` in the Space settings.
5. Wait for the Space to **build and run**. When it is ready, you get a URL like `https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space`. Open it and test: upload a galaxy image and check that prediction and Grad-CAM work.

This URL is your **backend API URL**. The frontend will use it in the next step.

---

## Step 5: Deploy the Frontend to Vercel

The frontend is only the React app in `galaxy-ui`. Vercel does not run Python or the model; it only serves the built website, which calls your Hugging Face Space in the browser.

1. **Put your code on GitHub:**
   - Create a new repository on GitHub (e.g. `ComputerVision-Galaxy-Classification`).
   - In your project root, initialise Git if needed, add all files, commit, and push. You can include the whole project (including `main.py` and `galaxy-ui`) or only what you need for deployment. For Vercel you **must** have the **`galaxy-ui`** folder in the repo.
   - **Do not** commit `kaggle.json`, `.env` (with secrets), or the large `.pth` file if you prefer to keep the repo small; the model is already on Hugging Face. For Vercel, only the contents of `galaxy-ui` (and the repo structure that contains it) matter.
2. **Connect the repo to Vercel:**
   - Go to [vercel.com](https://vercel.com) and sign in (e.g. with GitHub).
   - Click **Add New** → **Project** and import your GitHub repository.
3. **Configure the project:**
   - **Root Directory:** Click **Edit** and set it to **`galaxy-ui`**. This tells Vercel that the app to build is inside that folder.
   - **Build Command:** `npm run build` (usually auto-detected).
   - **Output Directory:** `dist` (usually auto-detected for Vite).
4. **Set the backend URL:**
   - In the same Vercel project, go to **Settings** → **Environment Variables**.
   - Add a variable:
     - **Name:** `VITE_GRADIO_URL`
     - **Value:** Your Hugging Face Space URL (e.g. `https://YOUR_USERNAME-Galaxy-Morphology-Classification.hf.space`).
   - Save. Apply it to **Production** (and **Preview** if you want).
5. **Deploy:** Trigger a new deployment (e.g. **Deploy** from the Vercel dashboard). When the build finishes, Vercel gives you a URL (e.g. `https://your-project.vercel.app`). Open it: the UI should load and use your Hugging Face Space as the backend.

If something fails, check the **Build Logs** and **Runtime Logs** in Vercel, and make sure `VITE_GRADIO_URL` is set and that your Space is **Running** on Hugging Face.

---

## Quick Reference: Running Locally Only

- **Backend:** From project root run `pip install -r requirements.txt` then `python main.py`. Open http://localhost:7860.
- **Frontend:** From `galaxy-ui` run `npm install`, add `.env` with `VITE_GRADIO_URL=http://localhost:7860`, then `npm run dev`. Open the URL shown (e.g. http://localhost:5173).

---

## How the Model and Grad-CAM Work

- **Model:** ResNet-18, pretrained on ImageNet, with the last layer replaced for 2 classes (Elliptical, Spiral). Input: 224×224 RGB images. The last residual block is fine-tuned; the rest is frozen.
- **Grad-CAM:** Uses the last convolutional layer of ResNet-18 to produce a heatmap on the image, showing which regions the model used for the prediction.

---

## API (Gradio Backend)

The Gradio app exposes a prediction endpoint.

- **Input:** An image (file or base64).
- **Output:** A list that includes:
  - A Grad-CAM overlay image.
  - Text with the predicted class (Elliptical or Spiral) and probability.

---

## License and Data

MIT (or as specified for Galaxy Zoo data and any third-party models). When using Galaxy Zoo data, follow Kaggle and Galaxy Zoo terms of use.

---

## Acknowledgments

- [Galaxy Zoo – The Galaxy Challenge](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge) (Kaggle)
- [PyTorch](https://pytorch.org/), [TorchVision](https://pytorch.org/vision/), [Gradio](https://gradio.app/), [Hugging Face Spaces](https://huggingface.co/spaces)

## Worked on by
Tiffany Degbotse
Sharmil Nanjappa

## AI Citation
AI was used to generate syntax for part of the code, CSS and Java scripts for UI, process steps to link Gradio with React and deploy on Vercel. AI was used to understand and learn only, the concept and ideation was original.

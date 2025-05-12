# Sign Language Recognition - BISINDO (Words Only)

This project focuses on recognizing Indonesian Sign Language (BISINDO) specifically for a subset of **words** using a machine learning approach. The model is trained to classify 6 specific signs: `Bertemu`, `Halo`, `Kamu`, `Perkenalkan`, `Saya`, and `Senang`.

Dataset used: [TALKEE BISINDO Sign Language Dataset](https://www.kaggle.com/datasets/niputukarismadewi/talkee-bisindo-sign-language-dataset)

## 📁 Project Structure

```
sign-language-recognition/
│
├── data/
│   └── raw/
│       └── BISINDO/
│           ├── bertemu/
│           ├── halo/
│           ├── kamu/
│           ├── perkenalkan/
│           ├── saya/
│           └── senang/
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb   # Data preprocessing and preparation
│   ├── 02_model_training.ipynb       # Model architecture and training loop
│   └── 03_evaluation.ipynb           # Evaluation and visualization
│
├── scripts/
│   ├── preprocess.py        # CLI version of preprocessing pipeline
│   ├── train.py             # CLI version of training script
│   ├── evaluate.py          # CLI version of evaluation script
│   └── infer.py             # CLI version of model inference
│
├── utils/
│   ├── dataset_loader.py    # Custom Dataset loader
│   ├── metrics.py           # Metrics calculation (accuracy, precision, etc.)
│   └── visualize.py         # Helper functions for plotting and visualization
│
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd sign-language-recognition
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download & Extract Dataset

* Download from [Kaggle - TALKEE BISINDO](https://www.kaggle.com/datasets/niputukarismadewi/talkee-bisindo-sign-language-dataset)
* Only include the following **word** folders:

```
Bertemu/, Halo/, Kamu/, Perkenalkan/, Saya/, Senang/
```

* Place inside the following structure:

```
data/raw/BISINDO/<word-name>/
```

### 4. Run the Jupyter Notebooks

You can explore the full ML pipeline through the following notebooks:

* **01\_data\_preprocessing.ipynb** – Load, preprocess, and prepare the dataset.
* **02\_model\_training.ipynb** – Define and train the CNN-LSTM model.
* **03\_evaluation.ipynb** – Evaluate the model using classification metrics and confusion matrix.

> 📌 Notebooks are located in the `notebooks/` directory and have been tested end-to-end.

## 📌 Notes

* This version currently focuses on using **notebooks** as the main working pipeline.
* The `scripts/` directory contains equivalent CLI-based Python scripts which can be adapted for automation or production.
* Only a limited set of 6 BISINDO words is used in this project.
* This setup is part of a backend model for a larger sign language recognition application (e.g. integrated with Flutter frontend).

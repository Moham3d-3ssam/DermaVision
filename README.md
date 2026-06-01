# DermaVision

DermaVision is a deep-learning project for **skin disease image classification**.  
It provides an easy-to-use interface where users can upload a skin image and receive a predicted disease class with confidence.

## Live Demo

Try the app here:  
**https://huggingface.co/spaces/Moham3d-3saam/dermavision**

## 🎯 Project Overview

This project implements a hybrid approach to skin disease classification by combining:
- **Deep Learning (CNN)**: For image-based feature extraction and classification
- **Ensemble Learning**: Multiple classifiers (XGBoost, LightGBM, Random Forest) for improved accuracy
- **Feature Fusion**: Integration of image features with patient metadata for enhanced predictions

The system can classify six types of skin conditions:
| Code | Disease Name | Description |
|------|-------------|-------------|
| ACK | Actinic Keratosis | Pre-cancerous skin growth caused by sun damage |
| BCC | Basal Cell Carcinoma | Most common type of skin cancer |
| MEL | Melanoma | Most dangerous type of skin cancer |
| NEV | Nevus | Common mole, usually benign |
| SCC | Squamous Cell Carcinoma | Second most common skin cancer |
| SEK | Seborrheic Keratosis | Non-cancerous skin growth |

## Project Goals

- Build a practical skin-condition classifier using image data.
- Provide a simple interactive interface for predictions.
- Support quick local setup and deployment.
- Keep the repository lightweight while preserving data structure.

## Main Features

- Upload image support (`jpg`, `jpeg`, `png`)
- Automatic preprocessing (resize + normalization)
- Top predicted class with confidence score
- Helpful treatment-resource links based on predicted class
- Ready-to-run Streamlit application

## Tech Stack

- **Python**
- **TensorFlow / Keras**
- **TensorFlow Hub**
- **NumPy**
- **Pillow (PIL)**
- **Streamlit**

## Repository Structure

```text
DermaVision/
├── app/
│   ├── templates/
│       └── index.html
│   └── app.py
├── data/       # We added only one image in each data folder because the full dataset is very large.
│   ├── flip_blur_imgs/
│   ├── flip_noise_color_imgs/
│   ├── main_imgs/
│   ├── resized_imgs/
│   └── rotate_zoom_imgs/
├── notebook/
│   ├── 01. Data_Augmentation.ipynb
│   ├── 02. Data_Processing.ipynb
│   ├── 03. Increase_Data.ipynb
│   ├── 04. Train_CNN_TF_Model.ipynb
│   ├── 05. Train_ANN_Model.ipynb
│   └── 06. Train_Final_Ensemble_Model.ipynb
├── saved_models/
├── Procfile
└── requirements.txt
```

## Local Setup

### 1) Clone the repository

```bash
git clone https://github.com/Moham3d-3ssam/DermaVision.git
cd DermaVision
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run the app

```bash
streamlit run app/app.py
```

## Usage

1. Open the Streamlit app in your browser.
2. Upload a skin image.
3. Click **Classify**.
4. View predicted class and confidence.
5. Check the treatment-resource link shown by the app.

## Important Note

This project is intended for **educational and research purposes**.  
Predictions are model-generated and should not replace professional medical diagnosis.

## Author

- **Moham3d-3ssam**

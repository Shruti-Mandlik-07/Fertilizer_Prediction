# Terra — Fertilizer Recommendation System

A machine learning-based web application that recommends a fertilizer based on soil nutrient levels, environmental conditions, soil type, and crop type.

**Live Demo:** [https://terra-fertilizer.onrender.com](https://terra-fertilizer.onrender.com)

**GitHub Repository:** [Fertilizer_Prediction](https://github.com/Shruti-Mandlik-07/Fertilizer_Prediction)

---

## Overview

Terra is a web application developed to demonstrate how machine learning can support fertilizer recommendation using agricultural input data. Users enter soil nutrient values and environmental conditions, select their soil and crop types, and receive a fertilizer prediction from a trained classification model.

The application combines a trained machine learning pipeline with an interactive Streamlit interface.

## Features

- **Fertilizer Prediction:** Predicts a fertilizer category from user-provided inputs.
- **Soil Nutrient Analysis:** Accepts Nitrogen, Phosphorous, and Potassium values.
- **Environmental Inputs:** Includes temperature, humidity, and soil moisture.
- **Crop and Soil Selection:** Supports categorical inputs for soil type and crop type.
- **Interactive Interface:** Provides an easy-to-use web interface built with Streamlit.
- **Web Deployment:** Hosted online using Render.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| Pandas | Data handling |
| Scikit-learn | Machine learning pipeline and prediction |
| Joblib | Saving and loading the trained model |
| Streamlit | Interactive web interface |
| Render | Cloud deployment |
| Git and GitHub | Version control and source code hosting |

## Input Features

The application uses the following features:

| Feature | Description |
|---|---|
| Nitrogen | Nitrogen nutrient value |
| Phosphorous | Phosphorous nutrient value |
| Potassium | Potassium nutrient value |
| Temperature | Environmental temperature |
| Humidity | Environmental humidity |
| Moisture | Soil moisture value |
| Soil Type | Category of soil |
| Crop Type | Type of crop being cultivated |

## Machine Learning Approach

The project uses a supervised classification approach to predict the fertilizer label.

The workflow includes:

1. Loading and preparing the fertilizer dataset.
2. Separating numerical and categorical features.
3. Preprocessing numerical features using standardization.
4. Encoding categorical features using one-hot encoding.
5. Training and evaluating classification models.
6. Saving the selected model pipeline using Joblib.
7. Loading the trained pipeline in the Streamlit application.
8. Generating predictions from user inputs.

The saved model pipeline includes the preprocessing steps required to transform the inputs before prediction.

## Project Structure

```text
Fertilizer_Prediction/
│
├── app.py
├── fertilizer_model.pkl
├── model_info.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Shruti-Mandlik-07/Fertilizer_Prediction.git
cd Fertilizer_Prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will display a local URL in your terminal, usually `http://localhost:8501`.

## Deployment

The application is deployed on Render and connected to the GitHub repository.

**Live application:** [Terra Fertilizer Recommendation System](https://terra-fertilizer.onrender.com)

## Disclaimer

Terra is an educational machine learning project. Its predictions reflect patterns learned from the training dataset and should not be treated as verified agricultural recommendations. Actual fertilizer requirements depend on laboratory soil testing, crop variety, growth stage, local conditions, and professional agronomic guidance.

## Author

**Shruti Mandlik**

- GitHub: [Shruti-Mandlik-07](https://github.com/Shruti-Mandlik-07)

---

*Built with Python, machine learning, and Streamlit to explore data-driven fertilizer recommendation.*

Commit `fertilizer_model.pkl` only if you are comfortable storing the trained model in the repository. If you prefer not to store it in Git, use a private artifact download/storage flow and adapt `app.py` to load it securely.

## Important
The model predicts fertilizer labels from the dataset. Its output is not validated agronomic advice; verify fertilizer choice and application rates with soil testing and local agricultural guidance.

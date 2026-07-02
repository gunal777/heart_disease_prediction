# ❤️ Heart Disease Prediction

A machine learning web application that predicts the likelihood of heart disease based on patient health parameters. The project is built using **Python**, **Flask**, and **Scikit-learn**, with a **Random Forest Classifier** for prediction.

## Features

* Predicts heart disease risk from medical inputs.
* User-friendly web interface built with Flask.
* Real-time predictions using a trained ML model.
* Deployed on Render.

## Tech Stack

* Python
* Flask
* Scikit-learn
* Pandas
* NumPy
* Joblib
* HTML & CSS

## Project Structure

```text
heart_disease_prediction/
│
├── app.py
├── train_model.py
├── model.pkl
├── heart.csv
├── requirements.txt
├── templates/
├── static/
└── .gitignore
```

## Installation

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Train the model (if `model.pkl` is not available):

```bash
python train_model.py
```

5. Run the application:

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

## Dataset

This project uses the UCI Heart Disease dataset containing patient medical information such as age, cholesterol, blood pressure, ECG results, and other clinical parameters.

## Disclaimer

This application is intended for educational purposes only and should not be used as a substitute for professional medical advice or diagnosis.

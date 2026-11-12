# Titanic Survival Prediction App

A Flask-based machine learning project for predicting whether a Titanic passenger survived based on passenger information such as class, sex, age, family size, and ticket fare. The app includes a dashboard for model analysis, an experiment page for tuning the model, and a form for predicting survival for new passengers.

## Project overview

This project trains a Random Forest classifier on the Titanic dataset and compares:

- a baseline feature set
- an enhanced feature set with engineered features like `FamilySize`, `IsAlone`, and passenger title extraction

It provides an interactive web interface to:

- upload CSV/XLSX datasets
- browse stored datasets
- explore original and processed data
- run model training with configurable hyperparameters
- inspect performance metrics
- predict survival outcomes for custom passenger records

## Tech stack

- Python
- Flask
- Pandas
- NumPy
- scikit-learn
- HTML/CSS/JavaScript with Tailwind CSS

## Project structure

```text
Titanic-ML/
├── main.py                  # Flask app + ML pipeline
├── titanic.csv              # Local Titanic dataset used by default
├── datasets/                # Uploaded CSV/XLSX datasets
├── templates/
│   ├── dashboard.html
│   ├── index.html
│   └── insert.html
├── README.md
└── .git/                    # repository metadata (if present)
```

## Features

- Titanic dataset loading from local CSV or fallback remote source
- Data preprocessing and feature engineering
- Random Forest model training and evaluation
- Model performance metrics:
  - accuracy
  - precision
  - recall
  - F1-score
  - confusion matrix
- Upload support for CSV and XLSX files
- Dashboard for comparing baseline vs enhanced feature sets
- Passenger form for single-record survival prediction
- Simple experiment interface for tuning model parameters

## How it works

The app loads Titanic data and prepares two variations:

1. Baseline model
   - uses core passenger fields like class, age, fare, and sex

2. Featured model
   - adds engineered features such as:
     - `FamilySize`
     - `IsAlone`
     - extracted passenger titles from names

The model then splits the data into training and testing sets, trains a Random Forest classifier, and evaluates it on the test set.

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install flask pandas numpy scikit-learn
```

3. Run the application:

```bash
python main.py
```

4. Open the app in your browser:

- Dashboard: http://localhost:5000/dashboard
- Experiment page: http://localhost:5000/experiment
- Insert data page: http://localhost:5000/insert

## Model configuration

The app allows experimentation with the following Random Forest parameters:

- `use_features` toggle for feature generation
- `n_estimators`
- `max_depth`
- `min_samples_leaf`

These values are handled in the `/run_ml` endpoint in `main.py`.

## Dataset notes

The application looks for a local `titanic.csv` file first. If it is not present, it falls back to the standard Titanic public dataset URL. Uploaded files are saved under `datasets/` and can be listed/deleted from within the web app.

## Example usage

- Visit the dashboard to inspect the raw dataset and engineered features.
- Use the experiment page to train different Random Forest configurations.
- Use the insert form to simulate a new passenger and get a survival prediction.

## License

This project is intended for learning and experimentation with machine learning on the Titanic dataset.

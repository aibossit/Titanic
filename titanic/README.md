# Titanic Survival Prediction

ML project for the Kaggle Titanic competition.

## Goal

Predict passenger survival using reproducible preprocessing, cross-validation, model training, and submission generation.

## Project Structure

```text
titanic/
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing_features.ipynb
│   ├── 03_model_selection.ipynb
│   └── 04_final_submission.ipynb
├── src/
│   ├── config.py
│   ├── features.py
│   ├── predict.py
│   └── train.py
├── requirements.txt
└── submission.csv
```

## Workflow

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python run.py
```

Run one step only:

```bash
python run.py features
python run.py train
python run.py predict
```

The old module-style commands also work:

```bash
python -m src.features
python -m src.train
python -m src.predict
```

## Current Pipeline

- Feature engineering is implemented in `src/features.py`.
- Train/test preprocessing is fitted from train statistics and applied consistently to test.
- Model and validation settings live in `config/config.yaml`.
- Training saves the fitted artifact to `models/titanic_model.joblib`.
- Cross-validation metrics are saved to `reports/metrics.json`.
- Prediction writes `submission.csv`.
- The Python scripts are intentionally simple and notebook-like, so the project is easy to read and modify.

## Models

The scripted pipeline compares Logistic Regression, Random Forest, NeuralNetwork, LightGBM and CatBoost. It selects the model with the best cross-validation accuracy and saves it to `models/titanic_model.joblib`.

## Validation

- StratifiedKFold
- 5 folds
- Accuracy, Precision, Recall, F1, ROC AUC metrics

## Current Results

Best scripted model: CatBoost.

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8058 | 0.7538 | 0.7337 | 0.7424 | 0.8595 |
| Random Forest | 0.8350 | 0.8187 | 0.7338 | 0.7730 | 0.8764 |
| CatBoost | 0.8440 | 0.8287 | 0.7485 | 0.7865 | 0.8831 |
| Neural Network | 0.8159 | 0.7684 | 0.7484 | 0.757 | 0.8681 |
| LightGBM | 0.826 | 0.779 | 0.763 | 0.771 | 0.8823 |

| Metric | CV Mean | CV Std |
|---|---:|---:|
| Accuracy | 0.8440 | 0.0128 |
| Precision | 0.8287 | 0.0221 |
| Recall | 0.7485 | 0.0174 |
| F1 | 0.7865 | 0.0172 |
| ROC AUC | 0.8831 | 0.0201 |

## Notes

Notebooks are kept for exploration and reporting. The `src` package is the reproducible project entrypoint.

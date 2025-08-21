# IBM Employee Attrition Analysis

A ready-to-run project for exploring, modeling, and predicting employee attrition on the IBM HR Analytics schema.

## ✨ What’s inside
- `app.py` — Streamlit dashboard for EDA, training, and scoring
- `src/preprocess.py` — feature prep utilities (one-hot encoding, splits)
- `src/train.py` — CLI training script (Random Forest / Logistic Regression)
- `data/sample_ibm_attrition.csv` — small synthetic dataset with the common IBM columns
- `models/` — saved model artifacts (created after training)
- `requirements.txt` — dependencies

## 🚀 Quickstart (local)
```bash
# create env
python -m venv venv
venv\Scripts\activate     # Windows
# source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
streamlit run app.py
```

## 🧪 Train via CLI
```bash
python -m src.train --data data/sample_ibm_attrition.csv --model rf --output models/attrition_model.joblib
```

## 📄 Notes
- The bundled dataset is **synthetic** for demo. You can replace it with the IBM HR Analytics dataset from Kaggle (same columns).
- Target column: `Attrition` with values `Yes` / `No`.
- Categorical & numeric columns are defined in `src/preprocess.py` — update if your dataset differs.

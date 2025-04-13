# 🤖 Robotic Grasp Stability Predictor

This Streamlit app predicts robotic grasp stability using sensor input data and machine learning models trained on the Shadow Robot Hand dataset.

---

## 🚀 Features

- Predict grasp stability using:
  - Linear Regression
  - Polynomial Regression
  - XGBoost
  - Neural Network (Keras)
- Real-time interactive sliders for 24 joint features (position, velocity, effort)
- Visual stability gauge and model performance comparison
- **No large dataset needed** — all models and scalers are pre-trained and lightweight

---

## 📦 Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🧠 Files You Need in the Same Folder

- `linear_reg.pkl` — Trained Linear Regression model
- `poly_transform.pkl` — Polynomial feature transformer
- `poly_reg_model.pkl` — Polynomial Regression model
- `xgb_model.json` — XGBoost model
- `neural_net_model.h5` — Keras Neural Network model
- `scaler.pkl` — Fitted StandardScaler object
- `columns.pkl` — Ordered list of input features
- `feature_ranges.pkl` — Min/max values for each input feature (for slider setup)

---

## ▶️ Running the App

From the terminal:

```bash
streamlit run app.py
```

---

## ☁️ Deployment Notes

- Compatible with Mac (Apple Silicon) and cloud platforms (Streamlit Cloud, GCP, etc.)
- Does **not** use the full training dataset during runtime — only essential files
- Neural net model is loaded with `compile=False` and used with CPU only (for maximum compatibility)

---

## 💬 Contact

For questions or issues, contact Lakshay Chawla or open an issue on the repo.

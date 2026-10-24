# STOCK Predicter - Mini Project

## Stock Market Direction Prediction using Machine Learning

**Dataset:** NIFTYJR (NSE Daily Indices) 1997–2009 (3000 records)  
**Models:** Decision Tree, Naive Bayes, Logistic Regression, SVM, Random Forest + Voting Ensemble  
**Best Model:** Decision Tree (55.84% Accuracy)

---

## Project Structure

```
STOCK_Predicter/
├── app.py                          # Streamlit Web Application
├── train_improved.py               # Improved training script (recommended)
├── train_model.py                  # Original training script
├── NSE_Daily_indices_data_3000.csv # Dataset
├── best_model.pkl                  # Saved best model (Decision Tree)
├── scaler.pkl                      # Feature scaler
├── selector.pkl                    # Feature selector
├── all_features.pkl                # All feature names
├── selected_features.pkl           # Selected feature names
├── model_comparison.csv            # Model performance comparison
└── README.md                       # This file
```

---

## How to Run

### 1. Install Requirements
```bash
pip install streamlit scikit-learn joblib pandas numpy matplotlib
```

### 2. (Optional) Re-train the model
```bash
python train_improved.py
```

### 3. Launch the Web App
```bash
streamlit run app.py
```

The browser will open automatically at `http://localhost:8501`

---

## Features

- Dataset Analysis & Preprocessing
- Feature Engineering (RSI, MACD, Bollinger Bands, Stochastic, etc.)
- Feature Selection
- 5 Supervised Models + Voting Ensemble
- Model Comparison (Accuracy, Precision, Recall, F1)
- Streamlit Web GUI for prediction
- Sustainable Development Goals: SDG 8 & SDG 9

---

## Model Results

| Model               | Accuracy |
|---------------------|----------|
| Decision Tree       | 55.84%   |
| Random Forest       | 55.16%   |
| Voting Ensemble     | 54.65%   |
| Naive Bayes         | 54.82%   |
| Logistic Regression | 53.47%   |
| SVM                 | 53.47%   |

---

## Sustainable Development Goals

- **SDG 8**: Decent Work and Economic Growth
- **SDG 9**: Industry, Innovation and Infrastructure

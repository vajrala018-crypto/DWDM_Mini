# ============================================================
# STOCK Predicter - Premium Dark UI (Hackathon Ready)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="STOCK Predicter", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

# ===================== PREMIUM DARK CSS =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default streamlit branding */
#MainMenu, footer, header {visibility: hidden;}
.stDeployButton {display: none;}

/* Main background */
.stApp {
    background: linear-gradient(145deg, #0b0f19 0%, #111827 50%, #0f172a 100%);
    color: #e2e8f0;
}

/* Hero */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem 1rem;
}
.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero p {
    color: #94a3b8;
    font-size: 1.15rem;
}

/* Cards */
.card {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}
.card-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 0.8rem;
}

/* Metric pills */
.metric-pill {
    background: linear-gradient(135deg, #1e293b, #334155);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    border: 1px solid rgba(56, 189, 248, 0.2);
}
.metric-pill .val {
    font-size: 1.6rem;
    font-weight: 700;
    color: #38bdf8;
}
.metric-pill .lbl {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 0.2rem;
}

/* Step indicator */
.steps {
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    margin: 1.5rem 0 2rem 0;
    flex-wrap: wrap;
}
.step {
    padding: 0.5rem 1.2rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    background: #1e293b;
    color: #64748b;
    border: 1px solid #334155;
}
.step.active {
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
    border: none;
}

/* Buttons */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #2563eb, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    height: 3rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: transform 0.15s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(37, 99, 235, 0.35);
}

/* Inputs */
.stNumberInput input, .stTextInput input, .stSelectbox div {
    background: #1e293b !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
    border: 1px solid #334155 !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* Success / Error boxes */
.stSuccess, .stError, .stInfo, .stWarning {
    border-radius: 12px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    background: #1e293b;
    border-radius: 10px;
    color: #94a3b8;
    padding: 0.5rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #2563eb, #7c3aed) !important;
    color: white !important;
}

/* Progress */
.stProgress > div > div {
    background: linear-gradient(90deg, #2563eb, #7c3aed);
}

section.main > div {padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)

# ===================== SESSION STATE =====================
for key in ["df", "results", "best_model", "scaler", "features", "best_name", "best_acc", "step"]:
    if key not in st.session_state:
        st.session_state[key] = None
if st.session_state.step is None:
    st.session_state.step = 1

# ===================== HERO =====================
st.markdown("""
<div class="hero">
    <h1>STOCK Predicter</h1>
    <p>AI-powered next-day stock direction forecasting</p>
</div>
""", unsafe_allow_html=True)

# Step indicator
steps_html = ""
labels = ["Upload", "Explore", "Train", "Predict"]
for i, lab in enumerate(labels, 1):
    cls = "step active" if st.session_state.step == i else "step"
    steps_html += f'<span class="{cls}">{i}. {lab}</span>'
st.markdown(f'<div class="steps">{steps_html}</div>', unsafe_allow_html=True)

# Navigation buttons
nav1, nav2, nav3, nav4 = st.columns(4)
with nav1:
    if st.button("① Upload", use_container_width=True):
        st.session_state.step = 1
        st.rerun()
with nav2:
    if st.button("② Explore", use_container_width=True):
        st.session_state.step = 2
        st.rerun()
with nav3:
    if st.button("③ Train", use_container_width=True):
        st.session_state.step = 3
        st.rerun()
with nav4:
    if st.button("④ Predict", use_container_width=True):
        st.session_state.step = 4
        st.rerun()

st.markdown("---")

# ===================== STEP 1: UPLOAD =====================
if st.session_state.step == 1:
    st.markdown("### Upload Dataset")
    st.caption("CSV must contain columns: OPEN, HIGH, LOW, CLOSE")

    col_u1, col_u2 = st.columns([2, 1])
    with col_u1:
        uploaded = st.file_uploader("Drop your CSV file here", type=["csv"], label_visibility="collapsed")
    with col_u2:
        use_sample = st.checkbox("Use sample NIFTYJR data", value=False)

    if use_sample:
        try:
            df = pd.read_csv("NSE_Daily_indices_data_3000.csv")
            st.session_state.df = df
            st.success(f"Sample loaded — {df.shape[0]} rows × {df.shape[1]} columns")
        except Exception as e:
            st.error(str(e))

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.session_state.df = df
            st.success(f"Uploaded — {df.shape[0]} rows × {df.shape[1]} columns")
        except Exception as e:
            st.error(str(e))

    if st.session_state.df is not None:
        df = st.session_state.df
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="metric-pill"><div class="val">{df.shape[0]}</div><div class="lbl">Rows</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-pill"><div class="val">{df.shape[1]}</div><div class="lbl">Columns</div></div>', unsafe_allow_html=True)
        with m3:
            miss = int(df.isnull().sum().sum())
            st.markdown(f'<div class="metric-pill"><div class="val">{miss}</div><div class="lbl">Missing</div></div>', unsafe_allow_html=True)

        st.markdown("#### Preview")
        st.dataframe(df.head(8), use_container_width=True)

        if st.button("Continue to Explore →", type="primary"):
            st.session_state.step = 2
            st.rerun()

# ===================== STEP 2: EXPLORE =====================
elif st.session_state.step == 2:
    st.markdown("### Explore Data")

    if st.session_state.df is None:
        st.warning("Upload a dataset first.")
    else:
        df = st.session_state.df
        t1, t2, t3 = st.tabs(["Full Table", "Single Row", "Stats"])

        with t1:
            st.dataframe(df, use_container_width=True, height=380)

        with t2:
            idx = st.number_input("Row index", 0, len(df)-1, 0)
            st.dataframe(df.iloc[[idx]], use_container_width=True)

        with t3:
            st.dataframe(df.describe(), use_container_width=True)

        if st.button("Continue to Train →", type="primary"):
            st.session_state.step = 3
            st.rerun()

# ===================== STEP 3: TRAIN =====================
elif st.session_state.step == 3:
    st.markdown("### Train & Evaluate Models")

    if st.session_state.df is None:
        st.warning("Upload a dataset first.")
    else:
        df_raw = st.session_state.df.copy()
        col_map = {c.lower(): c for c in df_raw.columns}
        need = ["open", "high", "low", "close"]
        miss = [x for x in need if x not in col_map]

        if miss:
            st.error(f"Missing required columns: {miss}")
        else:
            if st.button("Start Training", type="primary"):
                with st.spinner("Engineering features & training 5 models..."):
                    df = df_raw.rename(columns={
                        col_map["open"]: "OPEN", col_map["high"]: "HIGH",
                        col_map["low"]: "LOW", col_map["close"]: "CLOSE"
                    })[["OPEN", "HIGH", "LOW", "CLOSE"]].dropna().reset_index(drop=True)

                    df["Target"] = (df["CLOSE"].shift(-1) > df["CLOSE"]).astype(int)
                    df["Return"] = df["CLOSE"].pct_change()
                    df["SMA_5"] = df["CLOSE"].rolling(5).mean()
                    df["SMA_10"] = df["CLOSE"].rolling(10).mean()
                    df["SMA_20"] = df["CLOSE"].rolling(20).mean()
                    df["Volatility"] = df["Return"].rolling(10).std()
                    df["Momentum"] = df["CLOSE"] - df["CLOSE"].shift(5)
                    df["HL_Spread"] = df["HIGH"] - df["LOW"]
                    df["OC_Spread"] = df["CLOSE"] - df["OPEN"]
                    for lag in [1, 2, 3]:
                        df[f"Return_lag{lag}"] = df["Return"].shift(lag)
                    delta = df["CLOSE"].diff()
                    gain = delta.where(delta > 0, 0).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    df["RSI"] = 100 - (100 / (1 + gain / (loss + 1e-8)))
                    df = df.dropna().reset_index(drop=True)

                    feats = [c for c in df.columns if c != "Target"]
                    X, y = df[feats], df["Target"]
                    split = int(0.8 * len(X))
                    Xtr, Xte = X.iloc[:split], X.iloc[split:]
                    ytr, yte = y.iloc[:split], y.iloc[split:]

                    scaler = StandardScaler()
                    Xtr_s = scaler.fit_transform(Xtr)
                    Xte_s = scaler.transform(Xte)

                    models = {
                        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
                        "Naive Bayes": GaussianNB(),
                        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
                        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42),
                    }

                    rows, best_acc, best_m, best_n = [], 0, None, ""
                    for name, m in models.items():
                        m.fit(Xtr_s, ytr)
                        pred = m.predict(Xte_s)
                        acc = accuracy_score(yte, pred)
                        rows.append({
                            "Model": name,
                            "Accuracy %": round(acc * 100, 2),
                            "Precision %": round(precision_score(yte, pred, zero_division=0) * 100, 2),
                            "Recall %": round(recall_score(yte, pred, zero_division=0) * 100, 2),
                            "F1 %": round(f1_score(yte, pred, zero_division=0) * 100, 2),
                        })
                        if acc > best_acc:
                            best_acc, best_m, best_n = acc, m, name

                    st.session_state.results = pd.DataFrame(rows)
                    st.session_state.best_model = best_m
                    st.session_state.scaler = scaler
                    st.session_state.features = feats
                    st.session_state.best_name = best_n
                    st.session_state.best_acc = best_acc

                st.success(f"Done — Best: **{best_n}** ({best_acc*100:.2f}%)")

            if st.session_state.results is not None:
                st.markdown("#### Results")
                st.dataframe(st.session_state.results, use_container_width=True, hide_index=True)

                fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#0f172a")
                ax.set_facecolor("#0f172a")
                res = st.session_state.results
                colors = ["#38bdf8" if m == st.session_state.best_name else "#475569" for m in res["Model"]]
                ax.barh(res["Model"], res["Accuracy %"], color=colors)
                ax.set_xlabel("Accuracy %", color="#94a3b8")
                ax.tick_params(colors="#94a3b8")
                ax.set_xlim(0, 100)
                for spine in ax.spines.values():
                    spine.set_color("#334155")
                st.pyplot(fig)

                st.info(f"Best Model → **{st.session_state.best_name}** | Accuracy **{st.session_state.best_acc*100:.2f}%**")

                if st.button("Continue to Predict →", type="primary"):
                    st.session_state.step = 4
                    st.rerun()

# ===================== STEP 4: PREDICT =====================
elif st.session_state.step == 4:
    st.markdown("### Predict Direction")

    if st.session_state.best_model is None:
        st.warning("Train models first (Step 3).")
    else:
        st.success(f"Model: **{st.session_state.best_name}** ({st.session_state.best_acc*100:.2f}%)")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            open_p = st.number_input("Open", value=4500.0)
        with c2:
            high_p = st.number_input("High", value=4550.0)
        with c3:
            low_p = st.number_input("Low", value=4480.0)
        with c4:
            close_p = st.number_input("Close", value=4520.0)

        c5, c6 = st.columns(2)
        with c5:
            rsi_p = st.number_input("RSI", value=55.0, min_value=0.0, max_value=100.0)
        with c6:
            vol_p = st.number_input("Volatility", value=0.015, format="%.4f")

        if st.button("Get Prediction", type="primary"):
            ret = (close_p - open_p) / open_p if open_p else 0
            feat_map = {
                "OPEN": open_p, "HIGH": high_p, "LOW": low_p, "CLOSE": close_p,
                "Return": ret, "SMA_5": close_p, "SMA_10": close_p * 0.99, "SMA_20": close_p * 0.98,
                "Volatility": vol_p, "Momentum": close_p * 0.01,
                "HL_Spread": high_p - low_p, "OC_Spread": close_p - open_p,
                "Return_lag1": ret * 0.8, "Return_lag2": ret * 0.5, "Return_lag3": ret * 0.3,
                "RSI": rsi_p,
            }
            row = {c: feat_map.get(c, 0) for c in st.session_state.features}
            X = pd.DataFrame([row])[st.session_state.features]
            Xs = st.session_state.scaler.transform(X)
            pred = st.session_state.best_model.predict(Xs)[0]
            proba = st.session_state.best_model.predict_proba(Xs)[0] if hasattr(st.session_state.best_model, "predict_proba") else None

            st.markdown("---")
            if pred == 1:
                st.markdown("""
                <div style="text-align:center;padding:1.5rem;background:rgba(34,197,94,0.15);border-radius:16px;border:1px solid #22c55e;">
                    <div style="font-size:2.5rem;">📈</div>
                    <div style="font-size:1.8rem;font-weight:700;color:#4ade80;">UP</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align:center;padding:1.5rem;background:rgba(239,68,68,0.15);border-radius:16px;border:1px solid #ef4444;">
                    <div style="font-size:2.5rem;">📉</div>
                    <div style="font-size:1.8rem;font-weight:700;color:#f87171;">DOWN</div>
                </div>
                """, unsafe_allow_html=True)

            if proba is not None:
                conf = float(max(proba))
                st.metric("Confidence", f"{conf*100:.1f}%")
                st.progress(conf)

st.markdown("---")
st.caption("STOCK Predicter · SDG 8 & SDG 9 · Mini Project")

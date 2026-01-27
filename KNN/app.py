# app.py

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier


# -----------------------------
# CSS Styling (Beautiful UI)
# -----------------------------
def add_css():
    st.markdown("""
    <style>
    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        font-family: "Segoe UI", sans-serif;
        color: white;
    }

    /* Main Title */
    h1 {
        text-align: center;
        font-size: 42px !important;
        font-weight: 800 !important;
        color: #38bdf8 !important;
        text-shadow: 0px 0px 15px rgba(56, 189, 248, 0.6);
    }

    /* Subheaders */
    h2, h3 {
        color: #f1f5f9 !important;
        font-weight: 700 !important;
    }

    /* Description */
    .description-box {
        text-align: center;
        font-size: 17px;
        color: #cbd5e1;
        background: rgba(255, 255, 255, 0.08);
        padding: 15px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0px 4px 18px rgba(0, 0, 0, 0.35);
    }

    /* ---------------- Sidebar Full Styling Fix ---------------- */

    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.98);
        border-right: 2px solid rgba(56, 189, 248, 0.35);
        padding: 18px;
    }

    /*  Make ALL sidebar normal text brighter */
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    /* Sidebar heading */
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #38bdf8 !important;
        font-size: 20px !important;
        font-weight: 800 !important;
        text-shadow: 0px 0px 12px rgba(56, 189, 248, 0.65);
    }

    /*  Labels (Age, Annual Income, Loan Amount, etc.) */
    section[data-testid="stSidebar"] label {
        color: #f8fafc !important;
        font-weight: 700 !important;
        font-size: 15px !important;
    }

    /* ✅ Slider numbers (like 31, 5) */
    section[data-testid="stSidebar"] div[data-testid="stSlider"] span {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    /* ✅ Input card look */
    section[data-testid="stSidebar"] .stSlider,
    section[data-testid="stSidebar"] .stNumberInput,
    section[data-testid="stSidebar"] .stRadio {
        background: rgba(255, 255, 255, 0.10) !important;
        padding: 12px !important;
        border-radius: 16px !important;
        margin-bottom: 14px !important;
        border: 1px solid rgba(56, 189, 248, 0.35) !important;
        box-shadow: 0px 4px 14px rgba(0,0,0,0.35) !important;
    }

    /* ✅ Number input box styling */
    section[data-testid="stSidebar"] input {
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #0f172a !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        border: 2px solid rgba(56, 189, 248, 0.35) !important;
    }

    /* ✅ Radio buttons text */
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        color: #f8fafc !important;
        font-weight: 700 !important;
    }

    /* Button Style */
    div.stButton > button {
        width: 100%;
        border-radius: 14px;
        background: linear-gradient(90deg, #38bdf8, #06b6d4);
        color: black;
        font-size: 18px;
        font-weight: 700;
        padding: 12px;
        border: none;
        box-shadow: 0px 4px 18px rgba(56, 189, 248, 0.4);
        transition: 0.3s ease-in-out;
    }

    div.stButton > button:hover {
        transform: scale(1.04);
        box-shadow: 0px 0px 25px rgba(56, 189, 248, 0.8);
        background: linear-gradient(90deg, #06b6d4, #38bdf8);
    }

    /* Prediction Results */
    .risk-high {
        text-align: center;
        font-size: 30px;
        font-weight: 800;
        color: #ff4d4d;
        background: rgba(255, 77, 77, 0.15);
        padding: 20px;
        border-radius: 18px;
        border: 2px solid rgba(255, 77, 77, 0.6);
        box-shadow: 0px 4px 20px rgba(255, 77, 77, 0.3);
        margin-top: 20px;
    }

    .risk-low {
        text-align: center;
        font-size: 30px;
        font-weight: 800;
        color: #22c55e;
        background: rgba(34, 197, 94, 0.15);
        padding: 20px;
        border-radius: 18px;
        border: 2px solid rgba(34, 197, 94, 0.6);
        box-shadow: 0px 4px 20px rgba(34, 197, 94, 0.3);
        margin-top: 20px;
    }

    /* Metric Boxes */
    .metric-box {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0px 4px 18px rgba(0, 0, 0, 0.35);
        margin-top: 10px;
    }

    /* Dataframe table */
    .stDataFrame {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0px 4px 18px rgba(0, 0, 0, 0.35);
    }
    </style>
    """, unsafe_allow_html=True)


# -----------------------------
# Page Config + CSS
# -----------------------------
st.set_page_config(page_title="Customer Risk Prediction System (KNN)", page_icon="💳", layout="centered")
add_css()

# -----------------------------
# App Header
# -----------------------------
st.title("Customer Risk Prediction System (KNN)")
st.markdown(
    "<div class='description-box'>This system predicts customer risk by comparing them with similar customers.</div>",
    unsafe_allow_html=True
)

# -----------------------------
# Load dataset
st.subheader("📂 Upload Dataset")
uploaded_file = st.file_uploader("Upload credit_risk_dataset.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully!")
else:
    st.warning("⚠️ Please upload the dataset to continue.")
    st.stop()


# -----------------------------
# Validate dataset columns
# -----------------------------
required_cols = [
    "person_age",
    "person_income",
    "loan_amnt",
    "cb_person_cred_hist_length",
    "cb_person_default_on_file",
    "loan_status"
]

missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    st.error(f"❌ Dataset is missing these columns: {missing_cols}")
    st.stop()

# -----------------------------
# Sidebar – User Input
# -----------------------------
st.sidebar.header("🧾 User Input")

age = st.sidebar.slider("Age", min_value=18, max_value=80, value=25)

income = st.sidebar.number_input("Annual Income", min_value=0, value=50000, step=1000)

loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, value=10000, step=500)

credit_history = st.sidebar.radio("Credit History (Default on File?)", ["No", "Yes"])
credit_history_val = 1 if credit_history == "Yes" else 0

k_value = st.sidebar.slider("K Value (Nearest Neighbors)", min_value=1, max_value=15, value=5)

# -----------------------------
# Data Preparation for KNN
# -----------------------------
features = [
    "person_age",
    "person_income",
    "loan_amnt",
    "cb_person_cred_hist_length",
    "cb_person_default_on_file"
]
target = "loan_status"

# Convert Y/N → 1/0
df["cb_person_default_on_file"] = df["cb_person_default_on_file"].map({"Y": 1, "N": 0})

# Drop missing rows
df = df.dropna(subset=[target])

X = df[features]
y = df[target]

# Missing value handling
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN model
knn = KNeighborsClassifier(n_neighbors=k_value)
knn.fit(X_train_scaled, y_train)

# -----------------------------
# Main Prediction Button
# -----------------------------
st.subheader("🎯 Predict Customer Risk")

predict_btn = st.button("Predict Customer Risk")

if predict_btn:
    median_cred_len = float(df["cb_person_cred_hist_length"].median())

    new_customer = np.array([[age, income, loan_amount, median_cred_len, credit_history_val]])

    # preprocess user input
    new_customer_imputed = imputer.transform(new_customer)
    new_customer_scaled = scaler.transform(new_customer_imputed)

    # prediction
    pred = knn.predict(new_customer_scaled)[0]

    # -----------------------------
    # Prediction Output (Center Screen)
    # -----------------------------
    if pred == 1:
        st.markdown("<div class='risk-high'>🔴 High Risk Customer</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='risk-low'>🟢 Low Risk Customer</div>", unsafe_allow_html=True)

    # -----------------------------
    # Nearest Neighbors Explanation
    # -----------------------------
    st.subheader("🔍 Nearest Neighbors Explanation")

    distances, indices = knn.kneighbors(new_customer_scaled)
    neighbor_labels = y_train.iloc[indices[0]].values

    high_risk_count = int(np.sum(neighbor_labels == 1))
    low_risk_count = int(np.sum(neighbor_labels == 0))

    majority_class = "High Risk" if high_risk_count > low_risk_count else "Low Risk"

    st.markdown(f"""
    <div class="metric-box">
        ✅ <b>Number of neighbors considered (K):</b> {k_value}<br>
        ✅ <b>Majority class among neighbors:</b> {majority_class}<br>
        🔴 <b>High Risk Neighbors:</b> {high_risk_count}<br>
        🟢 <b>Low Risk Neighbors:</b> {low_risk_count}
    </div>
    """, unsafe_allow_html=True)

    # Table of nearest customers
    st.markdown("### 📌 Similar Customers Table (Nearest Neighbors)")

    X_train_df = pd.DataFrame(X_train, columns=features)
    neighbors_df = X_train_df.iloc[indices[0]].copy()
    neighbors_df["risk_label(0=Low,1=High)"] = neighbor_labels
    neighbors_df["distance"] = distances[0]

    st.dataframe(neighbors_df)

    # -----------------------------
    # Business Insight Section
    # -----------------------------
    st.subheader("💡 Business Insight")
    st.info("This decision is based on similarity with nearby customers in feature space.")
else:
    st.info("📌 Enter details in the sidebar and click **Predict Customer Risk** to see the result.")


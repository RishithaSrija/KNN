import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor

from sklearn.metrics import r2_score, mean_squared_error

import matplotlib.pyplot as plt


# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="🏠 Smart House Price Predictor – Stacking Model", layout="wide")


# -------------------------------
# BEAUTIFUL PREMIUM CSS
# -------------------------------
st.markdown("""
<style>

/* Google Font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"]{
    font-family:'Poppins',sans-serif;
}

/* ------------------------------------------------ */
/* MAIN APP BACKGROUND */
/* ------------------------------------------------ */

.stApp{
    background:
    linear-gradient(
        135deg,
        #fff8f2 0%,
        #fffdf8 35%,
        #f6f3ff 70%,
        #eefaf2 100%
    );
}

/* ------------------------------------------------ */
/* TEXT */
/* ------------------------------------------------ */

p, span, label{
    color:#4b5563 !important;
}

h1{
    color:#3f3d56 !important;
    font-size:42px !important;
    font-weight:800 !important;
    text-align:center;
}

h2,h3{
    color:#3f3d56 !important;
    font-weight:700 !important;
}

/* ------------------------------------------------ */
/* SIDEBAR */
/* ------------------------------------------------ */

section[data-testid="stSidebar"]{

    background:
    linear-gradient(
        180deg,
        #f6f3ff 0%,
        #fff8f2 50%,
        #eefaf2 100%
    );

    border-right:2px solid #e7dff9;
}

section[data-testid="stSidebar"] *{
    color:#4a4a68 !important;
}

/* Sidebar Header */

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3{

    color:#7c6bcf !important;
    font-weight:800 !important;
}

/* Sidebar Cards */

section[data-testid="stSidebar"] .stSlider,
section[data-testid="stSidebar"] .stNumberInput,
section[data-testid="stSidebar"] .stSelectbox{

    background:rgba(255,255,255,0.75);

    padding:10px;

    border-radius:18px;

    margin-bottom:10px;

    border:1px solid #ece7e1;
}

/* Inputs */

section[data-testid="stSidebar"] input{

    background:#ffffff !important;

    color:#3f3d56 !important;

    border-radius:12px !important;
}

/* ------------------------------------------------ */
/* HERO SECTION */
/* ------------------------------------------------ */

.hero{

    background:
    linear-gradient(
        135deg,
        #ffd6ba,
        #f6c1c1,
        #dccef9
    );

    padding:28px;

    border-radius:28px;

    text-align:center;

    margin-bottom:25px;

    box-shadow:
    0px 10px 30px rgba(0,0,0,0.08);
}

/* ------------------------------------------------ */
/* CARDS */
/* ------------------------------------------------ */

.card{

    background:#ffffff;

    padding:22px;

    border-radius:24px;

    border:1px solid #f0ebe5;

    box-shadow:
    0px 10px 25px rgba(0,0,0,0.05);

    margin-bottom:20px;
}

/* ------------------------------------------------ */
/* KPI SECTION */
/* ------------------------------------------------ */

.kpi-container{

    display:flex;

    flex-wrap:wrap;

    gap:15px;
}

.kpi{

    flex:1;

    min-width:200px;

    border-radius:22px;

    padding:18px;

    border:none;

    box-shadow:
    0px 6px 16px rgba(0,0,0,0.04);
}

.kpi:nth-child(1){
    background:#eefaf2;
}

.kpi:nth-child(2){
    background:#fff1e6;
}

.kpi:nth-child(3){
    background:#f6f3ff;
}

.kpi-title{

    font-size:14px;

    color:#64748b;

    font-weight:700;
}

.kpi-value{

    font-size:28px;

    color:#3f3d56;

    font-weight:900;
}
            
/* Card Content Visibility */

.card,
.card p,
.card li,
.card ul,
.card b,
.card strong,
.card h1,
.card h2,
.card h3{

    color:#374151 !important;
}

/* ------------------------------------------------ */
/* BUTTON */
/* ------------------------------------------------ */

div.stButton > button{

    width:100%;

    background:
    linear-gradient(
        90deg,
        #a8d5ba,
        #dccef9
    );

    color:#3f3d56;

    font-weight:800;

    border:none;

    border-radius:18px;

    padding:14px;

    font-size:17px;

    box-shadow:
    0px 8px 20px rgba(0,0,0,0.08);

    transition:0.3s;
}

div.stButton > button:hover{

    background:
    linear-gradient(
        90deg,
        #95c9ab,
        #cebdf5
    );

    transform:translateY(-2px);

    box-shadow:
    0px 12px 24px rgba(0,0,0,0.10);
}

/* ------------------------------------------------ */
/* PREDICTION BOX */
/* ------------------------------------------------ */

.price-box{

    background:
    linear-gradient(
        135deg,
        #eefaf2,
        #dff5e7
    );

    color:#2f5d50;

    padding:22px;

    border-radius:24px;

    border:2px solid #b7e0c4;

    text-align:center;

    font-size:32px;

    font-weight:900;

    box-shadow:
    0px 8px 22px rgba(0,0,0,0.08);
}

/* ------------------------------------------------ */
/* DATAFRAME */
/* ------------------------------------------------ */

[data-testid="stDataFrame"]{

    border-radius:20px !important;

    overflow:hidden !important;

    border:1px solid #ece7e1 !important;

    box-shadow:
    0px 6px 18px rgba(0,0,0,0.05);
}

/* ------------------------------------------------ */
/* ALERTS */
/* ------------------------------------------------ */

div[data-testid="stAlert"]{

    border-radius:18px;

    border:none;

    box-shadow:
    0px 6px 18px rgba(0,0,0,0.05);
}

/* ------------------------------------------------ */
/* METRICS */
/* ------------------------------------------------ */

[data-testid="metric-container"]{

    background:#ffffff;

    border-radius:18px;

    padding:12px;

    border:1px solid #f0ebe5;
}

/* ------------------------------------------------ */
/* SCROLLBAR */
/* ------------------------------------------------ */

::-webkit-scrollbar{
    width:10px;
}

::-webkit-scrollbar-thumb{
    background:#d8c8f7;
    border-radius:10px;
}

::-webkit-scrollbar-track{
    background:#faf7ff;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD DATASET
# -------------------------------
@st.cache_data
def load_data():
    file_path = "kc_house_data.csv"
    if not os.path.exists(file_path):
        st.error("❌ Dataset file not found! Upload kc_house_data.csv in the same folder as this app file.")
        st.stop()
    return pd.read_csv(file_path)


df = load_data()


# -------------------------------
# TITLE
# -------------------------------
# st.title("🏠 Smart House Price Predictor – Stacking Model")

st.markdown("""
<div class="hero">
<h2>🏡 Smart House Price Predictor</h2>

<h3 style="font-size:17px;color:#4b5563;">
Find the estimated value of your dream home using
a powerful Stacking Ensemble Machine Learning model.
</h3>

</div>
""", unsafe_allow_html=True)
st.divider()


# -------------------------------
# TARGET + PREPROCESSING
# -------------------------------
target_col = "price"

# ✅ Fix date column issue
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%S", errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df.drop(columns=["date"], inplace=True)

# ✅ Drop target + id
X = df.drop(columns=[target_col, "id"])
y = df[target_col]

# ✅ Convert to float32 for memory
X = X.astype(np.float32)
y = y.astype(np.float32)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -------------------------------
# STACKING MODEL (LIGHTWEIGHT)
# -------------------------------
base_models = [
    ("Linear Regression", LinearRegression()),
    ("Decision Tree", DecisionTreeRegressor(random_state=42)),
    ("Random Forest", RandomForestRegressor(
        n_estimators=50,
        random_state=42,
        n_jobs=-1
    ))
]

meta_model = LinearRegression()

stack_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=3,
    passthrough=False,
    n_jobs=1
)

stacking_pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("stack", stack_model)
])

# Train
stacking_pipeline.fit(X_train, y_train)

# Evaluate
y_test_pred = stacking_pipeline.predict(X_test)
r2 = r2_score(y_test, y_test_pred)

mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)


# -------------------------------
# KPI DISPLAY
# -------------------------------
st.markdown(f"""
<div class="card">
    <h3>📌 Model Performance</h3>
    <div class="kpi-container">
        <div class="kpi">
            <div class="kpi-title">✅ R² Score</div>
            <div class="kpi-value">{r2:.3f}</div>
        </div>
        <div class="kpi">
            <div class="kpi-title">📉 RMSE</div>
            <div class="kpi-value">{rmse:,.2f}</div>
        </div>
        <div class="kpi">
            <div class="kpi-title">📦 Dataset Rows</div>
            <div class="kpi-value">{len(df):,}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# -------------------------------
# SIDEBAR INPUTS
# -------------------------------
st.sidebar.header("📝 Enter House Details")

bedrooms = st.sidebar.number_input("Bedrooms", min_value=0, max_value=33, value=3)
bathrooms = st.sidebar.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
sqft_living = st.sidebar.number_input("Sqft Living", min_value=100, value=1800)
sqft_lot = st.sidebar.number_input("Sqft Lot", min_value=500, value=5000)
floors = st.sidebar.number_input("Floors", min_value=1.0, max_value=4.0, value=1.0, step=0.5)

waterfront = st.sidebar.selectbox("Waterfront", options=[0, 1])
view = st.sidebar.selectbox("View", options=[0, 1, 2, 3, 4])
condition = st.sidebar.selectbox("Condition (1-5)", options=[1, 2, 3, 4, 5], index=2)
grade = st.sidebar.selectbox("Grade (1-13)", options=list(range(1, 14)), index=6)

sqft_above = st.sidebar.number_input("Sqft Above", min_value=100, value=1500)
sqft_basement = st.sidebar.number_input("Sqft Basement", min_value=0, value=0)

yr_built = st.sidebar.number_input("Year Built", min_value=1900, max_value=2015, value=1970)
yr_renovated = st.sidebar.number_input("Year Renovated (0 if none)", min_value=0, max_value=2015, value=0)

zipcode = st.sidebar.number_input("Zipcode", min_value=98001, max_value=98199, value=98052)
lat = st.sidebar.number_input("Latitude", value=47.57, format="%.5f")
long = st.sidebar.number_input("Longitude", value=-122.23, format="%.5f")

sqft_living15 = st.sidebar.number_input("Sqft Living (2015)", min_value=100, value=1800)
sqft_lot15 = st.sidebar.number_input("Sqft Lot (2015)", min_value=500, value=5000)

year = st.sidebar.number_input("Prediction Year", min_value=2014, max_value=2015, value=2014)
month = st.sidebar.number_input("Prediction Month", min_value=1, max_value=12, value=5)
day = st.sidebar.number_input("Prediction Day", min_value=1, max_value=31, value=27)


# -------------------------------
# MODEL ARCHITECTURE DISPLAY
# -------------------------------
st.subheader("🏗️ Model Architecture (Stacking Ensemble)")
st.markdown("""
<div class="card">
<b>✅ Base Models Used:</b>
<ul>
<li>Linear Regression</li>
<li>Decision Tree Regressor</li>
<li>Random Forest Regressor</li>
</ul>

<b>✅ Meta Model Used:</b>
<ul>
<li>Linear Regression</li>
</ul>

📌 <b>Stacking</b> combines base model outputs and trains a meta-model for better accuracy.
</div>
""", unsafe_allow_html=True)

st.divider()


# -------------------------------
# PREDICTION BUTTON
# -------------------------------
if st.button("🔘 Predict House Price (Stacking Model)"):

    user_input = pd.DataFrame([{
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": sqft_living,
        "sqft_lot": sqft_lot,
        "floors": floors,
        "waterfront": waterfront,
        "view": view,
        "condition": condition,
        "grade": grade,
        "sqft_above": sqft_above,
        "sqft_basement": sqft_basement,
        "yr_built": yr_built,
        "yr_renovated": yr_renovated,
        "zipcode": zipcode,
        "lat": lat,
        "long": long,
        "sqft_living15": sqft_living15,
        "sqft_lot15": sqft_lot15,
        "year": year,
        "month": month,
        "day": day
    }]).astype(np.float32)

    # ✅ Access fitted stacking model
    fitted_stack = stacking_pipeline.named_steps["stack"]

    # ✅ Base models should use RAW input (not scaled)
    base_preds = {}
    for name, model in fitted_stack.named_estimators_.items():
        base_preds[name] = float(model.predict(user_input)[0])

    # ✅ Final stacking output from full pipeline
    final_price = float(stacking_pipeline.predict(user_input)[0])

    st.subheader("📌 Prediction Result")
    st.markdown(
        f"<div class='price-box'>💰 Estimated House Price: ${final_price:,.2f}</div>",
        unsafe_allow_html=True
    )

    st.markdown("<div class='card'><h3>📊 Base Model Predictions</h3></div>", unsafe_allow_html=True)

    pred_df = pd.DataFrame({
        "Model": list(base_preds.keys()),
        "Predicted Price": [f"${p:,.2f}" for p in base_preds.values()]
    })
    st.dataframe(pred_df, use_container_width=True)

    st.markdown("<div class='card'><h3>📌 Random Forest Feature Importance</h3></div>", unsafe_allow_html=True)

    # ✅ Feature importance from Random Forest
    rf_model = fitted_stack.named_estimators_["Random Forest"]

    importances = rf_model.feature_importances_
    feature_names = X.columns

    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(10)

    st.write("✅ Top 10 Most Important Features (Random Forest)")
    st.dataframe(imp_df, use_container_width=True)

    # Plot feature importance bar chart
    fig = plt.figure(figsize=(10, 4))
    plt.bar(imp_df["Feature"], imp_df["Importance"])
    plt.xticks(rotation=45, ha="right")
    plt.title("Top 10 Feature Importances (Random Forest)")
    st.pyplot(fig)

    st.markdown("<div class='card'><h3>💼 Business Explanation</h3></div>", unsafe_allow_html=True)
    st.success(
        "This system uses a Stacking Ensemble model that combines Linear Regression, Decision Tree, and Random Forest. "
        "It learns from their predictions to provide a more accurate final house price estimate."
    )

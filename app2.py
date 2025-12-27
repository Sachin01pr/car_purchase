import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Car Purchase Decision (ML)",
    page_icon="🚗",
    layout="centered"
)

# ---------------- TITLE ----------------
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>
    🚗 Car Purchase Decision System
    </h1>
    <p style='text-align: center;'>
    Machine Learning based Decision Tree Model
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------------- DATA ----------------
data = {
    "color": ["red","red","red","red","yellow","yellow","yellow","yellow"],
    "model": [2015,2005,2005,2005,2018,2018,2018,2018],
    "mileage": [40,40,60,60,30,30,30,30],
    "make": ["bmw","bmw","bmw","bmw","ferrari","toyota","toyota","toyota"],
    "buy": [1,1,0,0,1,0,0,0]
}

df = pd.DataFrame(data)

# ---------------- ENCODING ----------------
color_encoder = LabelEncoder()
make_encoder = LabelEncoder()

df["color_encoded"] = color_encoder.fit_transform(df["color"])
df["make_encoded"] = make_encoder.fit_transform(df["make"])

# ---------------- FEATURES ----------------
X = df[["color_encoded", "model", "mileage", "make_encoded"]]
y = df["buy"]

# ---------------- TRAIN MODEL ----------------
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X, y)

# ---------------- USER INPUT UI ----------------
st.subheader("🔍 Enter Car Details")

col1, col2 = st.columns(2)

with col1:
    color = st.selectbox("Car Color", ["red", "yellow"])
    car_model = st.number_input("Model Year", min_value=1990, max_value=2025, value=2018)

with col2:
    mileage = st.number_input("Mileage", min_value=0, max_value=200, value=30)
    make = st.selectbox("Car Make", ["bmw", "ferrari", "toyota"])

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Decision"):
    color_encoded = color_encoder.transform([color])[0]
    make_encoded = make_encoder.transform([make])[0]

    prediction = dt_model.predict(
        [[color_encoded, car_model, mileage, make_encoded]]
    )[0]

    st.divider()

    if prediction == 1:
        st.success("✅ **Decision: BUY the car**")
    else:
        st.error("❌ **Decision: DON'T BUY the car**")

# ---------------- EXPLANATION ----------------
with st.expander("🧠 How does this model work?"):
    st.markdown("""
    - This app uses a **Decision Tree Classifier**
    - Inputs:
        - Car Color
        - Model Year
        - Mileage
        - Brand
    - The model learns **patterns from past data**
    - Final decision is made by **tree-based rules**
    """)

# ---------------- FOOTER ----------------
st.markdown(
    """
    <hr>
    <p style='text-align:center; font-size:12px;'>
    Built with ❤️ using Streamlit & Machine Learning
    </p>
    """,
    unsafe_allow_html=True
)


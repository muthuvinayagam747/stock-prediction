import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression


# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Stock Prediction",
    page_icon="📈"
)


# ============================================
# TITLE
# ============================================

st.title("📈 Stock Prediction Using Tiingo Dataset")

st.write(
    "Stock closing price prediction using "
    "machine learning."
)


# ============================================
# SAMPLE TRAINING DATA
# ============================================

X = np.array([
    [150, 153, 148, 1000],
    [152, 155, 150, 1100],
    [154, 157, 152, 1050],
    [153, 156, 151, 1200],
    [156, 159, 154, 1150],
    [158, 161, 156, 1300],
    [160, 163, 158, 1250],
    [162, 165, 160, 1400],
    [161, 164, 159, 1350],
    [164, 167, 162, 1500],
    [166, 169, 164, 1450],
    [168, 171, 166, 1600],
    [170, 173, 168, 1550],
    [169, 172, 167, 1700],
    [172, 175, 170, 1650]
])

y = np.array([
    152, 154, 156, 155, 158,
    160, 162, 164, 163, 166,
    168, 170, 172, 171, 174
])


# ============================================
# TRAIN MODEL
# ============================================

model = LinearRegression()

model.fit(X, y)


# ============================================
# USER INPUT
# ============================================

st.header("Enter Stock Information")

open_price = st.number_input(
    "Open Price",
    value=175.0
)

high_price = st.number_input(
    "High Price",
    value=178.0
)

low_price = st.number_input(
    "Low Price",
    value=173.0
)

volume = st.number_input(
    "Volume",
    value=1800.0
)


# ============================================
# PREDICTION
# ============================================

if st.button("Predict Stock Price"):

    input_data = np.array([
        [
            open_price,
            high_price,
            low_price,
            volume
        ]
    ])

    prediction = model.predict(
        input_data
    )

    st.success(
        "Predicted Closing Price: "
        + str(round(prediction[0], 2))
    )


st.write("---")

st.info(
    "This is an educational stock prediction "
    "demonstration and should not be used as "
    "financial advice."
)

import streamlit as st
import pickle


model = pickle.load(open("outputs/lin_reg_model.pkl", "rb"))


def predict_sales_price():
    st.subheader("Linear Regression")
    LotArea = st.slider("Lot size in square feet:", 1300, 215245)
    YearBuilt = st.slider("Original construction date:", 1872, 2010)
    OverallQual = st.slider(
        "Rates the overall material and finish of the house:", 1, 10
    )
    GrLivArea = st.slider("Above grade (ground) living area square feet:", 334, 5642)
    inputs = [[LotArea, YearBuilt, OverallQual, GrLivArea]]

    if st.button("Predict"):
        prediction = model.predict(inputs)
        st.success(f"The prediced Sales Price is: {round(prediction[0], 4)}")
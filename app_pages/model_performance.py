import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def model_performance():
    # Load the data
    data = pd.read_csv("inputs/datasets/raw/house_prices_records.csv")

    # Prepare the data
    data = data.dropna(subset=["SalePrice"])
    data = data.loc[data["SalePrice"].notna()]

    # Select the features
    features = ["LotArea", "YearBuilt", "OverallQual", "GrLivArea"]

    # Split the data into train and test sets
    X = data[features]
    y = data["SalePrice"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Display evaluation results in Streamlit
    st.header("Model Performance Evaluation")
    st.subheader("Mean Squared Error (MSE):")
    st.write(mse)
    st.subheader("Mean Absolute Error (MAE):")
    st.write(mae)
    st.subheader("R-squared (R2):")
    st.write(r2)

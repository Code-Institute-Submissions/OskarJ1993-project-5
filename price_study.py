import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def price_study():
    # Load the data
    data = pd.read_csv("inputs/datasets/raw/house_prices_records.csv")

    # Prepare the data
    data = data.dropna(subset=["SalePrice"])
    data = data.loc[data["SalePrice"].notna()]

    # Select the features
    features = ["LotArea", "YearBuilt", "OverallQual", "GrLivArea"]

    # Correlation Study Summary
    correlation_matrix = data[features].corr()

    headings = [
        "LotArea vs Sale Price",
        "YearBuilt vs Sale Price",
        "OverallQual vs Sale Price",
        "GrLivArea vs Sale Price",
    ]

    # Correlation Matrix Visualization
    st.subheader("Correlation Study")
    figure = plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    st.pyplot(figure)

    # Individual plots per variable
    for heading, variable in zip(headings, features):
        st.subheader(heading)
        figure = plt.figure(figsize=(6, 4))
        sns.scatterplot(data=data, x=variable, y="SalePrice")
        st.pyplot(figure)
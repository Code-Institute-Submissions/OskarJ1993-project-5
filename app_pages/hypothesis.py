"""
This file and its contents has been informed and adapted 
from the Churnometer Walkthrough Project.
"""

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def hypothesis_body():
    # Load the dataset
    data = pd.read_csv("inputs/datasets/raw/house_prices_records.csv")

    st.info(
        "### Project Hypothesis\n\n"
        "The following are the hypotheses that I put forward for this project:\n\n"
        "1. I assume that a house with greater OverallQual would sell for a higher price.\n"
        "We could possible confirm this with a correlation analysis between OverallQual and SalePrice.\n\n"
        "2. I further assume that a house with a greater GrLivArea sells for a higher price.\n"
        "We could possible confirm this with a correlation analysis between GrLivArea and SalePrice can show this relationship.\n\n"
    )

    # Validation of hypotheses
    st.write("---")
    st.write("### Hypothesis Validation")
    # Variable Study
    # Hypothesis 1: House with higher OverallQual sells for a higher price
    correlation_overallqual = data["OverallQual"].corr(data["SalePrice"])

    # Hypothesis 2: House with greater GrLivArea sells for a higher price
    correlation_grlivarea = data["GrLivArea"].corr(data["SalePrice"])

    # Correlation Analysis
    correlation_matrix = data[["OverallQual", "GrLivArea", "SalePrice"]].corr()

    # Streamlit app
    st.title("Heritage Housing Issues")

    # Display Hypotheses
    st.subheader("Hypotheses")
    st.write("Hypothesis 1: A house with greater OverallQual sells for a higher price.")
    st.write("Hypothesis 2: A house with greater GrLivArea sells for a higher price.")

    # Correlation Analysis
    st.subheader("Correlation Analysis")
    st.write("Correlation between OverallQual and SalePrice:", correlation_overallqual)
    st.write("Correlation between GrLivArea and SalePrice:", correlation_grlivarea)

    # Correlation Matrix Visualization
    st.subheader("Correlation Matrix")
    figure = plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    st.pyplot(figure)

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the data
data = pd.read_csv("inputs/datasets/raw/house_prices_records.csv")
data.head()

# Prepare the data
data = data.dropna(subset=["SalePrice"])
data = data.loc[data["SalePrice"].notna()]
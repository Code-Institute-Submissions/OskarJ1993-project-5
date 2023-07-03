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

# Select the features
features = ["LotArea", "YearBuilt", "OverallQual", "GrLivArea"]

# List of variables that correlate to SalePrice
# Inspect data
data[features].head()

# Correlation Study Summary
correlation_matrix = data[features].corr()
print(correlation_matrix)

# Correlation Study
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Study")
plt.show()

# Individual plots per variable
for variable in features:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=data, x=variable, y="SalePrice")
    plt.title(f"{variable} vs. SalePrice")
    plt.show()

# Split the data into train and test sets
X = data[features]
y = data["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

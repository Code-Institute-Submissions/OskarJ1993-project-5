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

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)


# Predict on the test set
y_pred = model.predict(X_test)


# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display evaluation results
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2):", r2)


pickle.dump(model, open("outputs/lin_reg_model.pkl", "wb"))

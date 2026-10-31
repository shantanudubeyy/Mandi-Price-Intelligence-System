
# Mandi Price Intelligence System

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(
    r"C:\Users\Lenovo\Desktop\Mandi_Price_Intelligence_System\dataset\9ef84268-d588-465a-a308-a864a43d0070.csv"
)

# Display first five rows
print(df.head())

# Display dataset shape
print("Dataset Shape:", df.shape)

# Display column names
print("Columns:", df.columns)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate records
print("\nDuplicate Records:", df.duplicated().sum())

# ---------------- DATA CLEANING ----------------

# 1. Rename column names
df.rename(columns={
    "Min_x0020_Price": "Min_Price",
    "Max_x0020_Price": "Max_Price",
    "Modal_x0020_Price": "Modal_Price"
}, inplace=True)

# 2. Remove extra spaces from text columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

# 3. Convert price columns into numeric format
df["Min_Price"] = pd.to_numeric(df["Min_Price"], errors="coerce")
df["Max_Price"] = pd.to_numeric(df["Max_Price"], errors="coerce")
df["Modal_Price"] = pd.to_numeric(df["Modal_Price"], errors="coerce")

# 4. Convert Arrival Date into date format
df["Arrival_Date"] = pd.to_datetime(
    df["Arrival_Date"], dayfirst=True, errors="coerce"
)

# 5. Remove duplicate records
df.drop_duplicates(inplace=True)

# 6. Remove missing values
df.dropna(inplace=True)

# 7. Check invalid price records
invalid = (
    (df["Min_Price"] <= 0) |
    (df["Max_Price"] < df["Min_Price"]) |
    (df["Modal_Price"] < df["Min_Price"]) |
    (df["Modal_Price"] > df["Max_Price"])
)

print("Invalid Price Records:", invalid.sum())

# Remove invalid records
df = df[~invalid]

# 8. Save cleaned dataset
df.to_csv(
    r"C:\Users\Lenovo\Desktop\Mandi_Price_Intelligence_System\dataset\cleaned_mandi_data.csv",
    index=False
)

# Display final results
print("\nDATA CLEANING COMPLETED")
print("Final Dataset Shape:", df.shape)
print("Missing Values:", df.isnull().sum().sum())
print("Duplicate Records:", df.duplicated().sum())
print(df.head())


# ==========================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================

# EDA 1: Statistical Analysis

print("\n===== STATISTICAL ANALYSIS =====")

print(df[["Min_Price", "Max_Price", "Modal_Price"]].describe())


# EDA 2: Top 10 Commodities

print("\n===== TOP 10 COMMODITIES =====")

top_commodities = df["Commodity"].value_counts().head(10)

print(top_commodities)

top_commodities.plot(kind="bar", figsize=(9, 5))

plt.title("Top 10 Commodities")
plt.xlabel("Commodity")
plt.ylabel("Number of Records")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# EDA 3: State Wise Analysis

print("\n===== STATE WISE ANALYSIS =====")

top_states = df["State"].value_counts().head(10)

print(top_states)

top_states.plot(kind="bar", figsize=(9, 5))

plt.title("Top 10 States by Mandi Records")
plt.xlabel("State")
plt.ylabel("Number of Records")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# EDA 4: Modal Price Distribution

print("\n===== PRICE DISTRIBUTION =====")

plt.figure(figsize=(8, 5))

plt.hist(df["Modal_Price"], bins=30)

plt.title("Modal Price Distribution")
plt.xlabel("Modal Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# EDA 5: Price Correlation Analysis

print("\n===== PRICE CORRELATION =====")

corr = df[[
    "Min_Price",
    "Max_Price",
    "Modal_Price"
]].corr()

print(corr)

plt.figure(figsize=(7, 5))

plt.imshow(corr, cmap="coolwarm")
plt.colorbar()

plt.xticks(range(3), corr.columns)
plt.yticks(range(3), corr.columns)

plt.title("Price Correlation Heatmap")
plt.tight_layout()
plt.show()


# EDA 6: Date Analysis

print("\n===== DATE ANALYSIS =====")

print("Unique Dates:", df["Arrival_Date"].nunique())
print("Minimum Date:", df["Arrival_Date"].min())
print("Maximum Date:", df["Arrival_Date"].max())


# EDA COMPLETED

print("\n===== EDA COMPLETED SUCCESSFULLY =====")
print("\n===== DATASET DETAILS =====")

print("Unique Commodities:", df["Commodity"].nunique())
print("Unique States:", df["State"].nunique())
print("Unique Markets:", df["Market"].nunique())
print("Unique Districts:", df["District"].nunique())

print("\n===== PRICE RANGE =====")
print("Minimum Modal Price:", df["Modal_Price"].min())
print("Maximum Modal Price:", df["Modal_Price"].max())

print("\n===== TOP 10 EXPENSIVE COMMODITIES =====")
print(
    df.groupby("Commodity")["Modal_Price"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)
# ==========================================
# ADDITIONAL DATA VALIDATION
# ==========================================

print("\n===== EXTREME PRICE RECORDS =====")

print(
    df.nlargest(10, "Max_Price")[
        ["State", "Commodity", "Market",
         "Min_Price", "Max_Price", "Modal_Price"]
    ]
)

print("\n===== DATE CHECK =====")
print(df["Arrival_Date"].value_counts())

print("\n===== PRICE CHECK =====")
print("Modal Price below Min Price:",
      (df["Modal_Price"] < df["Min_Price"]).sum())

print("Modal Price above Max Price:",
      (df["Modal_Price"] > df["Max_Price"]).sum())


# ==========================================
# DATA PREPROCESSING
# ==========================================

from sklearn.model_selection import train_test_split

# Select input features
X = df[
    ["State", "District", "Market",
     "Commodity", "Variety", "Grade"]
]

# Target variable
y = df["Modal_Price"]

# Convert categorical data into numerical data
X = pd.get_dummies(X, drop_first=True)

print("\n===== PREPROCESSING RESULTS =====")
print("Input Features Shape:", X.shape)
print("Target Shape:", y.shape)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\n===== TRAIN TEST SPLIT =====")
print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

print("\nPREPROCESSING COMPLETED")


# ==========================================
# MODEL 1: MULTIPLE LINEAR REGRESSION
# ==========================================

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Create model
lr_model = LinearRegression()

# Train model
lr_model.fit(X_train, y_train)

# Make predictions
lr_predictions = lr_model.predict(X_test)

# Evaluate model
lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))
lr_r2 = r2_score(y_test, lr_predictions)

print("\n===== MULTIPLE LINEAR REGRESSION =====")
print("MAE:", lr_mae)
print("RMSE:", lr_rmse)
print("R2 Score:", lr_r2)

# Compare actual and predicted prices
comparison = pd.DataFrame({
    "Actual Price": y_test.values[:10],
    "Predicted Price": lr_predictions[:10]
})

print("\n===== ACTUAL VS PREDICTED =====")
print(comparison)


# ==========================================
# MODEL 2: DECISION TREE REGRESSOR
# ==========================================

from sklearn.tree import DecisionTreeRegressor

# Create model
dt_model = DecisionTreeRegressor(
    max_depth=15,
    random_state=42
)

# Train model
dt_model.fit(X_train, y_train)

# Make predictions
dt_predictions = dt_model.predict(X_test)

# Evaluate model
dt_mae = mean_absolute_error(y_test, dt_predictions)
dt_rmse = np.sqrt(mean_squared_error(y_test, dt_predictions))
dt_r2 = r2_score(y_test, dt_predictions)

print("\n===== DECISION TREE REGRESSION =====")
print("MAE:", dt_mae)
print("RMSE:", dt_rmse)
print("R2 Score:", dt_r2)

# Actual vs Predicted
dt_comparison = pd.DataFrame({
    "Actual Price": y_test.values[:10],
    "Predicted Price": dt_predictions[:10]
})

print("\n===== ACTUAL VS PREDICTED =====")
print(dt_comparison)


# ==========================================
# MODEL 3: RANDOM FOREST REGRESSOR
# ==========================================

from sklearn.ensemble import RandomForestRegressor

# Create model
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

# Train model
rf_model.fit(X_train, y_train)

# Make predictions
rf_predictions = rf_model.predict(X_test)

# Evaluate model
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_r2 = r2_score(y_test, rf_predictions)

print("\n===== RANDOM FOREST REGRESSION =====")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R2 Score:", rf_r2)

# Actual vs Predicted
rf_comparison = pd.DataFrame({
    "Actual Price": y_test.values[:10],
    "Predicted Price": rf_predictions[:10]
})

print("\n===== ACTUAL VS PREDICTED =====")
print(rf_comparison)




# MODEL 4: POLYNOMIAL REGRESSION

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Use same categorical features as other models
X_poly = df[["State", "District", "Market",
             "Commodity", "Variety", "Grade"]]

y_poly = df["Modal_Price"]

# Convert categorical data into numbers
X_poly = pd.get_dummies(X_poly, drop_first=True)

# Same train-test split
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_poly, y_poly, test_size=0.20, random_state=42
)

# Select a small set of features to avoid memory error
X_train_p = X_train_p.iloc[:, :20]
X_test_p = X_test_p.iloc[:, :20]

# Scaling
scaler_p = StandardScaler()
X_train_scaled = scaler_p.fit_transform(X_train_p)
X_test_scaled = scaler_p.transform(X_test_p)

# Polynomial transformation
poly = PolynomialFeatures(degree=2)
X_train_transformed = poly.fit_transform(X_train_scaled)
X_test_transformed = poly.transform(X_test_scaled)

# Train model
poly_model = LinearRegression()
poly_model.fit(X_train_transformed, y_train_p)

# Prediction
poly_predictions = poly_model.predict(X_test_transformed)

# Evaluation
print("\nPOLYNOMIAL REGRESSION RESULTS")
print("MAE:", mean_absolute_error(y_test_p, poly_predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test_p, poly_predictions)))
print("R2 Score:", r2_score(y_test_p, poly_predictions))

# ==========================================
# MODEL 5: K-MEANS CLUSTERING
# ==========================================

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Select price features
X_cluster = df[["Min_Price", "Max_Price", "Modal_Price"]]

# Scale data
cluster_scaler = StandardScaler()
X_scaled = cluster_scaler.fit_transform(X_cluster)

# Create K-Means model
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Train model and assign clusters
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Display results
print("\n===== K-MEANS CLUSTERING =====")
print(df["Cluster"].value_counts().sort_index())

print("\n===== CLUSTER WISE PRICE ANALYSIS =====")
print(
    df.groupby("Cluster")[
        ["Min_Price", "Max_Price", "Modal_Price"]
    ].mean()
)

# Plot clusters
plt.figure(figsize=(8, 5))

plt.scatter(
    df["Min_Price"],
    df["Modal_Price"],
    c=df["Cluster"],
    alpha=0.5
)

plt.title("Mandi Price Clusters")
plt.xlabel("Minimum Price")
plt.ylabel("Modal Price")
plt.tight_layout()
plt.show()

print("\nK-MEANS CLUSTERING COMPLETED")


# ==========================================
# FAIR MODEL COMPARISON
# ==========================================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Common input features
X_common = df[
    ["State", "District", "Market",
     "Commodity", "Variety", "Grade"]
]

y_common = df["Modal_Price"]

# Encoding
X_common = pd.get_dummies(X_common, drop_first=True)

# Same train-test split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_common, y_common,
    test_size=0.20,
    random_state=42
)

# Create models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=15, random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
}

# Store results
results = []

for name, model in models.items():

    model.fit(X_train_c, y_train_c)

    predictions = model.predict(X_test_c)

    mae = mean_absolute_error(y_test_c, predictions)
    rmse = np.sqrt(mean_squared_error(y_test_c, predictions))
    r2 = r2_score(y_test_c, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

# Display comparison
comparison = pd.DataFrame(results)

print("\n===== FINAL MODEL COMPARISON =====")
print(comparison.to_string(index=False))

print("\nFAIR MODEL COMPARISON COMPLETED")

# ==========================================
# MODEL COMPARISON GRAPH
# ==========================================

plt.figure(figsize=(9, 5))

plt.bar(
    comparison["Model"],
    comparison["R2 Score"]
)

plt.title("Comparison of Regression Models")
plt.xlabel("Machine Learning Models")
plt.ylabel("R2 Score")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

print("\nBEST MODEL:")
print(comparison.loc[comparison["R2 Score"].idxmax()])

# ==========================================
# MODEL 4: POLYNOMIAL REGRESSION
# MEMORY FRIENDLY VERSION
# ==========================================

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression

# Select only two numerical features
X_poly = df[["Min_Price", "Max_Price"]]
y_poly = df["Modal_Price"]

# Split data
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_poly, y_poly,
    test_size=0.20,
    random_state=42
)

# Scale features
scaler_poly = StandardScaler()

X_train_scaled = scaler_poly.fit_transform(X_train_p)
X_test_scaled = scaler_poly.transform(X_test_p)

# Polynomial transformation
poly = PolynomialFeatures(degree=2)

X_train_transformed = poly.fit_transform(X_train_scaled)
X_test_transformed = poly.transform(X_test_scaled)

# Train model
poly_model = LinearRegression()
poly_model.fit(X_train_transformed, y_train_p)

# Prediction
poly_predictions = poly_model.predict(X_test_transformed)

# Evaluation
poly_mae = mean_absolute_error(y_test_p, poly_predictions)
poly_rmse = np.sqrt(mean_squared_error(y_test_p, poly_predictions))
poly_r2 = r2_score(y_test_p, poly_predictions)

print("\n===== POLYNOMIAL REGRESSION =====")
print("MAE:", poly_mae)
print("RMSE:", poly_rmse)
print("R2 Score:", poly_r2)

print("\nPOLYNOMIAL REGRESSION COMPLETED")
print("Features used:", X_train_p.shape[1])
print("Polynomial features:", X_train_transformed.shape[1])
print("Predictions range:", poly_predictions.min(), poly_predictions.max())

# ==========================================
# ACTUAL VS PREDICTED GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.scatter(y_test, lr_predictions, alpha=0.5)

plt.xlabel("Actual Modal Price")
plt.ylabel("Predicted Modal Price")
plt.title("Linear Regression: Actual vs Predicted")

plt.tight_layout()
plt.show()

# RESIDUAL ERROR GRAPH

residuals = y_test - lr_predictions

plt.figure(figsize=(8, 5))
plt.scatter(lr_predictions, residuals, alpha=0.5)

plt.axhline(y=0, color='red', linestyle='--')

plt.xlabel("Predicted Modal Price")
plt.ylabel("Residual Error")
plt.title("Linear Regression: Residual Error Plot")

plt.tight_layout()
plt.show()


# FIVE ALGORITHMS COMPARISON

model_names = [
    "Linear Regression",
    "Polynomial Regression",
    "Decision Tree",
    "Random Forest",
    "K-Means"
]

r2_scores = [
    0.7814,
    -116084985.6742,
    0.5462,
    0.5582,
    0  # K-Means has no R2 score; placeholder only
]

plt.figure(figsize=(10, 5))

plt.bar(model_names, r2_scores)

plt.xlabel("Algorithms")
plt.ylabel("R2 Score")
plt.title("Comparison of Five ML Algorithms")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()
# FIVE ALGORITHMS OVERVIEW

algorithms = [
    "Linear Regression",
    "Polynomial Regression",
    "Decision Tree",
    "Random Forest",
    "K-Means"
]

scores = [
    0.7814,
    0.0295,
    0.5462,
    0.5582,
    0  # K-Means has no R2 score
]

plt.figure(figsize=(10, 5))
plt.bar(algorithms, scores)

plt.xlabel("ML Algorithms")
plt.ylabel("R2 Score (Regression Models Only)")
plt.title("Comparison of Five ML Algorithms")
plt.xticks(rotation=15)
plt.ylim(0, 1)

plt.tight_layout()
plt.show()

# K-MEANS CLUSTER VISUALIZATION

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Min_Price"],
    df["Modal_Price"],
    c=kmeans.labels_,
    cmap="viridis",
    alpha=0.5
)

plt.xlabel("Minimum Price")
plt.ylabel("Modal Price")
plt.title("K-Means Clustering of Mandi Prices")
plt.colorbar(label="Cluster")

plt.tight_layout()
plt.savefig("kmeans_clustering.png")
plt.show()
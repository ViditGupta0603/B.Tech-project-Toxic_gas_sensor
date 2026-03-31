import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from xgboost import XGBRegressor
# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("gas_sensor_data.csv")




target = "AdsorptionEnergy"

X = df.drop(columns=[target])
y = df[target]

# -------------------------------
# 2. Encode categorical
# -------------------------------
X = pd.get_dummies(X)

# -------------------------------
# 3. Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_test = y_test.reset_index(drop=True)

# -------------------------------
# 4. Scaling
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 5. Models
# -------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, verbosity=0),
    "SVR": SVR(kernel='rbf'),
    "Neural Network": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
}

# -------------------------------
# 6. Train + Evaluate
# -------------------------------
results = []

for name, model in models.items():
    print(f"\n==============================")
    print(f"Training {name}")
    print(f"==============================")

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    exec_time = end - start

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Time": exec_time
    })

    # -------------------------------
    # Sample Output
    # -------------------------------
    print("\nSample Predictions:")
    y_arr = y_test.values
    for i in range(min(5, len(y_arr))):
        print(f"Actual: {y_arr[i]:.3f} | Predicted: {predictions[i]:.3f}")

    # -------------------------------
    # Error Calculation
    # -------------------------------
    errors = abs(y_arr - predictions)

    # Save CSV
    pd.DataFrame({
        "Actual": y_arr,
        "Predicted": predictions,
        "Error": errors
    }).to_csv(f"{name.replace(' ', '_')}_predictions.csv", index=False)

    # -------------------------------
    # Error Plot
    # -------------------------------
    plt.figure()
    plt.plot(errors)
    plt.title(f"{name} - Error vs Samples")
    plt.xlabel("Sample Index")
    plt.ylabel("Absolute Error")
    plt.show()

    # -------------------------------
    # Error Distribution
    # -------------------------------
    plt.figure()
    plt.hist(errors, bins=20)
    plt.title(f"{name} - Error Distribution")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.show()

# -------------------------------
# 7. Final Comparison
# -------------------------------
results_df = pd.DataFrame(results)

print("\nFinal Comparison Table:\n")
print(results_df.sort_values(by="R2", ascending=False))

# -------------------------------
# 8. Comparison Graphs
# -------------------------------

# MAE
plt.figure()
plt.bar(results_df["Model"], results_df["MAE"])
plt.title("MAE Comparison (Lower is Better)")
plt.xticks(rotation=30)
plt.show()

# RMSE
plt.figure()
plt.bar(results_df["Model"], results_df["RMSE"])
plt.title("RMSE Comparison (Lower is Better)")
plt.xticks(rotation=30)
plt.show()

# R2
plt.figure()
plt.bar(results_df["Model"], results_df["R2"])
plt.title("R2 Score Comparison (Higher is Better)")
plt.xticks(rotation=30)
plt.show()

# Time
plt.figure()
plt.bar(results_df["Model"], results_df["Time"])
plt.title("Execution Time Comparison (Lower is Better)")
plt.xticks(rotation=30)
plt.show()
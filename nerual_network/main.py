import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
df = pd.read_csv("Well production analysis data.csv")

# ----------------------------------------------------------------------
# 2. CLEAN DATA
# ----------------------------------------------------------------------
# Keep only production wells (not injection) that were actually flowing
df = df[(df["FLOW_KIND"] == "production") & (df["ON_STREAM_HRS"] > 0)].copy()

# Drop rows with impossible negative volumes
df = df[df["BORE_OIL_VOL"] >= 0]

# Features (sensor readings) and target
feature_cols = [
    "ON_STREAM_HRS",
    "AVG_DOWNHOLE_PRESSURE",
    "AVG_DOWNHOLE_TEMPERATURE",
    "AVG_DP_TUBING",
    "AVG_ANNULUS_PRESS",
    "AVG_CHOKE_SIZE_P",
    "AVG_WHP_P",
    "AVG_WHT_P",
    "DP_CHOKE_SIZE",
]
target_col = "BORE_OIL_VOL"

df = df.dropna(subset=feature_cols + [target_col])

print(f"Rows after cleaning: {len(df)}")
print(f"Wells included     : {df['NPD_WELL_BORE_NAME'].nunique()}")

X = df[feature_cols].values
y = df[target_col].values

# ----------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# Scale features (essential for neural networks)
x_scaler = StandardScaler().fit(X_train)
X_train_s = x_scaler.transform(X_train)
X_test_s = x_scaler.transform(X_test)

y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
y_train_s = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
# y_test kept in original units for reporting

# ----------------------------------------------------------------------
# 4. MODEL A: SHALLOW NEURAL NETWORK (1 hidden layer, ReLU)
# ----------------------------------------------------------------------
model_a = MLPRegressor(
    hidden_layer_sizes=(32,),
    activation="relu",
    solver="adam",
    max_iter=2000,
    random_state=RANDOM_STATE,
    early_stopping=True,
)
model_a.fit(X_train_s, y_train_s)

pred_a_scaled = model_a.predict(X_test_s)
pred_a = y_scaler.inverse_transform(pred_a_scaled.reshape(-1, 1)).ravel()

# ----------------------------------------------------------------------
# 5. MODEL B: DEEP NEURAL NETWORK (3 hidden layers, Tanh)
# ----------------------------------------------------------------------
model_b = MLPRegressor(
    hidden_layer_sizes=(64, 32, 16),
    activation="tanh",
    solver="adam",
    max_iter=2000,
    random_state=RANDOM_STATE,
    early_stopping=True,
)
model_b.fit(X_train_s, y_train_s)

pred_b_scaled = model_b.predict(X_test_s)
pred_b = y_scaler.inverse_transform(pred_b_scaled.reshape(-1, 1)).ravel()

# ----------------------------------------------------------------------
# 6. METRICS
# ----------------------------------------------------------------------
def report(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name}")
    print(f"  RMSE : {rmse:,.2f}")
    print(f"  MAE  : {mae:,.2f}")
    print(f"  R^2  : {r2:.4f}")
    return rmse, mae, r2

rmse_a, mae_a, r2_a = report("Model A - Shallow NN (ReLU, 1 layer)", y_test, pred_a)
rmse_b, mae_b, r2_b = report("Model B - Deep NN (Tanh, 3 layers)", y_test, pred_b)

# ----------------------------------------------------------------------
# 7. PLOTS
# ----------------------------------------------------------------------
errors_a = y_test - pred_a
errors_b = y_test - pred_b

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# (a) Error histogram comparison
axes[0, 0].hist(errors_a, bins=40, alpha=0.6, label="Shallow NN (ReLU)", color="#1f77b4")
axes[0, 0].hist(errors_b, bins=40, alpha=0.6, label="Deep NN (Tanh)", color="#ff7f0e")
axes[0, 0].axvline(0, color="black", linestyle="--", linewidth=1)
axes[0, 0].set_title("Prediction Error Distribution")
axes[0, 0].set_xlabel("Error (Actual - Predicted) [Oil Volume]")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].legend()

# (b) Actual vs Predicted scatter
axes[0, 1].scatter(y_test, pred_a, alpha=0.4, s=15, label="Shallow NN", color="#1f77b4")
axes[0, 1].scatter(y_test, pred_b, alpha=0.4, s=15, label="Deep NN", color="#ff7f0e")
lims = [0, max(y_test.max(), pred_a.max(), pred_b.max())]
axes[0, 1].plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
axes[0, 1].set_title("Actual vs Predicted Oil Volume")
axes[0, 1].set_xlabel("Actual")
axes[0, 1].set_ylabel("Predicted")
axes[0, 1].legend()

# (c) Metric comparison bar chart
metrics_labels = ["RMSE", "MAE"]
a_vals = [rmse_a, mae_a]
b_vals = [rmse_b, mae_b]
x = np.arange(len(metrics_labels))
width = 0.35
axes[1, 0].bar(x - width/2, a_vals, width, label="Shallow NN", color="#1f77b4")
axes[1, 0].bar(x + width/2, b_vals, width, label="Deep NN", color="#ff7f0e")
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(metrics_labels)
axes[1, 0].set_title("Error Metric Comparison")
axes[1, 0].legend()

# (d) Training loss curves
axes[1, 1].plot(model_a.loss_curve_, label="Shallow NN", color="#1f77b4")
axes[1, 1].plot(model_b.loss_curve_, label="Deep NN", color="#ff7f0e")
axes[1, 1].set_title("Training Loss Curve")
axes[1, 1].set_xlabel("Iteration")
axes[1, 1].set_ylabel("Loss")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("nn_comparison.png", dpi=150)
print("\nSaved plot to nn_comparison.png")

# Display the plot window (works when run locally, e.g. `python nn_compare.py`)
plt.show()
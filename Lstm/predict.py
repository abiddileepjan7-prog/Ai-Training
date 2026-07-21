# ============================================================
# Test the Trained Oil Production LSTM Model
# ============================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------

DATA_PATH = Path("Oil well.xlsx")
MODEL_PATH = Path("oil_production_lstm.keras")
SCALER_PATH = Path("oil_production_scaler.pkl")

LOOKBACK = 30
TRAIN_RATIO = 0.80


# ------------------------------------------------------------
# 2. Check Required Files
# ------------------------------------------------------------

required_files = [
    DATA_PATH,
    MODEL_PATH,
    SCALER_PATH,
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )


# ------------------------------------------------------------
# 3. Load and Clean Dataset
# ------------------------------------------------------------

df = pd.read_excel(
    DATA_PATH,
    skiprows=2
)

# Remove newline characters and extra spaces
df.columns = (
    df.columns
    .str.replace("\n", "", regex=False)
    .str.strip()
)

# Keep only date and oil-production columns
df = df[
    ["Date", "Oil volume (m3/day)"]
].rename(
    columns={
        "Date": "date",
        "Oil volume (m3/day)": "oil_volume",
    }
)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce",
)

df["oil_volume"] = pd.to_numeric(
    df["oil_volume"],
    errors="coerce",
)

# Clean and arrange the time series
df = (
    df.dropna(subset=["date"])
    .drop_duplicates(subset="date")
    .sort_values("date")
    .set_index("date")
    .asfreq("D")
    .ffill()
    .dropna()
    .reset_index()
)

print("Dataset loaded successfully.")
print("Total rows:", len(df))
print(
    "Date range:",
    df["date"].min().date(),
    "to",
    df["date"].max().date(),
)


# ------------------------------------------------------------
# 4. Load Model and Scaler
# ------------------------------------------------------------

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

scaler = joblib.load(SCALER_PATH)

print("\nModel and scaler loaded successfully.")
print("Expected model input shape:", model.input_shape)


# ------------------------------------------------------------
# 5. Prepare Test Data
# ------------------------------------------------------------

values = df[["oil_volume"]].to_numpy()

split_index = int(
    len(values) * TRAIN_RATIO
)

scaled_values = scaler.transform(values)

# Include the last 30 training values as test context
test_scaled = scaled_values[
    split_index - LOOKBACK:
]


def create_sequences(data, lookback):
    """Create LSTM input sequences and targets."""

    X = []
    y = []

    for index in range(lookback, len(data)):
        X.append(
            data[index - lookback:index]
        )

        y.append(
            data[index]
        )

    return np.array(X), np.array(y)


X_test, y_test = create_sequences(
    test_scaled,
    LOOKBACK,
)

print("\nTest samples:", len(X_test))
print("X_test shape:", X_test.shape)


# ------------------------------------------------------------
# 6. Test the Model
# ------------------------------------------------------------

scaled_predictions = model.predict(
    X_test,
    verbose=0,
)

predictions = scaler.inverse_transform(
    scaled_predictions
).flatten()

actual_values = scaler.inverse_transform(
    y_test
).flatten()

test_dates = (
    df["date"]
    .iloc[split_index:]
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# 7. Evaluate the Model
# ------------------------------------------------------------

mae = mean_absolute_error(
    actual_values,
    predictions,
)

rmse = np.sqrt(
    mean_squared_error(
        actual_values,
        predictions,
    )
)

print("\nModel Evaluation")
print("----------------")
print(f"MAE  : {mae:.2f} m³/day")
print(f"RMSE : {rmse:.2f} m³/day")


results = pd.DataFrame({
    "date": test_dates,
    "actual": actual_values,
    "predicted": predictions,
})

print("\nFirst 10 predictions:")
print(results.head(10).to_string(index=False))


# ------------------------------------------------------------
# 8. Predict the Next Day
# ------------------------------------------------------------

last_sequence = scaled_values[
    -LOOKBACK:
].reshape(
    1,
    LOOKBACK,
    1,
)

next_day_scaled = model.predict(
    last_sequence,
    verbose=0,
)

next_day_prediction = scaler.inverse_transform(
    next_day_scaled
)[0, 0]

next_date = (
    df["date"].max()
    + pd.Timedelta(days=1)
)

print("\nNext-Day Forecast")
print("-----------------")
print("Forecast date:", next_date.date())

print(
    f"Predicted production: "
    f"{next_day_prediction:.2f} m³/day"
)


# ------------------------------------------------------------
# 9. Plot Actual and Predicted Values
# ------------------------------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(
    test_dates,
    actual_values,
    label="Actual production",
    color="blue",
)

plt.plot(
    test_dates,
    predictions,
    label="Predicted production",
    color="red",
)

plt.title("Actual vs Predicted Oil Production")
plt.xlabel("Date")
plt.ylabel("Oil Volume (m³/day)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
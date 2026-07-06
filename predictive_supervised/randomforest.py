import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv("market_pipe_thickness_loss_dataset_clean.csv")

feature_cols = ["Pipe_Size_mm", "Material", "Grade", "Max_Pressure_psi",
                 "Temperature_C", "Corrosion_Impact_Percent", "Time_Years"]
X = df[feature_cols]
y = df["Thickness_mm"]


X = pd.get_dummies(X, columns=["Material", "Grade"], drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

print("R2 score:", round(r2, 3))
print("MAE:", round(mae, 3))
print("RMSE:", round(rmse, 3))


plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linestyle="--"
)

plt.title("Actual vs Predicted Thickness")
plt.xlabel("Actual Thickness (mm)")
plt.ylabel("Predicted Thickness (mm)")

plt.grid(True)

plt.show()

residuals = y_test - y_pred

plt.figure(figsize=(8, 6))

plt.scatter(y_pred, residuals)

plt.axhline(
    y=0,
    color="red",
    linestyle="--"
)

plt.title("Residual Plot")
plt.xlabel("Predicted Thickness")
plt.ylabel("Residual")

plt.grid(True)

plt.show()

plt.figure(figsize=(8, 6))

plt.hist(
    residuals,
    bins=20,
    edgecolor="black"
)

plt.title("Distribution of Errors")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")

plt.grid(True)

plt.show()


importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(importance_df)

plt.figure(figsize=(12, 6))

plt.bar(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.title("Feature Importance (Random Forest)")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.xticks(rotation=90)

plt.tight_layout()

plt.show()
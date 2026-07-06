import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (r2_score,mean_absolute_error,mean_squared_error)

df=pd.read_csv("market_pipe_thickness_loss_dataset_clean.csv")
del df["Condition"]
print(df.head())
print(df.isnull().sum())

y = df["Thickness_mm"]
X = df.drop(columns=["Thickness_mm"])
X = pd.get_dummies(X, drop_first=True)
print(X.head())
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

comparison = X_test.copy()

comparison["Actual Thickness"] = y_test
comparison["Predicted Thickness"] = y_pred

r2 = r2_score(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

print("========== Model Evaluation ==========")

print("R² Score :", r2)

print("MAE :", mae)

print("MSE :", mse)

print("RMSE :", rmse)



plt.figure(figsize=(8,6))

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

plt.figure(figsize=(8,6))

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


plt.figure(figsize=(8,6))

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


coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

coefficients = coefficients.sort_values(
    by="Coefficient",
    key=abs,
    ascending=False
)

print(coefficients)


plt.figure(figsize=(12,6))

plt.bar(
    coefficients["Feature"],
    coefficients["Coefficient"]
)

plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Coefficient")

plt.xticks(rotation=90)

plt.tight_layout()

plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1 -- Choose a dataset & host in GitHub
url = 'https://raw.githubusercontent.com/gzhou624/ml-regression-dataset/main/winequality-white.csv'

# 2 -- Pre-process the dataset
df = pd.read_csv(url, delimiter=';')

df = df.dropna()
df = df.drop_duplicates()
print(df.head)

target = 'quality'

X = df.drop(columns=[target])
y = df[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_preprocessed = pd.DataFrame(X_scaled, columns=X.columns)

# Scale target variable
y = y.values
y_scaled = scaler.fit_transform(y.reshape(-1, 1)).ravel()

# 3 -- Split into training/test
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y_scaled, test_size=0.2, random_state=42)

# # 4 -- Construct a linear regression model using gradient descent
iterations = [10000]
tolerances = [1e-8]
learning_rates = ['constant']
eta0s = [0.001]

best_mse, best_r2 = float('inf'), 0
best_iter, best_tolerance, best_lr, best_eta0 = 50, 1e-10, 'constant', 0.0001

for i in iterations:
    for t in tolerances:
        for lr in learning_rates:
            for e in eta0s:
                model = SGDRegressor(max_iter=i, tol=t, learning_rate=lr, eta0=e, random_state=42)
                model.fit(X_train, y_train)


                # 5 -- Apply the model on the test part of dataset
                y_pred = model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                if mse < best_mse:
                    best_mse = mse
                    best_r2 = r2
                    best_iter = i
                    best_tolerance = t
                    best_lr = lr
                    best_eta0 = e

                print(f"Iterations: {i:<10}\tTolerance: {t:<10}\tLearning Rate: {lr:<10}\teta0: {e:<10}\tMSE: {mse:.5f}\tR² Score: {r2:.5f}")

rmse = np.sqrt(best_mse)
print("\n---BEST HYPERPARAMETERS---")
print(f"Number of Iterations: {best_iter}")
print(f"Tolerance: {best_tolerance}")
print(f"Learning Rate: {best_lr}")
print(f"eta0: {best_eta0}")

print("\n---EVALUATION STATISTICS---")
print(f"MSE: {best_mse:.5f}")
print(f"RMSE: {rmse:.5f}")
print(f"R² Score: {best_r2:.5f}")

# # Additional Requirements
model = SGDRegressor(max_iter=best_iter, tol=best_tolerance, learning_rate=best_lr, eta0=best_eta0, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()

# Plotting
plt.scatter(y_test_original, y_pred_original, alpha=0.5)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("SGDRegressor: Actual vs Predicted")
plt.show()

# # actual vs predicted
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("SGDRegressor: Actual vs Predicted")
plt.show()
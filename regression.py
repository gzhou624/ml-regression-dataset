import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

# 1 -- Choose a dataset & host in GitHub
url = 'https://raw.githubusercontent.com/gzhou624/ml-regression-dataset/main/student-mat.csv'

# 2 -- Pre-process the dataset
df = pd.read_csv(url, delimiter=';')

df = df.dropna()
df = df.drop_duplicates()

# convert categorical to continuous
le = LabelEncoder()
categorical_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian',
                    'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

target = 'G3'

# remove low correlation features
correlation_with_target = df.corr()[target].sort_values(ascending=False)
low_corr_features = correlation_with_target[abs(correlation_with_target) < 0.1].index
X = df.drop(columns=low_corr_features)

X = X.drop(columns=[target])
y = df[target]

# 3 -- split data into training and testing parts with a 6/4 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4 -- Construct a linear regression model using gradient descent
# trial history in log.txt -- the following hyperparameters were derived from tuning
best_iter, best_tolerance, best_lr, best_eta0 = 5000, 1e-15, 'adaptive', 0.01

model = model = SGDRegressor(max_iter=best_iter, tol=best_tolerance, learning_rate=best_lr, eta0=best_eta0, random_state=42)
model.fit(X_train, y_train)

# apply the model to the train part to find training error
y_train_pred = model.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_train_pred)
train_explained_variance = explained_variance_score(y_train, y_train_pred)

# 5 -- apply the model on the test part to find testing error
y_test_pred = model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_pred)
test_explained_variance = explained_variance_score(y_test, y_test_pred)

# print results
print("---BEST HYPERPARAMETERS---")
print(f"Number of Iterations: {best_iter}")
print(f"Tolerance: {best_tolerance}")
print(f"Learning Rate: {best_lr}")
print(f"eta0: {best_eta0}")

print("\n---EVALUATION STATISTICS---")
print("TRAINING SET")
print(f"MSE: {train_mse:.5f}")
print(f"RMSE: {train_rmse:.5f}")
print(f"R² Score: {train_r2:.5f}")
print(f"Explained Variance: {train_explained_variance:.5f}")
print("\nTESTING SET")
print(f"MSE: {test_mse:.5f}")
print(f"RMSE: {test_rmse:.5f}")
print(f"R² Score: {test_r2:.5f}")
print(f"Explained Variance: {test_explained_variance:.5f}")

coefficients = model.coef_
intercept = model.intercept_
print("\n---WEIGHT COEFFICIENTS---")
print(f"intercept: {intercept[0]:.5f}")
for feature, coef in zip(X.columns, coefficients):
    print(f"{feature}: {coef:.5f}")


# Additional Requirements -- plots
# actual value vs predicted value
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("G3: Actual vs Predicted")
plt.show()

# each feature compared to output
X_df = pd.DataFrame(X, columns=X.columns)
for column in X_df.columns:
    plt.figure(figsize=(8, 6))
    plt.scatter(X_df[column], y, alpha=0.5)
    plt.xlabel(column)
    plt.ylabel("G3")
    plt.title(f"G3 vs. {column}")
    plt.show()

# correlation heatmap
correlation_matrix = X.copy()
correlation_matrix[target] = y
correlation_matrix = correlation_matrix.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.show()

# Importing the necessary libraries for data manipulation, modeling, visualization, and model performance metrics.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Loading the dataset and exploring its structure, and summary statistics.
df = pd.read_csv("/content/STUDENT_SCORE.csv")  
df.head()
df.describe()
print("Missing values:", df.isnull().sum())
print("Correlation matrix:\n")
df.corr()

# Preparing the data involves splitting it into features (hours of study) and the target variable (scores).
X = df[['Hours']].values
y = df['Scores'].values

# Further division into training and testing sets facilitates model training and evaluation.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the linear regression model involves fitting it to the training data.
LR = LinearRegression()
LR.fit(X_train, y_train)

# Utilizing the trained model (LR) to make predictions on the test set and extrapolate predictions for specific scenarios.
hours_predict = np.array([[9.25]])
predicted_score = LR.predict(hours_predict.reshape(-1, 1))
y_pred = LR.predict(X_test)

# Print the predicted score for 9.25 hours of study.
print(f"Predicted Score for 9.25 hours of study: {predicted_score[0]}")

# Visualizing the results through plots.
plt.scatter(X_test, y_test, color='blue', label='Actual Scores')

# Plot the linear regression line
plt.plot(X_test, y_pred, color='red', label='Linear Regression')

# Plot the predicted score for 9.25 hours
plt.scatter(9.25, predicted_score, color='green', label='Predicted Score (9.25 hours)')

plt.title('Hours Studied vs. Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.legend()
plt.show()

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# Calculate R-squared (R2) score
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared (R2) Score:", r2)
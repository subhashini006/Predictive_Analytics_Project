import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Load dataset
data = pd.read_csv("student_data.csv")

# Convert gender into numbers
data['gender'] = data['gender'].map({
    'female': 0,
    'male': 1
})

# Input features
X = data[['gender', 'math score', 'reading score']]

# Output target
y = data['writing score']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict values
predictions = model.predict(X_test)

# Accuracy
print("R2 Score:", r2_score(y_test, predictions))
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))

# Visualization
plt.scatter(y_test, predictions)

plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")

plt.title("Actual vs Predicted Scores")

plt.show()
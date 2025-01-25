import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
import pickle

warnings.filterwarnings("ignore")

# Load the dataset
data = pd.read_csv("datasets/Forest_fire.csv")

# Prepare features (X) and target (y)
X = data.iloc[:, 1:-1].values  # Exclude 'Area' (object column) and the last column
y = data.iloc[:, -1].values    # Target column (Fire Occurrence)

# Convert target to integers if not already
y = y.astype('int')

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Save the model for future use
with open("models/forest_fire_model.pkl", "wb") as model_file:
    pickle.dump(log_reg, model_file)

# Calculate accuracy on the test set
y_pred = log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Dynamic input for prediction
try:
    user_input = input("Enter input values (e.g., Oxygen, Temperature, Humidity) separated by spaces: ")
    input_values = [float(x) for x in user_input.split()]
    scaled_input = scaler.transform([input_values])  # Scale input values

    # Predict class (fire occurrence or not)
    predicted_class = log_reg.predict(scaled_input)[0]  # Predicted class (0 or 1)

    # Display result
    if predicted_class == 1:
        print("Fire will occur.")
    else:
        print("No fire.")

except Exception as e:
    print(f"Error in processing input: {e}")

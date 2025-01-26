import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import numpy as np
import joblib

# Load dataset
data = pd.read_csv('datasets/final_data.csv')

# Remove the 'max_temp' column if it exists
data = data.drop(columns=['max_temp'], errors='ignore')

# Reduce the size of the dataset to 70%
small_data = data.sample(frac=0.7, random_state=42)  # Use 70% of the data

# Separate features and target variable
X = small_data.drop(columns=['class'])
y = small_data['class']

# Add noise to feature columns to increase randomness
np.random.seed(42)
noise = np.random.normal(0, 0.1, X.shape)  # Add random noise with mean=0 and std=0.1
X_noisy = X + noise

# Introduce label noise (flip 10% of the labels randomly)
flip_indices = np.random.choice(y.index, size=int(0.1 * len(y)), replace=False)
y_noisy = y.copy()
y_noisy.loc[flip_indices] = 1 - y_noisy.loc[flip_indices]  # Assuming binary classification (0/1)

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_noisy, y_noisy)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Initialize Random Forest model with constrained hyperparameters
model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report_str)
print("\nConfusion Matrix:\n", conf_matrix)

# Save the model to a file
joblib.dump(model, 'models/flood_prediction_model.pkl')
print("Model saved as 'flood_prediction_model.pkl'")

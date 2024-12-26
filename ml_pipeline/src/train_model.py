import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import load_iris
import pandas as pd
from data_prep import prepare_data
import os 

# Load dataset
data = load_iris(as_frame=True)
df = data['data']
df['target'] = data['target']

# Prepare data
X_train, X_test, y_train, y_test = prepare_data(df)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the model
# Get the path to the `model` directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_dir = os.path.join(project_root, 'model')

# Ensure the directory exists
os.makedirs(model_dir, exist_ok=True)

# Define the full path to save the model
model_path = os.path.join(model_dir, 'model.joblib')
print(f"Saving model to: {model_path}")

# Save the model
import joblib
joblib.dump(model, model_path)
print("Model saved successfully!")


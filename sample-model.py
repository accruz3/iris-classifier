# train_model.py
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib

# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save model
joblib.dump(model, 'model.pkl')
print("✅ Model saved to model.pkl")
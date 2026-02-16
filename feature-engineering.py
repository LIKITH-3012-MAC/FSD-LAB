from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# Load the dataset
X=np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y=np.array([2, 3, 4, 5, 6])
# Define the features and target variable
make_pipeline=Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
# Fit the pipeline to the data
make_pipeline.fit(X, y)
# Predict using the pipeline
predictions=make_pipeline.predict(X)
# Visualize the predictions
sns.scatterplot(x=X[:, 0], y=y, label='Actual')
sns.scatterplot(x=X[:, 0], y=predictions, label='Predicted')
plt.xlabel('Feature 1')
plt.ylabel('Target Variable')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()


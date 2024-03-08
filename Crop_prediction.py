import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


data = pd.read_csv('Data_for_analysis.csv')


data.fillna(value=np.random.random(), inplace=True)


bins = [0, 1, 2, 3, 4]  
labels = [1, 2, 3, 4]  
data['climate.category'] = pd.cut(data['climate.category'], bins=bins, labels=labels)


features = ['temp.average', 'precip.average', 'temp.min', 'precip.min', 'temp.max', 'precip.max', 'temp.growing.season', 'precip.growing.season']
target = 'climate.category'


X = data[features]
y = data[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier()


model.fit(X_train, y_train)


predictions = model.predict(X_test)


accuracy = accuracy_score(y_test, predictions)
classification_rep = classification_report(y_test, predictions)


print(f"Model Accuracy: {accuracy}")
print("Classification Report:\n", classification_rep)


feature_importance = model.feature_importances_
sns.barplot(x=feature_importance, y=features)
plt.title('Feature Importance')
plt.show()

# Best Time for Crop Plantation - Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='climate.category', y='temp.growing.season', data=data, palette='viridis')
plt.xlabel('Climate Category')
plt.ylabel('Temperature During Growing Season (°C)')
plt.title('Best Time for Crop Plantation Based on Climate Category')
plt.show()

# Kernel Density Estimation of Average Temperature for Each Climate Category
plt.figure(figsize=(12, 8))
for category in data['climate.category'].unique():
    subset = data[data['climate.category'] == category]
    sns.histplot(subset['temp.average'], bins=20, kde=True, label=f'Category {category}')

plt.xlabel('Average Temperature (°C)')
plt.ylabel('Density')
plt.title('Kernel Density Estimation of Average Temperature for Each Climate Category')
plt.legend()
plt.show()

# Visualizing the data for the specific crop
crop_data = data.sample(1)  # Here in the sample we can choose which crop we want to visualize
crop_prediction = model.predict(crop_data[features])
print(f"Suitability Prediction for the Crop: {crop_prediction[0]}")

# Scatter Plot for Different Crops
plt.figure(figsize=(10, 6))
for crop in data['Crop'].unique():
    crop_subset = data[data['Crop'] == crop]
    plt.scatter(crop_subset['temp.average'], crop_subset['precip.average'], c=crop_subset['climate.category'], cmap='viridis', s=50, alpha=0.8, edgecolors='w', linewidths=0.5, label=crop)

plt.xlabel('Average Temperature (°C)')
plt.ylabel('Average Precipitation (mm/month)')
plt.title('Climate Categories based on Temperature and Precipitation for Different Crops')
plt.legend()
plt.show()


joblib.dump(model, 'crop_suitability_model.joblib')

# Histogram of Average Temperature for each Climate Category
plt.figure(figsize=(10, 6))
for category in data['climate.category'].unique():
    subset = data[data['climate.category'] == category]
    sns.histplot(subset['temp.average'], bins=20, kde=True, label=f'Category {category}')

plt.xlabel('Average Temperature (°C)')
plt.ylabel('Density')
plt.title('Kernel Density Estimation of Average Temperature for Each Climate Category')
plt.legend()
plt.show()

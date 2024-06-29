import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from numpy import NaN, NAN, nan
from scipy.stats import chi2_contingency
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,mean_absolute_percentage_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from joblib import dump

# Load your dataset
dfreg = pd.read_csv('housing.csv')
dfreg.head()
dfreg.info()

dfreg.shape
dfreg.dtypes
dfreg.describe()
dfreg.isnull().sum()
dfreg.duplicated().sum()
# Drop rows with missing values
dfreg.dropna(inplace=True)

# Convert categorical variable 'ocean_proximity' to dummy variables
dfreg['ocean_proximity'] = dfreg['ocean_proximity'].convert_dtypes()
dfreg.dtypes

dfreg = pd.concat([dfreg,pd.get_dummies(dfreg['ocean_proximity'])],axis =1)
dfreg.drop('ocean_proximity',axis =1,inplace=True)

#Visulaization
dfreg.hist(figsize=(15, 8),bins = 50)
figure = plt.figure(figsize=(15,10))
sns.heatmap(dfreg.corr(),annot =True)


# Define features (X) and target (y)
x = dfreg[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 
           'population', 'households', 'median_income']]
y = dfreg['median_house_value']

# Split data into training and testing sets
X_trainreg, X_testreg, y_trainreg, y_testreg = train_test_split(x, y, test_size=0.3)
print(f'Shape of the X_train: {X_trainreg.shape}')
print(f'Shape of the X_test: {X_testreg.shape}')
print(f'Shape of the y_train: {y_trainreg.shape}')
print(f'Shape of the y_test: {y_testreg.shape}')
# Train the Linear Regression model
reg_lr=LinearRegression()
reg_lr.fit(x,y)

#Evaluate model
y_pred = reg_lr.predict(X_testreg)
mae = mean_absolute_error(y_testreg, y_pred)
mape = mean_absolute_percentage_error(y_testreg,y_pred)
mse = mean_squared_error(y_testreg,y_pred)
r2 = r2_score(y_testreg, y_pred)


print("Mean Absolute Error:", round(mae, 2))
print("Mean Absolute Percentage Error:", round(mape, 2))
print("Mean Squared Error:", round(mse, 2))
print("R-squared:", round(r2, 2))

reg_lr.predict(X_testreg)

print("Training Accuracy :", reg_lr.score(X_trainreg, y_trainreg ))
print("Testing Accuracy :", reg_lr.score(X_testreg, y_testreg))

scaler_reg=StandardScaler()
X_train_s = scaler_reg.fit_transform(X_trainreg)
reg_lr.fit(X_train_s, y_trainreg)
X_test_s = scaler_reg.transform(X_testreg)
y_pr=reg_lr.predict(X_test_s)

dump(scaler_reg, 'standard_scaler.pkl')

plt.figure(figsize=(10, 6))
plt.scatter(y_testreg, y_pr, alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values for Median House Value')
plt.plot([min(y_testreg), max(y_testreg)], [min(y_testreg), max(y_testreg)], color='red') 
plt.show()
# Save the trained model
dump(reg_lr, 'linear_regression_model.pkl')

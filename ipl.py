# Importing essential libraries
import pandas as pd
import pickle
import datetime
from sklearn import metrics
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

# Import the Dataset
df = pd.read_csv("F:/Python Data Science/Krish Naik/Machine Learning Projects/Datasets/ipl.csv")
df.head()

# See some information about the data
df.info()

# Check for the null values
df.isna().sum()

# Remove unwanted columns
columns_to_remove = ["mid", "bowler"]
df.drop(columns_to_remove, axis=1, inplace=True)

# See the names of all the teams
df['bat_team'].unique()

# Keep only consistent teams
consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                    'Delhi Daredevils', 'Sunrisers Hyderabad']

df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]
df['bat_team'].value_counts()

# Converting 'date' columns to datetype
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Extract year feature from datetime
years = df['date'].dt.year

# Let's add 'year' column to dataframe
df['year'] = years

# Drop the 'date' column
df.drop('date', axis=1, inplace=True)

# Check for Number of categories in categorical features
categorical_features = [feature for feature in df.columns if df[feature].dtype=='O']

categorical_features

# Check the number of venues
df['venue'].value_counts().shape

# Check the number of Batsman
df['batsman'].value_counts().shape


# Converting the categorical features using one hot encoding
# We'll one hot encode only 'bat_team' and 'bowl_team' as they contain only few categories

df = pd.get_dummies(data = df, columns=['bat_team', 'bowl_team'], drop_first=True)


# Let's do mean encoding to venue since it has many categories
# Make a dictionary
mean_encode = df.groupby(["venue"])['total'].mean().to_dict()
mean_encode

# Map the dictionary back to the dataframe
df['venue_mean_encode'] = df['venue'].map(mean_encode)

# Drop the 'venue' column
df.drop("venue", axis=True, inplace=True)


# Let's do mean encoding to 'batsman' column, since it has many categories

# Make a dictionary
mean_encode_batsman = df.groupby(["batsman"])['total'].mean().to_dict()
mean_encode_batsman

# Map the dictionary back to the dataframe
df['batsman_mean_encode'] = df['batsman'].map(mean_encode_batsman)

# Drop the 'Batsman' column
df.drop("batsman", axis=True, inplace=True)



# Splitting the data into train and test set

# Independent Variable
X_train = df.drop(labels='total', axis=1)[df['year'] <= 2016]
X_test = df.drop(labels='total', axis=1)[df['year'] >= 2017]

# Dependent Variable
y_train = df[df['year'] <= 2016]['total']
y_test = df[df['year'] >= 2017]['total']


# Model Building

# Linear Regression Model
regressor = LinearRegression()

# Fit the Linear Regression Model
regressor.fit(X_train, y_train)


# See the Linear Regression model score
regressor.score(X_train, y_train)

# Make prediction with Linear Regression Model
prediction = regressor.predict(X_test)

# See the Linear Regression Prediction Score
print(f"R_square :{regressor.score(X_test, y_test)}")


# See some metrics


print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))


# Check the R Square
metrics.r2_score(y_test, prediction)

# Plot the Linear Regression Prediction Residuals
sns.distplot(y_test-prediction)
plt.xlabel(" Residual")
plt.show()

# Plot original values vs Linear Regression Prediction values
plt.scatter(y_test, prediction, alpha=0.5)
plt.xlabel("Y Test")
plt.ylabel("Prediction")
plt.show()


## Lasso Regression
lasso = Lasso()

# Set different Alpha values
parameters = { 'alpha' : [1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}

# Do Lasso modelling with GridSearchCV Hyperparameter Tuning
lasso_regressor = GridSearchCV(lasso, parameters, scoring="r2", cv=5)

# Fit the Lasso Regression Model
lasso_regressor.fit(X_train, y_train)

print(f"Best Parameters : {lasso_regressor.best_params_}")
print(f"Lasso Model Best Score : {lasso_regressor.best_score_}")

# Do Lasso regression Prediction
prediction1 = lasso_regressor.predict(X_test)

print(f"Lasso Prediction Score : {lasso_regressor.score(X_test, y_test)}")
print('MAE:', metrics.mean_absolute_error(y_test, prediction1))
print('MSE:', metrics.mean_squared_error(y_test, prediction1))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction1)))
print(f"R_square : {metrics.r2_score(y_test, prediction1)}")

# Plot Lasso Regression Prediction Residuals
sns.distplot(y_test-prediction1)
plt.xlabel(" Lasso Residual")
plt.show()

# Plot Original Values vs Lasso Prediction Values
plt.scatter(y_test, prediction1, alpha=0.5)
plt.xlabel("Y Test")
plt.ylabel(" Lasso Prediction")
plt.show()

## Ridge Regression

ridge = Ridge()

# Set different Alpha values
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}

# Do ridge Regression modelling with GridSearchCV 
ridge_regressor = GridSearchCV(ridge, parameters, scoring="r2", cv=5)

# Fit the Ridge Regression Model
ridge_regressor.fit(X_train, y_train)

print(f"Best Parameters : {ridge_regressor.best_params_}")
print(f"Ridge Model Best Score : {ridge_regressor.best_score_}")

# Do Prediction with Ridge Regression
prediction2 = ridge_regressor.predict(X_test)

print(f"Ridge Prediction Score : {ridge_regressor.score(X_test, y_test)}")
print('MAE:', metrics.mean_absolute_error(y_test, prediction2))
print('MSE:', metrics.mean_squared_error(y_test, prediction2))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction2)))
print(f"R_square : {metrics.r2_score(y_test, prediction2)}")

# Plot Ridge Regression Prediction Residuals
sns.distplot(y_test-prediction2)
plt.xlabel(" Ridge Residual")
plt.show()

# Plot Original Values vs Ridge Prediction Values
plt.scatter(y_test, prediction2, alpha=0.5)
plt.xlabel("Y Test")
plt.ylabel(" Ridge Prediction")
plt.show()

# Since Linear Regression is performing well, we'll pickle it
# Creating a pickle file for the classifier
filename = 'iplmodel.pkl'
pickle.dump(regressor, open(filename, 'wb'))


l = [22, 2, 5, 22, 2, 11, 11, 2016, 0,0,0,0,0,1,0,  0,0,0,1,0,0,0, 166.969386 , 155.522500]

data = np.array([l])

data[0]

print(regressor.predict(data)[0])
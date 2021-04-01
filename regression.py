import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

housing = pd.read_csv("housing.csv")
housing.dropna(subset = ["longitude", 'latitude', 'housing_median_age', "total_rooms", "total_bedrooms", "population", "households", "median_income"], inplace=True)


X = housing.drop(columns=['median_house_value', 'ocean_proximity'])
y = housing['median_house_value']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = LinearRegression()
model.fit(X_train, y_train)

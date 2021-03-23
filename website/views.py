from flask import Blueprint, render_template
from flask import Flask, redirect, url_for, render_template, request
import sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


housing = pd.read_csv("housing.csv")
housing.dropna(subset = ["longitude", 'latitude', 'housing_median_age', "total_rooms", "total_bedrooms", "population", "households", "median_income"], inplace=True)


X = housing.drop(columns=['median_house_value', 'ocean_proximity'])
y = housing['median_house_value']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = LinearRegression()
model.fit(X_train, y_train)





views = Blueprint('views', __name__)

@views.route('/', methods=["POST", "GET"])
def calculator():
    if request.method == "POST":
        longitude = request.form['Longitude']
        latitude = request.form['Latitude']
        housing_median_age = request.form['Median Age']
        total_rooms = request.form['Rooms']
        total_bedrooms = request.form['Bedrooms']
        population = request.form['People']
        households = request.form['Households']
        median_income = request.form['Income']

        predictions = model.predict([[longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income]])

        new_pred = predictions.tolist()
        return f'<br><br><br><br><br><br><br><br><br><br><br><br><h1><center> Predicted House Price is: ${str(round(new_pred[0]))}</center></h1>'
    else:
        return render_template("calculator.html")


@views.route('/about')
def about():
    return render_template("about.html")

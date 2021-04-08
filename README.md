# Coronavirus Outcome Classifier

A predictive classifier built with the help of sckit-learn's random forest algorithm. 

[Learn more about Random Forests.](https://en.wikipedia.org/wiki/Random_forest#:~:text=Random%20forests%20or%20random%20decision,average%20prediction%20(regression)%20of%20the)

## Installation

To install the required packages: 

On macOS/Linux:
```
python -m pip install -r requirements.txt
```
On Windows:
```
py -m pip install -r requirements.txt
```

Run the python file in your favorite editor or by this command: 
```
python3 random_forest_model.py
```


## Predictive Classifiers & Random Forest

Predictive classifiers are built from training data (and generally large datasets) and are used to predict data without a determined outcome. In the case of this repository, this predictive classifier is a random forest classifier that is used to predict the outcome of patients infected with COVID-19. Choosing the appropriate classifier may be confusing with the multitude of possible classifiers to choose between. In this case, the main reasons I have chosen the random forest classifier is due to its efficiency in classifying large databases, the ability to handle a variety of input variables, has an effective method for estimating missing data and is generally robustness to outliers and noise. However, the main disadvantage of the random forest classifier is due to it’s time in computing the classifier, but due to the efficiency of sklearn’s random forest implementation and the careful preprocessing of data, the classifier is able to be built in a very reasonable time.

To find up to date datasets on coronavirus: 

https://github.com/CSSEGISandData/COVID-19

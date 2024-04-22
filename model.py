import pandas as pd
import numpy as np
from sklearn import linear_model
import pickle

df = pd.read_csv(
    'newhousing.csv')

#df.bedrooms = df.bedrooms.fillna(df.bedrooms.median())

reg = linear_model.LinearRegression()

# dropping the irrelevant columns in the multivariate dataset
reg.fit(df.drop(['price', 'hotwaterheating',
                 'airconditioning', 'prefarea', 'mainroad', 'semi-furnished', 'unfurnished'], axis='columns'), df.price)

# predicting with area, bedroom, bathrooms, 'basement', stories, guestroom, parking, areaperbedroom, bbratio
print(reg.predict(
    [[5500, 3, 2, 2, 1, 0, 1, 1833.22, 0.667]]))

# Saving model to disk
pickle.dump(reg, open('model.pkl', 'wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[5500, 3, 2, 2, 1, 0, 1, 1833.22, 0.667]]))

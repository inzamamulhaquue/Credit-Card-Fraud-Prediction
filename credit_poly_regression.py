import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv("credit_card_data.csv")

legit = dataset[dataset.Class == 0]
fraud = dataset[dataset.Class == 1]

legit_sample = legit.sample(n=492)

new_dataset = pd.concat([legit_sample,fraud], axis=0)

x= new_dataset.iloc[:,:30].values
y= new_dataset.iloc[:,30:].values

from sklearn.preprocessing import PolynomialFeatures

prg=PolynomialFeatures(degree=3)
x_poly=prg.fit_transform(x)

from sklearn.linear_model import LinearRegression
lg=LinearRegression()
lg.fit(x_poly, y)

y_poly=lg.predict(x_poly)


from sklearn import metrics
mse=metrics.mean_squared_error(y, y_poly)

rmse=np.sqrt(mse)
print("rmse:",rmse)

y_poly=y_poly.round()

accu=accuracy_score(y, y_poly)
print(accu)


plt.plot(x,y,'ro')
plt.plot(x,y_poly,color='blue')
plt.xlabel("Level")
plt.ylabel("Salary")
plt.title("Polynomial Salary vs Level graph")
plt.show()
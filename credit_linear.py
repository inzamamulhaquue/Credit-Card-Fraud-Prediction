import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset=pd.read_csv("credit_card_data.csv")

legit = dataset[dataset.Class == 0]
fraud = dataset[dataset.Class == 1]

legit_sample = legit.sample(n=492)

new_dataset = pd.concat([legit_sample,fraud], axis=0)

x= new_dataset.iloc[:,:30].values
y= new_dataset.iloc[:,30:].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

from sklearn.linear_model import LinearRegression

rg=LinearRegression()
rg.fit(x_train,y_train)

y_pred1=rg.predict(x_test)

y_pred=y_pred1.round()

accu=accuracy_score(y_test,y_pred)
print("Accuracy:-",accu)


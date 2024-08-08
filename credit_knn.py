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

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25, random_state=42)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
x_train=sc.fit_transform(x_train)

x_test=sc.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(weights=('distance'))
knn.fit(x_train, y_train)

y_pred=knn.predict(x_test)

from sklearn.metrics import confusion_matrix

print("My Confusion Matrix:\n",
      confusion_matrix(y_test, y_pred))

print("Accuracy:",accuracy_score(y_test, y_pred))


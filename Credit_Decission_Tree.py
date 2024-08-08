import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset=pd.read_csv("credit_card_data.csv")

legit = dataset[dataset.Class == 0]
fraud = dataset[dataset.Class == 1]

legit_sample = legit.sample(n=492)

new_dataset = pd.concat([legit_sample,fraud], axis=0)

X= new_dataset.iloc[:,:30].values
y= new_dataset.iloc[:,30:].values
 
X_train, X_test, y_train, y_test = train_test_split(X, y,
                        test_size = 0.25, random_state = 0)
 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'gini',
                                    random_state = 0)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)

print("Accuracy:",accuracy_score(y_test, y_pred))


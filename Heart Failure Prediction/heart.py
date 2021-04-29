
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
heart = pd.read_csv("heart_failure_clinical_records_dataset.csv")
heart
target = heart.DEATH_EVENT

heart = heart.drop('DEATH_EVENT', axis='columns')
heart

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(heart, target, test_size=0.2, random_state=10)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
predict = model.predict(X_test)


from sklearn.metrics import classification_report
report = classification_report(y_test, predict)

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test, predict)

import pickle
pickle.dump(model, open('heart.pkl', 'wb'))
model=pickle.load(open('heart.pkl','rb'))
print(model.predict([[75.0, 0, 582, 0, 20, 1, 265000.00, 1.9, 130, 1, 0, 4]]))
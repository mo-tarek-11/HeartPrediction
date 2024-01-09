import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import  train_test_split
from sklearn.ensemble import RandomForestClassifier

#To turn off warning messages.
import warnings
warnings.filterwarnings('ignore')

# Importing the dataset 
data = pd.read_csv("heart.csv")  

#Scale all values for good Accuracy
sc = StandardScaler()
col = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
data[col] = sc.fit_transform(data[col])

# Splitting the data  
X = data.drop(["target"], axis = 1)  
y = data["target"]  
  
#Splitting the data into the training and testing set  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)  

#Train model
model =  RandomForestClassifier(criterion = 'entropy' , max_depth = 10 , n_estimators = 100)
model.fit(X_train, y_train)

print("Train Score:", model.score(X_train,y_train))
print("Test Score:", model.score(X_test,y_test))

pickle.dump(model, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
res = model.predict([[22, 0, 0, 10 , 1,0 ,0,6 ,0 ,0 ,0 ,0 ,0]])
if res == 1:
    print('You have heart problems')
else:
    print('Your Heart is okay')
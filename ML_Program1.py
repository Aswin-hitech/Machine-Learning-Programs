#Linear Discriminant Analysis (LDA)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

#Load Realworld Dataset
iris= load_iris()
x = iris.data
y = iris.target

#train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#LDA MODEL
lda= LinearDiscriminantAnalysis()
lda.fit(x_train,y_train)

#testing
y_pred = lda.predict(x_test)
print("LDA ACCURACY: ", accuracy_score(y_test,y_pred))

#Real-time input 
new_flower = [[5.8,2.7,5.1,1.9]]
prediction = lda.predict(new_flower)
print("Predicted Flower(LDA): ",iris.target_names[prediction][0])

#Quadratic Discriminant Analysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#QDA MODEL
qda = QuadraticDiscriminantAnalysis()
qda.fit(x_train,y_train)

#testing
y_pred_qda = qda.predict(x_test)
print("QDA Accuracy: ",accuracy_score(y_test,y_pred_qda))

#Real-Time Flower Measurements
prediction_qda = qda.predict(new_flower)
print("Predicted Flower(QDA): ",iris.target_names[prediction_qda][0])
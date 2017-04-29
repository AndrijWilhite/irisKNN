
import numpy as np
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
#import antigravity



iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

#k = int(input("Enter a 'K' Value: "))
def main():
        k = 1
        while (k < 113):
            knn = neighbors.KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)

            Y_pred = knn.predict(X_test)

            acc = accuracy_score(Y_pred, y_test)
            accPer = '{:.1%}'.format(acc)

            # print(Y_pred)
            # print(y_test)
            print(accPer, k)
            k = k + 1
        
main()

# sepalLength = input("Enter a Sepal Length: ")
# sepalWidth = input("Enter a Sepal Width: ")
# petalLength = input("Enter a Petal Length: ")
# petalWidth = input("Enter a Petal Width: ")

# guess = iris.target_names[knn.predict([[sepalLength, sepalWidth, petalLength, petalWidth]])]

# print("The probably species is %s. With a test accuracy of %s"%(guess,accPer))
from sklearn import neighbors, datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target
k = int(input("Enter a 'K' Value: "))
knn = neighbors.KNeighborsClassifier(n_neighbors=k)
knn.fit(X, y)


sepalLength = input("Enter a Sepal Length: ")
sepalWidth = input("Enter a Sepal Width: ")
petalLength = input("Enter a Petal Length: ")
petalWidth = input("Enter a Petal Width: ")


print(iris.target_names[knn.predict([[sepalLength, sepalWidth, petalLength, petalWidth]])])
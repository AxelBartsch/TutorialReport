import sklearn

from sklearn import svm, datasets, linear_model, neighbors, cluster

def estimator():
	iris = datasets.load_iris()
	clf = svm.LinearSVC()

	clf.fit(iris.data, iris.target)

	clf.predict([[5.0, 3.6, 1.3, 0.25]])

	print(clf.coef_)

def regression():
	reg = linear_model.LinearRegression()

	reg.fit([[0,0], [1,1], [2,2]], [0,1,2])

	print(reg.coef_)

def neighbor():
	iris = datasets.load_iris()

	knn = neighbors.KNeighborsClassifier()
	knn.fit(iris.data, iris.target)

	result = knn.predict([[0.1, 0.2, 0.3, 0.4]])
	print(result)

def clustering():
	iris = datasets.load_iris()
	#create clusters for k=3
	k=3
	k_means = cluster.KMeans(k)
	#fit data
	k_means.fit(iris.data)
	#print results
	print(k_means.labels_[::10])
	print(iris.target[::10])

if __name__ == "__main__":
	estimator()
	regression()
	neighbor()
	clustering()
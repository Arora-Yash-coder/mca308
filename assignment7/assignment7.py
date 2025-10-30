from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Naive Bayes Classifier Results (GaussianNB)")
print("===========================================")
print("Actual Labels:   ", y_test)
print("Predicted Labels:", y_pred)
print("Accuracy Score:  {:.2f}%".format(accuracy * 100))
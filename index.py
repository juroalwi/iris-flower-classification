import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

data = sns.load_dataset("iris")
X = data.drop("species", axis=1)
y = data["species"]


def plot_pairplots():
    sns.pairplot(data, hue="species")
    plt.savefig("pairplot.png", dpi=300, bbox_inches="tight")


def plot_barplots():
    bar_width = 0.2
    mean_values = np.array([X[y == j].mean() for j in y.unique()])
    x_axis = np.arange(len(X.columns))
    plt.figure()
    plt.bar(x_axis, mean_values[0], bar_width, label="Setosa")
    plt.bar(x_axis + bar_width, mean_values[1], bar_width, label="Versicolor")
    plt.bar(x_axis + bar_width * 2, mean_values[2], bar_width, label="Virginica")
    plt.xticks(x_axis + bar_width, list(X.columns))
    plt.xlabel("Features")
    plt.ylabel("Mean value in cm")
    plt.legend(bbox_to_anchor=(1, 1))
    plt.savefig("barplot.png", dpi=300, bbox_inches="tight")


def svm_classification():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    svc = SVC()
    svc.fit(X_train, y_train)
    print(f"SVM classification score: {svc.score(X_test, y_test)}")


def tree_classification_with_plot():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    plt.figure()
    ax = sns.scatterplot(
        x=X["petal_width"],
        y=X["petal_length"],
        hue=data["species"],
        style=dtc.predict(X),
    )
    ax.set_title("Real labels")
    ax.legend(title="Real labels and predicted labels")
    plt.savefig("tree-classification.png", dpi=300, bbox_inches="tight")
    print(f"Decision tree classification score: {dtc.score(X_test, y_test)}")


# plot_pairplots()
# plot_barplots()
svm_classification()
tree_classification_with_plot()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class labels']

df = pd.read_csv('iris.data', names=columns)

print('First rows:')
print(df.head(), '\n')

print('Data description:')
print(df.describe(), '\n')

# pair plots
pairplot = sns.pairplot(df, hue='Class labels')
plt.savefig('pairplot.png', dpi=300, bbox_inches='tight')

data = df.values
X = data[:, :4]
Y = data[:, 4]

# bar plots
bar_width = 0.2
mean_values = np.array([np.average(X[:,i][Y==j]) for i in range(X.shape[1]) for j in np.unique(Y)])
reshaped = mean_values.reshape(4,3)
reshaped = np.swapaxes(reshaped,0,1)
x_axis  =  np.arange(len(columns)-1)
plt.bar(x_axis, reshaped[0], bar_width, label="Setosa")
plt.bar(x_axis + bar_width, reshaped[1], bar_width, label="Versicolor")
plt.bar(x_axis + bar_width * 2, reshaped[2], bar_width, label="Virginica")
plt.xticks(x_axis + bar_width, columns[:4])
plt.xlabel("Features")
plt.ylabel("Mean value in cm")
plt.legend(bbox_to_anchor=(1,1))
plt.savefig('barplot.png', dpi=300, bbox_inches='tight')

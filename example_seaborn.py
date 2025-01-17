import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(24)
data = np.random.randint(low=10, high=100, size=(7, 7))
hm = sns.heatmap(data=data, cmap="viridis", annot=True)
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.datasets  import load_digits

mnist = load_digits()
data = mnist['images']

data = data.reshape(data.shape[0], -1)
mean_vec = np.mean(data, axis=0)
data_norm = data - mean_vec
covariance_matrix = (data_norm).T.dot((data_norm)) / (data_norm.shape[0])

eigen_vals, eigen_vects = eigh(covariance_matrix)
idx=eigen_vals.argsort()[::-1]
eigen_vals=eigen_vals[idx]
eigen_vects = eigen_vects[:,idx]

num_of_component = 2
eigen_values = eigen_vals[:num_of_component]
eigen_vectors = eigen_vects[:,:num_of_component]

# result after dimensionality reduction
result = np.dot(data,eigen_vectors) 
df = pd.DataFrame(data = result, columns = ("1st principal","2nd principal"))

target = np.array(mnist['target'])
df['target'] = target

plt.figure(figsize=(15,8))
for i, j in enumerate(np.unique(target)): 
    plt.scatter(df['1st principal'][target == j], df['2nd principal'][target == j], 
                 label = j) 
plt.title('Principal Component Analysis')  
plt.xlabel('1st principal')
plt.ylabel('2nd principal')
plt.legend()  
plt.show() 

explained_variance = [np.abs(i)/np.sum(eigen_vals) for i in eigen_vals]
total_variance = np.cumsum(explained_variance)
plt.figure(figsize=(25,10))
plt.plot(total_variance,marker='o',linestyle='--',color='b')
plt.xticks(np.arange(0,64,step=1))
plt.xlabel('Number of Components')
plt.ylabel('Variance(%)')
plt.title('Mnist Dataset Explained Variance')
plt.axhline(y=0.9,color = 'r',linestyle='-')
plt.grid(axis='x')
plt.show()
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans


data = pd.read_csv('California_Real_Estate.csv', sep=';')
df_real_estate = data.copy()
df_real_estate_nonull = df_real_estate[df_real_estate['Status'] == 1]
scaler = StandardScaler()
df_re_nonull_std = scaler.fit_transform(df_real_estate_nonull)
pca = PCA()
pca.fit_transform(df_re_nonull_std)
pca.explained_variance_ratio_
plt.figure(figsize = (11,6))
components = ['Component 1','Component 2','Component 3','Component 4','Component 5','Component 6','Component 7','Component 8']
var_exp = pca.explained_variance_ratio_
plt.bar(components, var_exp)
plt.title('Explained variance by principal components')
plt.xlabel('Principal components')
plt.ylabel('Explained variance ratio')
plt.show()
plt.figure(figsize=(10,6))
plt.plot(range(1,9),pca.explained_variance_ratio_.cumsum(),marker='o', linestyle='--')
plt.title('Explained variance by components')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()
pca = PCA(n_components=4)
pca.fit(df_re_nonull_std)
pca.components_
df_pca_comp = pd.DataFrame(data=pca.components_,
                        columns=df_real_estate.columns.values,
                        index=['Component 1','Component 2','Component 3','Component 4'])
sns.heatmap(df_pca_comp,
           vmin=-1,
           vmax=1,
           cmap='RdBu',
           annot=True)
plt.yticks([0,1,2,3],
          ['Component 1','Component 2','Component 3','Component 4'],
          rotation=45,
          fontsize=9)

plt.show()
pce_standard = pca.transform(df_re_nonull_std)
nums_rows,nums_cols=pce_standard.shape
x = data.iloc[:]
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of all four principal components
ax.scatter(pce_standard[:, 0], pce_standard[:, 1], pce_standard[:, 2], c=pce_standard[:, 3])

# Labels and title
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('PCA of df_re_nonull_std')

# Display the plot
plt.show()
inertias = []

for i in range(1,195):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(pce_standard)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,195), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()
from sklearn.cluster import KMeans

# Assuming you have already transformed your data using PCA and stored it in pce_standard

# Specify the number of clusters (k)
k = 100  # For example, let's say we want 3 clusters

# Apply k-means clustering
kmeans = KMeans(n_clusters=k, random_state=0)
clusters = kmeans.fit_predict(pce_standard)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of all four principal components
ax.scatter(pce_standard[:, 0], pce_standard[:, 1], pce_standard[:, 2], pce_standard[:, 3],c=clusters,cmap='viridis')

# Labels and title
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('PCA of df_re_nonull_std')

# Display the plot
plt.show()

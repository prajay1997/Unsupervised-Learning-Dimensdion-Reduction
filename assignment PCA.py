import pandas as pd
import matplotlib.pylab as plt

data1 = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Data Mining  Unsupervised Learning Dimension Reduction\wine.csv")

data1.describe()
data1.info()

data = data1.drop(["Type"], axis=1)

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(data)
df_norm.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(df_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));
plt.title('Hierarchical Clustering Dendrogram');
plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering choosing 3 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

data1['clust'] = cluster_labels # creating a new column and assigning it to new column 

data2 = data1.iloc[:, [14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
data2.head()

# Aggregate mean of each cluster
data2.iloc[:, 2:].groupby(data1.clust).mean().transpose()

# creating a csv file 
data2.to_csv("wine hierarchical.csv ", encoding = "utf-8")

import os
os.getcwd()

#### Kmeans Clustering #############



import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist 

# Generating random uniform numbers 
X = np.random.uniform(0,1,50)
Y = np.random.uniform(0,1,50)
df_xy = pd.DataFrame(columns=["X","Y"]) #  Creating the dataframe
df_xy.X = X
df_xy.Y = Y

df_xy.plot(x="X", y ="Y", kind = "scatter")

model1 = KMeans(n_clusters = 3).fit(df_xy)

df_xy.plot(x = "X", y = "Y", c = model1.labels_, kind="scatter", s = 10, cmap = plt.cm.coolwarm)

# Kmeans on University Data set 

datak1 = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Data Mining  Unsupervised Learning Dimension Reduction\wine.csv")
datak1.info()
datak1.describe()
datak = datak1.drop(["Type"], axis = 1)

# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_normk = norm_func(datak.iloc[:, :])

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_normk)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(df_normk)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
datak1['clust'] = mb # creating a  new column and assigning it to new column 

datak1.head()
df_normk.head()

datak2 = datak1.iloc[:,[14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
datak2.head()

datak2.iloc[:, 2:].groupby(datak2.clust).mean().transpose()

datak2.to_csv("wineK means.csv", encoding = "utf-8")

import os
os.getcwd()

### PCA applied for same datasets and then doing the clustering on the dataets##

import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Data Mining  Unsupervised Learning Dimension Reduction\wine.csv")
df.describe()

df = df1.drop(["Type"], axis = 1)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 

# Considering only numerical data 
df.data = df.iloc[:, 1:]

# Normalizing the numerical data 
df_normal = scale(df.data)
df_normal

pca = PCA(n_components = 13)
pca_values = pca.fit_transform(df_normal)

# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var

# PCA weights
pca.components_
pca.components_[0]

# Cumulative variance 
var1 = np.cumsum(np.round(var, decimals = 4) * 100)
var1

# Variance plot for PCA components obtained 
plt.plot(var1, color = "red")

# PCA scores
pca_values

pca_data = pd.DataFrame(pca_values)
pca_data.columns = "comp0", "comp1", "comp2", "comp3", "comp4", "comp5","comp6","comp7","comp8","comp9","comp10","comp11","comp12"
final = pd.concat([df.Type, pca_data.iloc[:, 0:3]], axis = 1)

# Scatter diagram
import matplotlib.pylab as plt
ax = final.plot(x='comp0', y='comp1', kind='scatter',figsize=(12,8));plt.label("pca scatter plot")

# convert the data into csv file
final.to_csv("pca on wine.csv", encoding = "utf-8")


### Hirarchical clustering on PCA data ###

import pandas as pd
import matplotlib.pylab as plt

final1 = pd.read_csv(r"C:\Users\praja\pca on wine.csv")

final1.describe()
final1.info()

final =final1.drop(["Unnamed: 0"], axis=1)

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(final.iloc[:, 1:])
df_norm.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(df_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));
plt.title('Hierarchical Clustering Dendrogram');
plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(final) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

final['clust'] = cluster_labels # creating a new column and assigning it to new column 

final2 = final.iloc[:, [4,0,1,2,3]]
final2.head()

# Aggregate mean of each cluster
final2.iloc[:, 2:].groupby(final2.clust).mean()

# creating a csv file 
final2.to_csv("Hierachical 1 on pca.csv", encoding = "utf-8")
 
import os
os.getcwd()

################ Kmeans Clustering on decomposed data  #####

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist 

# Generating random uniform numbers 
X = np.random.uniform(0,1,50)
Y = np.random.uniform(0,1,50)
df_xy = pd.DataFrame(columns=["X","Y"]) #  Creating the dataframe
df_xy.X = X
df_xy.Y = Y

df_xy.plot(x="X", y ="Y", kind = "scatter")

model1 = KMeans(n_clusters = 3).fit(df_xy)

df_xy.plot(x = "X", y = "Y", c = model1.labels_, kind="scatter", s = 10, cmap = plt.cm.coolwarm)

# Kmeans on wine pca Data set 
a = pd.read_csv(r"C:\Users\praja\pca on wine.csv")
a.info()
a.describe()
b = a.drop(["Unnamed: 0"], axis = 1)

# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(b.iloc[:, 1:])

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters by pca ");plt.ylabel("total_within_SS by pca")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
b['clust'] = mb # creating a  new column and assigning it to new column 
b.head()
df_norm.head()

c = b.iloc[:,[4,0,1,2,3]]
c.head()

c.iloc[:, 2:].groupby(c.clust).mean()

c.to_csv("Kmeans_university on pca data.csv", encoding = "utf-8")

import os
os.getcwd()

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#####

# Q2)

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans


# Generating random uniform numbers 
X = np.random.uniform(0,1,50)
Y = np.random.uniform(0,1,50)
df_xy = pd.DataFrame(columns=["X","Y"]) #  Creating the dataframe
df_xy.X = X
df_xy.Y = Y

df_xy.plot(x="X", y ="Y", kind = "scatter")

model1 = KMeans(n_clusters = 3).fit(df_xy)

df_xy.plot(x = "X", y = "Y", c = model1.labels_, kind="scatter", s = 10, cmap = plt.cm.coolwarm)

# Kmeans on University Data set 
data1 = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Data Mining  Unsupervised Learning Dimension Reduction\heart disease.csv")
data1.info()
data1.describe()
data = data1.drop(["target"], axis = 1)

# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(data)

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 4 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
data1['clust'] = mb # creating a  new column and assigning it to new column 

data1.head()
df_norm.head()

data2 = data1.iloc[:,[14,13,0,1,2,3,4,5,6,7,8,9,10,11,12]]
data2.head()

mean_heart_disease = data2.iloc[:, 2:].groupby(data2.clust).mean()

data2.to_csv("Kmeans_heart deseaes2.csv", encoding = "utf-8")

import os
os.getcwd()


################### PCA ##########################

import pandas as pd
import numpy as np

df1 = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Data Mining  Unsupervised Learning Dimension Reduction\heart disease.csv")
df1.describe()

df1.info()
df = df1.drop(["target"], axis = 1)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 

# Considering only numerical data 
df.data = df.iloc[:, :]

# Normalizing the numerical data 
df_normal = scale(df.data)
df_normal

pca = PCA(n_components = 13)
pca_values = pca.fit_transform(df_normal)

# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var

# PCA weights
pca.components_
pca.components_[0]

# Cumulative variance 
var1 = np.cumsum(np.round(var, decimals = 4) * 100)
var1

# Variance plot for PCA components obtained 
plt.plot(var1, color = "red")

# PCA scores
pca_values

pca_data = pd.DataFrame(pca_values)
pca_data.columns = "comp0", "comp1", "comp2", "comp3", "comp4", "comp5", "comp6","comp7", "comp8","comp9","comp10","comp11","comp12"
final = pd.concat([df1.target, pca_data.iloc[:, 0:10]], axis = 1)

# Scatter diagram
import matplotlib.pylab as plt
ax = final.plot(x='comp0', y='comp1', kind='scatter',figsize=(12,8))

final.to_csv(" heart disease pca.csv", encoding = "utf-8")

####### K  means on PCA data ##################

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist 

# Generating random uniform numbers 
X = np.random.uniform(0,1,50)
Y = np.random.uniform(0,1,50)
df_xy = pd.DataFrame(columns=["X","Y"]) #  Creating the dataframe
df_xy.X = X
df_xy.Y = Y

df_xy.plot(x="X", y ="Y", kind = "scatter")

model1 = KMeans(n_clusters = 3).fit(df_xy)

df_xy.plot(x = "X", y = "Y", c = model1.labels_, kind="scatter", s = 10, cmap = plt.cm.coolwarm)

# Kmeans on University Data set 
heart = pd.read_csv(r"C:\Users\praja\ heart disease pca.csv")
heart.info()
heart.describe()
heart1 = heart.drop(["target"], axis = 1)

# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm_heart = norm_func(heart1)

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm_heart)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 4 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(df_norm_heart)

model.labels_ # getting the labels of clusters assigned to each row 
hb= pd.Series(model.labels_)  # converting numpy array into pandas series object 
heart['clust2'] = hb # creating a  new column and assigning it to new column 

data1.head()
df_norm.head()

heart3 = heart.iloc[:,[11,0,1,2,3,4,5,6,7,8,9,10]]
heart3.head()

mean_heart_pca =heart3.iloc[:, 2:].groupby(heart3.clust2).mean()

heart3.to_csv("Kmeans_heart deseaes pca.csv", encoding = "utf-8")

import os
os.getcwd()

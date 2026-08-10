import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib as jb



x_data= pd.read_csv('Mall_Customers.csv')
print(x_data.head())

x=x_data.iloc[:,[3,4]].values

def assign_cluster(x, centroids):
    m=x.shape[0]
    k=centroids.shape[0]

    c=np.zeros(m)
    for i in range(m):
        distances=[]
        for j in range(k):
            distance=np.sum((x[i] -  centroids[j])**2)
            distances.append(distance)
        c[i]=np.argmin(distances) #Minimize the cost function to get the index of cluster.
    return c

def compute_centroids(x,c,k):
    m,n= x.shape
    centroids=np.zeros((k,n))
    for i in range(k):
        if np.any(c == i): #We check if there are any points assigned to the cluster i. If not, we randomly select a point from the dataset as the new centroid.
            centroids[i] = np.mean(x[c == i], axis=0)

        else:
            centroids[i] = x[np.random.randint(0, m)]
    return centroids

def kmeans(x,k,intial_centroids,max_iters=100):
    m,n=x.shape
    centroids=intial_centroids
    for i in range(max_iters):
        c=assign_cluster(x,centroids)
        centroids=compute_centroids(x,c,k)

    return centroids,c

'''Here we use the elbow method to find the optimal number of clusters. '''
costs=[]
for i in range(1,11):
    initial_centroids=x[np.random.choice(x.shape[0],i,replace=False)]#Intialise random centroids from the dataset.
    centroids,c=kmeans(x,i,initial_centroids)#Using k-means to find the centroids and index of clusters.

    err=0# Calcualting the error for each example in the dataset. The error is the sum of the squared distances between each example and its assigned centroid.
    for j in range(i):
        err+=np.sum((x[c==j]-centroids[j])**2)
    costs.append(err)

plt.plot(range(1,11),costs)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Cost')
plt.xticks(range(1,11))
plt.show()

print("The optimal number of clusters is 5")

'''Now we will use the k means algorith to calculate the actual data needed'''

initial_centroids=x[np.random.choice(x.shape[0],5,replace=False)]
final_centroids,final_c=kmeans(x,5,initial_centroids)

print("Location of final centroids", final_centroids)

colors = ['red', 'blue', 'green', 'purple', 'orange']

for i in range(5):
    cluster_points = x[final_c == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                s=50, c=colors[i], label=f'Cluster {i+1}')

plt.scatter(final_centroids[:, 0], final_centroids[:, 1], 
            s=200, c='black', marker='X', label='Centroids')

plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

x_data['Cluster'] = final_c.astype(int)
cluster_summary = x_data.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean()
print(cluster_summary)

def predict_cluster(p, centroids):
    distances=[]
    for i in centroids:
        distance= np.sum((p - i)**2)
        distances.append(distance)

    return np.argmin(distances)

cluster_labels={0:"Standard Customers (Average Income, Average Spending)",
                1:"Careful Spenders (High Income, Low Spending)",
                2:"Budget-Conscious (Low Income, Low Spending)",
                3:"Target Customers (High Income, High Spending)",
                4:"Impulsive Spenders (Low Income, High Spending)" }

def classify(income, spending_score, centroids, cluster_labels):
    data= np.array([income,spending_score])
    c_i=predict_cluster(data,centroids)
    label= cluster_labels[c_i]

    return c_i, label

jb.dump(final_centroids, "centroids.pkl")
jb.dump(cluster_labels, "cluster_labels.pkl")




    

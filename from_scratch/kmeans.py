import numpy as np

class KMeans:

    def __init__(self, n_clusters=3, epoch=100) -> None:
        self.n_clusters=n_clusters
        self.epoch = epoch

    def fit(self, x_train):
        
        self.x_train = x_train
        self.n_train = x_train.shape[0]

        min_, max_ = np.min(self.x_train, axis=0), np.max(self.x_train, axis=0)
        centroids = [
            np.random.uniform(min_, max_) for _ in range(self.n_clusters)
        ] 
        # train for given number of epochs
        for _ in range(self.epoch):
            centroids = self.update_centroids(centroids)

        self.centroids = centroids
        return self


    def update_centroids(self, centroids):

        # store cluster lable 
        clusters = np.zeros(self.n_train)

        # assign each points to 
        for i in range(self.n_train):
            p = self.x_train[i]  # isolate a data point
            dists = [self.euclidean(p, centroid) for centroid in centroids]
            clusters[i] = np.argmin(dists) 
        # update the centroids 

        for i in range(self.n_clusters):
            points = self.x_train[np.array(clusters) == i]
            centroids[i] = points.mean(axis=0)
        return centroids

    def predict(self, x_test):
        
        n_test = x_test.shape[0]
        clusters = np.zeros(n_test)

        for i in range(self.n_train):
            p = x_test[i]  # isolate a data point
            dists = [self.euclidean(p, centroid) for centroid in self.centroids]
            clusters[i] = np.argmin(dists) 
        return clusters


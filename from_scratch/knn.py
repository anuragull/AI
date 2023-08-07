
# create knn 
"""
Algotihtm

1. distance function 
2. data
3. find neearest neighbor based on distance 

"""
from math import sqrt
import numpy as np

def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)

class KNN:

    def __init__(self, data) -> None:
        self.data = data
        
    def fit(self, x, k):

        distances = [euclidean_distance(x, row) for row in self.data]
        # sort by distance 
        k_idx = np.argsort(distances)[:k]
        result = [self.data[i] for i in k_idx]
        return result
    

def test():

    data = [ ( 0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6)]
    k = 2
    x = (0,5)
    knn = KNN(data)
    result = knn.fit(x, k)
    print(result)

test()

import numpy as np
class LogisticRegression:

    def __init__(self, lr=0.01, epoch=10) -> None:
        self.w = None
        self.b = 0
        self.lr = lr
        self.epoch = epoch

    def fit(self, X, y):
        """
        """
        # get features and rows 
        self.n_obs, self.n_feat = X.shape
        self.w = np.zeros(self.n_feat)

        self.X = X
        self.y = y
        losses = []
        for i in range(self.epoch):

            y_pred = self.predict(X)

            loss = self.compute_loss(y, y_pred)
            losses.append(loss)
            print(i, loss)
            # grad_w
            grad_w = -np.dot(self.X.T, (self.y - y_pred)) / self.n_obs
            # grad_b
            grad_b = -np.sum(self.y - y_pred) / self.n_obs
            self.w = self.w - self.lr*grad_w 
            self.b = self.b - self.lr*grad_b

        return self

    def predict(self, X):

        z = X.dot(self.w) + self.b

        # sigmoid 
        p = 1 / (1 + np.exp(-z))

        return np.where(p < 0.5, 0, 1)
    
    def compute_loss(self, y_true, y_pred):
    # binary cross entropy
        y_zero_loss = y_true * np.log(y_pred + 1e-9)
        y_one_loss = (1-y_true) * np.log(1 - y_pred + 1e-9)
        return -np.mean(y_zero_loss + y_one_loss)

def test():
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=1,
        n_clusters_per_class=1,
    )
    clf = LogisticRegression().fit(X, y)
    print(clf.predict(X[:2, :]))

test()

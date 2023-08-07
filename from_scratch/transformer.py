"""
"""
import numpy as np

class Transformer:

    def __init__(self) -> None:
        pass

    def softmax(self, x):
        """
        softmax implements to nomalizes the input vector to probability,  
        probability distribution  that is proportional to the exponential of the input numbers
        ref : https://wandb.ai/krishamehta/softmax/reports/How-to-Implement-the-Softmax-Function-in-Python--VmlldzoxOTUwNTc 
        1. get max
        2. exp x-max of each element
        3. divsor max(np.sum)
        """
        assert len(x.shape) == 2 # why? 

        s = np.max(x, axis=1)
        ex = np.exp(x-s)

        div = np.sum(ex, axis=1)

        return ex/ div

    def cross_entorpy(self):
        """
        cross entropy is measure of difference between predictd didtribution vs actuall
        loss -- evaluates how well the model fits the data 

        """
        pass

    def activation(self, x, kind):

        if kind == "gelu":
            """
            https://arxiv.org/abs/1606.08415v5
            The GELU nonlinearity weights inputs by their value, rather than gates inputs by their sign as in ReLUs 

            """
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        if kind == "relu":
            return np.maximum(0, x)

        if kind == "sigmoid":
            return 1/(1 + np.exp(-x))

    def normalization(self, x, kind):
        pass



def test():
    trans = Transformer()
    x1 = np.array([[-1,-2, -4, 1, 2, 3, 6]])
    
    print("softmax", trans.softmax(x1))
    print("sigmoid", trans.activation(x1, kind="sigmoid"))
    print("relu", trans.activation(x1, kind="relu"))
    print("gelu", trans.activation(x1, kind="gelu"))



test()
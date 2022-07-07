import numpy as np
def elu(x):
    #alpha = 1.0 #Hyperparameter, can be of any value
    alpha = 0.7
    if x>0:
        print(max(0,x)) #Why? Task.
    else:
        print(alpha * (np.exp(x) - 1))


if __name__ == '__main__':
    elu(-100)
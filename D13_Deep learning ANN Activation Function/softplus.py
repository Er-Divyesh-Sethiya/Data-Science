import numpy as np

def softplus(x):
    print(np.log(np.exp(x) + 1))

if __name__ == '__main__':
    #softplus(-100)
    softplus(-3)
    #Not for relu, only > 0.
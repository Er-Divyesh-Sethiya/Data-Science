def leaky_relu(x):
    if x>0:
        print(max(0,x)) #Why? Task.
    else:
        print(0.01*x)


if __name__ == '__main__':
    leaky_relu(-300)
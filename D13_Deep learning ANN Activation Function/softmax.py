#Softmax is going to convert all your values into normal probabilty distibution where it ranges between 0 to 1.
#Where sigmoid can be used for bnary classification and softmax since it is a normal prob distribution, it is used for multiclass classification.
import numpy as np

x = np.array([-2,3,4,60])

#Formula: e^x/summation of e^x
numerator = np.exp(x)
print(numerator)

denominator = np.sum(np.exp(x))
print(denominator)

softmax = numerator/denominator
print(softmax)

max_neuron = np.argmax(softmax)
print(max_neuron)

#Thus it is useful for Multiclass-classification.


import math
#
# #Approach 1
#
# #Step 3: Define sigmoid function
# def tanh(x):
#     y = (math.exp(x) - math.exp(-x))/(math.exp(x) + math.exp(-x)) #Include math module
#     return y
#
# #Step 2
# def tanh_activate(input,weights,bias):
#     #perform net input
#     h = 0
#     for x,w,b in zip(input,weights,bias):
#         h += ((x*w) + b) # Summarization = h = x*w + x*w + x*w
#         print(h)
#     print(h)
#     # perform activation
#     return tanh(h) #Go to step 3
#
#
# #Step 1
# if __name__ == '__main__':
#     input   = [.5,.3,.2]
#     weights = [.4,.7,.2]
#     bias    = [.1,.4,.5]
#     output  = tanh_activate(input,weights,bias)#Got to step 1
#     print(output)

# ************************************************
#  Approach 2
#     Call tanh, value<-1 and value>1
def tanh(x):
    y = (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))  # Include math module
    print(y)
    return y

if __name__ == '__main__':
    tanh(40)

#APPROACH 1

# #Step3: Write Relu AF
# def relu(x):
#     if x>0:
#         return max(0,x)
#     else:
#         return 0
#
# #Step2: Write Relu activation function
# def relu_activate(input,weights,bias):
#     h = 0
#     for x,w,b in zip(input,weights,bias):
#         h += (w*x) + b
#     print('Before Applying AF: ',h)
#
#     #Applying AF
#     return relu(h) #Go to step3
#
# #Step1: Start with main function
# if __name__ == '__main__':
#     input = [.5, -.3, .2]
#     weights = [.4, -.7, .2]
#     bias = [.1, -.7, .5]
#     output = relu_activate(input,weights,bias) #Perform Step 2
#     print(output)

#*****************************************************************

# APPROACH 2
def relu(x):
    if x>0:
        print(max(0,x)) #Why? Task.
        return max(0,x)
    else:
        print(0)
        return 0

if __name__ == '__main__':
    #relu(5)
    relu(300)
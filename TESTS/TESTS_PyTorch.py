# torch installation works only with Python 9.2 for some reason ???
import torch as t
import math

#31.07.22
#   Tutorial 1 - Introduction
#Autograd - automated differentation engine
#loss.backwards()

#neural network as a class with self.functions
#teaching the model based on imported datasets (TorchVision)

#   Tutorial 2 - Tensors
import torch.cuda

print(t.empty(3,4))

z = t.zeros(5,3)      # zeros(x,y)    x=#rows y=#cols
print(z)
print(z.dtype)

t.manual_seed(1729)
random3 = t.rand(2,3)
print(random3)

print(t.rand(1))      # rand(x) same as in matlab

#t.empty_like(tensor)       #saves the dimension of given tensor, can be used as a 'size-template' to create matrices of same size
#tensor.shape               #returns the size of given tensor
#t.tensor(data)             #creates a copy of the data

#while creating the tensor one can specify its data type within the function    e.g.: t.ones( (2,3), dtype=int32)
#changing the data type with    new = old.to(torch.dtype)
#changing of device is also possible with this function     t.torch('cpu'/'cuda' or my_device)

#tensor broadcasting

#t.eq(t1,t2)                #boolwise comparison of 2 tensors

#using mathematical operation on tensors doesn't overwrite the primary tensor itself, the function creates a separate entity and returns its value
#if one wants to overwrite the value, use the underscore after the function call        e.g.: t.sin_(tensor)

# assert function?
#t.clone()              #copys a tensor as a separate object
#t.detach()             #copys a tensor without autograd on (new tensor doesn't track its primary tensor)

if t.cuda.is_available():
    print('YES!')
else:
    print("NO")
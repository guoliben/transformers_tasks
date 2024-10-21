import torch

# print(torch.empty(5))
# print(torch.empty(2,3))
# print(torch.rand(2,3))
# print(torch.randn(2,3))
# print(torch.zeros(2,3))
# print(torch.arange(2))

#
#
# array = [[1.0, 3.8, 2.1], [8.6, 4.0, 2.4]]
# print(torch.tensor(array))
#
#
# import numpy as np
# array = np.array([[1.0, 3.8, 2.1], [8.6, 4.0, 2.4]])
# print(torch.from_numpy(array))
# # print( torch.rand(2, 3).cuda())
#
# tensor = torch.rand(2, 3).to(torch.device('cpu'))
# print(tensor)
#
#
# tensor = torch.rand(2, 3).to(torch.device('mps'))  #tensor([[0.9568, 0.8641, 0.9175], [0.0174, 0.3919, 0.5951]], device='mps:0')
# print(tensor)
#
# x = torch.tensor([1, 2, 3], dtype=torch.double)
# y = torch.tensor([4, 5, 6], dtype=torch.double)
# print(x + y)
# # tensor([5., 7., 9.], dtype=torch.float64)
# print(x - y)
# # tensor([-3., -3., -3.], dtype=torch.float64)
# print(x * y)
# # tensor([ 4., 10., 18.], dtype=torch.float64)
# print(x / y)
# # tensor([0.2500, 0.4000, 0.5000], dtype=torch.float64)
#
# print(x.dot(y))
# print(x.sin())
# print(x.exp())
# x = torch.tensor([[1, 2, 3], [ 4,  5,  6]], dtype=torch.double)
# y = torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.double)
# print(torch.cat((x, y), dim=0))
# print(torch.cat((x, y), dim=1))
#
#
# x = torch.tensor([2.])
# y = torch.tensor([3.])
# z = (x + y) * (y - 2)
# print(z)


import torch
import timeit

# M = torch.rand(1000, 1000)
# print(timeit.timeit(lambda: M.mm(M).mm(M), number=5000))
#
# # N = torch.rand(1000, 1000).cuda()
# N = torch.rand(1000, 1000).to(torch.device('mps'))
# print(timeit.timeit(lambda: N.mm(N).mm(N), number=5000))



# x = torch.tensor([2.], requires_grad=True)
# y = torch.tensor([3.], requires_grad=True)
# z = (x + y) * (y - 2)
# print(x, y, z)
# # tensor([5.], grad_fn=<MulBackward0>)
# print(z.backward())
# print(x, y, z)
#
# print(x.grad, y.grad) # x.grad
# # tensor([1.]) tensor([6.])

#
# import torch
# x = torch.tensor([1, 2, 3, 4, 5, 6])
# print(x, x.shape)
# # tensor([1, 2, 3, 4, 5, 6]) torch.Size([6])
# print(x.view(2, 3)) # shape adjusted to (2, 3)
# print(x.view(3, 2)) # shape adjusted to (3, 2)
# print(x.view(-1, 3)) # -1 means automatic inference
# x = x.view(3, 2)
# print(x, x.shape)


x = torch.arange(1, 4).view(3, 1)
y = torch.arange(4, 6).view(1, 2)
# print(torch.arange(4, 6))
print(x)
print(y)
print(x + y)
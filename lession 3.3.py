import numpy as np

def sigmoid(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def max_1(x, deriv=False):
    if (deriv == True):
        return 1
    return max(x)

error_glb = np.array([])
#X = np.array([0, 1, 2])
#X = np.array([15, 5, 15])
X = np.array([0, 1, 1])
Y = np.array([1])
#w1 = np.array([[2,2],[2,2],[2,2]])
#w1 = np.array([[8,10],[7,10],[8,9]])
#w1 = np.array([[0.2,0.2],[0.9,0.3],[0.6,0.7]])
w1 = np.array([[0.7,0.8],[0.2,0.3],[0.7,0.6]])
#w2 = np.array([[1],[1]])
#w2 = np.array([[10],[9]])
#w2 = np.array([[0.2],[0.5]])
w2 = np.array([[0.2],[0.4]])


# проходим вперёд по слоям 0, 1 и 2
l0 = X
#l1 = sigmoid(np.dot(l0, w1))
#print(w1)
#print(w1[::1,:2:2])
#print(w1[::1,1:2:2])
l11 = max_1(np.dot(l0, w1[::1,:2:2]))
l12 = sigmoid(np.dot(l0, w1[::1,1:2:2]))
l1 = np.append(l11, l12)
l2 = sigmoid(np.dot(l1, w2))
print(l2)
l2_error = l2-Y
error_glb = np.append(error_glb, np.mean(np.abs(l2_error)))

l2_delta = l2_error * sigmoid(l2, deriv=True)

l1_error = l2_delta.dot(w2.T)
l1_delta = l1_error[0] * max_1(l11, deriv=True)
l1_delta = np.append(l1_delta, l1_error[1] * sigmoid(l12, deriv=True))

print (l1_delta)

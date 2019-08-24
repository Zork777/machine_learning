import numpy as np
def sigmoid_prime(x):
    return 1 / (1 + np.exp(-x))

def get_error(deltas, sums, weights):
    """
    compute error on the previous layer of network
    deltas - ndarray of shape (n, n_{l+1})
    sums - ndarray of shape (n, n_l)
    weights - ndarray of shape (n_{l+1}, n_l)

    Сигнатура: get_error(deltas, sums, weights), где deltas — ndarray формы (n, nl+1),
    содержащий в i-й строке значения ошибок для i-го примера из входных данных,
    sums — ndarray формы (n, nl), содержащий в i-й строке значения сумматорных функций
    нейронов l-го слоя для i-го примера из входных данных, weights — ndarray формы (nl+1, nl),
    содержащий веса для перехода между l-м и l+1-м слоем сети. Требуется вернуть вектор
    δl — ndarray формы (nl, 1); мы не проверяем размер (форму) ответа, но это может помочь вам сориентироваться.
    Все нейроны в сети — сигмоидальные. Функции sigmoid и sigmoid_prime уже определены.
    """
    # here goes your code

    n, nl = sums.shape
    delt = (deltas.dot(weights))*sigmoid_prime(sums)
    otvet =  delt.mean(axis = 0)
    otvet1 = otvet.reshape(nl,1)
    print(otvet1)
    return otvet1

deltas = np.array([[0.3, 0.2], [0.3, 0.2]])
sums = np.array([[0, 1, 1],[0, 2, 2]])
weights = np.array([[0.7, 0.2, 0.7],[0.8, 0.3, 0.6]])

get_error(deltas, sums, weights)
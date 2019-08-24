import numpy as np



class Perceptron:

    def __init__(self, w, b):
        """
        Инициализируем наш объект - перцептрон.
        w - вектор весов размера (m, 1), где m - количество переменных
        b - число
        """

        self.w = w
        self.b = b

    def forward_pass(self, single_input):
        """
        Метод рассчитывает ответ перцептрона при предъявлении одного примера
        single_input - вектор примера размера (m, 1).
        Метод возвращает число (0 или 1) или boolean (True/False)
        """

        result = 0
        for i in range(0, len(self.w)):
            result += self.w[i] * single_input[i]
        result += self.b

        if result > 0:
            return 1
        else:
            return 0

    def vectorized_forward_pass(self, input_matrix):
        """
        Метод рассчитывает ответ перцептрона при предъявлении набора примеров
        input_matrix - матрица примеров размера (n, m), каждая строка - отдельный пример,
        n - количество примеров, m - количество переменных
        Возвращает вертикальный вектор размера (n, 1) с ответами перцептрона
        (элементы вектора - boolean или целые числа (0 или 1))
        """

        result = input_matrix.dot(self.w)+self.b
        return result > 0

    def train_on_single_example(self, example, y, in_signal):
        """
        принимает вектор активации входов example формы (m, 1)
        и правильный ответ для него (число 0 или 1 или boolean),
        обновляет значения весов перцептрона в соответствии с этим примером
        и возвращает размер ошибки, которая случилась на этом примере до изменения весов (0 или 1)
        (на её основании мы потом построим интересный график)
        """
        deltw = (y - example)
        self.w = self.w + deltw*in_signal
        self.b += deltw
        return deltw



        ## Этот метод необходимо реализовать
        pass

    def train_until_convergence(self, input_matrix, y, max_steps=1e8):
        """
        input_matrix - матрица входов размера (n, m),
        y - вектор правильных ответов размера (n, 1) (y[i] - правильный ответ на пример input_matrix[i]),
        max_steps - максимальное количество шагов.
        Применяем train_on_single_example, пока не перестанем ошибаться или до умопомрачения.
        Константа max_steps - наше понимание того, что считать умопомрачением.
        """
        i = 0
        errors = 1
        while errors and i < max_steps:
            i += 1
            errors = 0
            for example, answer in zip(input_matrix, y):
                example = example.reshape((example.size, 1))
                error = self.train_on_single_example(example, answer)
                errors += int(error)  # int(True) = 1, int(False) = 0, так что можно не делать if

def train_my_function(self, input_matrix, y, max_steps=100):
    pass

def create_perceptron(m):
    """Создаём перцептрон со случайными весами и m входами"""
    w = np.random.random((m, 1))
    return Perceptron(w, 1)


x = np.array([[0,0],[1,1],[0,1],[1,0]])
y = [0, 0, 1, 1]
#x = np.array([[0,0],[1,1],[0,1]])
#y = [0,0,1]


p5 = create_perceptron(2)
p6 = create_perceptron(2)
p9 = create_perceptron(2)


print(x)

i = 0
errors = 1
while errors and i < 1000:
    i += 1
    errors = 0
    for example, answer in zip(x, y):
        example = example.reshape((example.size, 1))
        y7 = p5.vectorized_forward_pass(example.T)
        y8 = p6.vectorized_forward_pass(example.T)
        y_ = np.hstack((y7, y8))
        y3 = p9.vectorized_forward_pass(y_)
#обучение
        delta = answer - y3
        xxx = y_*delta
        p9.w += xxx

        deltw3 = p9.train_on_single_example(y3, answer, y_)
        deltw2 = p6.train_on_single_example(y8, deltw3, example.T)
        deltw1 = p5.train_on_single_example(y7, deltw3, example.T)
        errors += abs(int(deltw3))

print("1-", p5.w[0])
print("2-", p6.w[0])
print("3-", p5.w[1])
print("4-", p6.w[1])
print("5-", p5.b)
print("6-", p6.b)
print("7-", p9.w[0])
print("8-", p9.w[1])
print("9-", p9.b)

for example, answer in zip(x, y):
    example = example.reshape((example.size, 1))
    y7 = p5.vectorized_forward_pass(example.T)
    y8 = p6.vectorized_forward_pass(example.T)
    y_ = np.hstack((y7, y8))
    y3 = p9.vectorized_forward_pass(y_)
    print("****", example.T)
    print(int(y3))



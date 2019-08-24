import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

error_glb = np.array([])

def animate(i1, x1, y1, m1 = 5):
    # отображение i-ого кадра
    '''
    кадр анимации,
    i - номер кадра
    x, y - данные
    m - число пропускаемых кадров
    '''

    plt.clf()  # стираю предыдуший кадр
    plt.plot(x1[:i1 * m1], y1[:i1 * m1], color='black', lw=1)
    plt.plot(x1[i1 * m1], y1[i1 * m1], 'ro', ms=10)

    plt.xlim(-1000, y1.size)
#    plt.ylim(-0.4, 1.2)
    plt.grid(ls='solid', lw=0.1)

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

        if result > 0.5:
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

        ## Этот метод необходимо реализовать
        return (input_matrix.dot(self.w) + self.b) > 0.5

    def train_on_single_example(self, example, y):
        """
        принимает вектор активации входов example формы (m, 1) 
        и правильный ответ для него (число 0 или 1 или boolean),
        обновляет значения весов перцептрона в соответствии с этим примером
        и возвращает размер ошибки, которая случилась на этом примере до изменения весов (0 или 1)
        (на её основании мы потом построим интересный график)
        """

        ## Этот метод необходимо реализовать
        deltw = (y - vectorized_forward_pass(self, example))
        self.w += deltw * example
        self.b += deltw
        return deltw

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


def create_perceptron(m):
    """Создаём перцептрон со случайными весами и m входами"""
    w = np.random.random((m, 1))
    return Perceptron(w, 0)


x = np.array([[0,0],[1,1],[0,1],[1,0]])
y = [0, 0, 1, 1]

p5 = create_perceptron(2)
p6 = create_perceptron(2)
p9 = create_perceptron(2)

y5 = p5.vectorized_forward_pass(x)
y6 = p6.vectorized_forward_pass(x)
y_ = np.hstack((y5, y6))
otvet = p9.vectorized_forward_pass(y_)
#print(otvet)

#train
i = 0
errors = 1
while errors and i < 10000:
    i += 1
    errors = 0
    for example, answer in zip(x, y):
#        example = example.reshape((example.size, 1))
#        y5 = example.dot(p5.w) + p5.b
        y5 = p5.vectorized_forward_pass(example)
        y6 = p6.vectorized_forward_pass(example)
#        y6 = example.dot(p6.w) + p6.b
        y_ = np.hstack((y5, y6))
        error9 = answer - p9.vectorized_forward_pass(y_)
        error_glb = np.append(error_glb, np.mean(np.abs(error9)))
        errors += abs(int(error9))  # int(True) = 1, int(False) = 0, так что можно не делать if
        error_ = error9 * p9.w
        p55 = error_ * example.reshape(error_.size, 1)
        p5.w += p55
#        p5.b += error9
        p6.w += error_ * example.reshape(error_.size, 1)
#        p6.b += error9
        p99 = (error9 * y_).reshape(2,1)
        p9.w += p99
#        p9.b += error9

x = np.array([[0,0],[1,1],[0,1],[1,0]])
#x = np.array([[1,1]])
#for example, answer in zip(x, y):
#    example = example.reshape((example.size, 1))
y5 = x.dot(p5.w) + p5.b
y6 = x.dot(p6.w) + p6.b
y_ = np.hstack((y5, y6))
otvet = p9.vectorized_forward_pass(y_)
print("****", x)
print(otvet)

m = 1 # коэффициент ускорения
n = len(error_glb) # число точек графика
fig = plt.figure(figsize=(6,4))
#plt.plot(np.arange(0, error_glb.size), error_glb, lw=1, color='black')
#plt.grid(ls='solid', lw=0.1)
fig = plt.figure(figsize=(6,4))
anim = animation.FuncAnimation(fig, animate,  fargs=(np.arange(0, error_glb.size), error_glb, m), frames=int(n/m), interval=1, repeat=False)
plt.show()
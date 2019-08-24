import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

    plt.xlim(-1000, 60000)
#    plt.ylim(-0.4, 1.2)
    plt.grid(ls='solid', lw=0.1)

def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

error_glb = np.array([])
#X = np.array([[0, 0, 0],
#              [0, 1, 0],
#              [1, 0, 0],
#              [1, 1, 0]])

#X = np.array([[0, 0],
#              [0, 1],
#              [1, 0],
#              [1, 1]])
X = np.array([0,1,2])

y = np.array([1])

#y = np.array([[0],
#              [1],
#              [1],
#              [0]])

np.random.seed(1)

# случайно инициализируем веса, в среднем - 0
#syn0 = np.random.random((2, 2))
#syn1 = np.random.random((2, 1))
syn0 = np.array([[2,2],[2,2],[2,2]])
syn1 = np.array([[1],[1]])



for j in range(60000):

    # проходим вперёд по слоям 0, 1 и 2
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    # как сильно мы ошиблись относительно нужной величины?
    l2_error = y - l2
    error_glb = np.append(error_glb, np.mean(np.abs(l2_error)))

    if (j % 10000) == 0:
        print ("Error:" + str(np.mean(np.abs(l2_error))))

    # в какую сторону нужно двигаться?
    # если мы были уверены в предсказании, то сильно менять его не надо
    l2_delta = l2_error * nonlin(l2, deriv=True)

    # как сильно значения l1 влияют на ошибки в l2?
    l1_error = l2_delta.dot(syn1.T)

    # в каком направлении нужно двигаться, чтобы прийти к l1?
    # если мы были уверены в предсказании, то сильно менять его не надо
    l1_delta = l1_error * nonlin(l1, deriv=True)
    print (l1_delta)

    syn0[2][0] += l0[2] * l1_delta[0]
    syn0[2][1] += l0[2] * l1_delta[1]

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

x = np.array([[0,0],[1,0],[0,1],[1,1]])
#x = np.array([[1,1]])
l0 = x
l1_= np.dot(l0, syn0)
l1 = nonlin(l1_)
l2 = nonlin(np.dot(l1, syn1))

print(syn0)
print(syn1)
#print(l1)
print(x)
print(l2)

n = len(error_glb) # число точек графика
#x = np.linspace(0., 2.*np.pi, n)
#y = np.sinc(x)
fig = plt.figure(figsize=(6,4))
plt.plot(np.arange(1, 60001), error_glb, lw=1, color='black')
plt.grid(ls='solid', lw=0.1)


m = 100 # коэффициент ускорения
#fig = plt.figure(figsize=(6,4))


#anim = animation.FuncAnimation(fig, animate,  fargs=(np.arange(1, 60001), error_glb, m), frames=int(n/m), interval=1, repeat=False)
plt.show()
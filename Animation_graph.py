import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


#def init():
# подготовка анимации
 #   pass


def animate(i, x, y, m = 5):
    # отображение i-ого кадра
    '''
    кадр анимации,
    i - номер кадра
    x, y - данные
    m - число пропускаемых кадров
    '''

    plt.clf()  # стираю предыдуший кадр
    plt.plot(x[:i * m], y[:i * m], color='black', lw=1)
    plt.plot(x[i * m], y[i * m], 'ro', ms=10)

#    plt.xlim(0, 7)
#    plt.ylim(-0.4, 1.2)
    plt.grid(ls='solid', lw=0.1)

# подготовка данных для анимации
# создание объектов листа и хотя бы одного рисунка
n = 501 # число точек графика
x = np.linspace(0., 2.*np.pi, n)
y = (x)
fig = plt.figure(figsize=(6,4))
plt.plot(x, y, lw=3, color='black')
plt.grid(ls='solid', lw=0.2)
#plt.show()


m = 10 # коэффициент ускорения
fig = plt.figure(figsize=(6,4))


anim = animation.FuncAnimation(fig, animate,  fargs=(x, y, m), frames=int(n/m), interval=1, repeat=False)
plt.show()
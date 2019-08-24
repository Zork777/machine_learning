import numpy as np


x = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

y99 = np.array([[0],
              [1],
              [1],
              [1]])

y0 = np.array([[0],
              [1],
              [0],
              [0]])

y1 = np.array([[0],
              [0],
              [1],
              [0]])

syn0 = np.array([[1, 2],
                 [1, 2]])
syn1 = np.array([1, 2])

p5 = 0
p6 = 0
p9 = 0

gerror = 1
while gerror != 0:
    gerror = 0
    for n in range(4):
#        print (abs(x[n][0]-1), x[n][1])
#проверка AND Y5
        error5=1
        while error5 != 0:
            error5 = 0
            y5 = (x[n][0] * syn0[0][0] + x[n][1] * syn0[1][0] + p5) > 0
            y6 = (x[n][0] * syn0[0][1] + x[n][1] * syn0[1][1] + p6) > 0
            y9 = (y5 * syn1[0] + y6 * syn1[1] + p9) > 0
            error5 = y[n] - y9
            gerror += abs(error5)
            syn1[0] += error5 * x[n][0]
            syn1[1] += error5 * x[n][1]
            syn0[0][0] += error5
            syn0[1][0] += error5
            syn0[0][1] += error5
            syn0[1][1] += error5
#            p5 += error5*0.1
#        print(y5)
print ("OTVET")
for n in range(4):
    print (x[n][0], x[n][1])
    y5 = (x[n][0] * syn0[0][0] + x[n][1] * syn0[1][0] + p5)
    y6 = (x[n][0] * syn0[0][1] + x[n][1] * syn0[1][1] + p6)
    y9 = (y5 * syn1[0] + y6 * syn1[1] + p9) > 0
    print(y9)


print("**ответ****")
print(syn0)
print(syn1)
print(p5)
print(p6)
print(p9)

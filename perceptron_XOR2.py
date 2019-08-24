enters = [0, 0]
hidden_layer = [0, 0]
synapses_hidden = [[0.3, 1.3], [0.5,0.1]]
synapses_output = [0.5, 0.1]
output = 0

learn = [[0, 1], [1, 0], [0, 0],[1, 1]]
learn_answers = [1, 1, 0, 0]

def sum():
    global hidden_layer
    global synapses_hidden
    global synapses_output
    global output
    global enters

    for i in range (len(hidden_layer)):
        hidden_layer[i] = 0
        for j in range(len(enters)):
            hidden_layer[i] += synapses_hidden[j][i] * enters[j]
        if hidden_layer[i] > 0:
            hidden_layer[i] = 1
        else:
            hidden_layer[i] = 0
    output = 0
    for i in range (len(hidden_layer)):
        output += synapses_output[i] * hidden_layer[i]
    if output > 0:
        output = 1
    else:
        output = 0

gError = 1 # глобальная ошибка
errors = [0,0] # слой ошибок

while gError > 0:
    gError = 0 # обнуляем
    for p in range(len(learn)):
        for i in range(len(enters)):
            enters[i] = learn[p][i] # подаём об.входы на входы сети
        # запускаем распространение сигнала
        sum()
        error = learn_answers[p] - output # получаем ошибку
        gError += abs(error) #записываем в глобальную
        print(gError)
        for i in range(len(errors)):
            errors[i] = error * synapses_output[i] # передаём ошибку на слой ошибок
          # по связям к выходу
        for i in range(len(enters)):
            for j in range(len(hidden_layer)):
                synapses_hidden[i][j] += 0.1*errors[i] * enters[j] # меняем веса
        for i in range(len(synapses_output)):
            synapses_output[i] += 0.1*error * hidden_layer[i] # меняем веса

for p in range(len(learn)):
    for i in range(len(enters)):
        enters[i] = learn[p][i] # записываем входы
    sum() # распространяем сигнал
    print(enters)
    print(output) # выводим ответы
print("hidden_layer-", hidden_layer)
print("synapses_hidden-", synapses_hidden)
print("synapses_output-", synapses_output)
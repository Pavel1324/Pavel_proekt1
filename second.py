# библиотека для удобных операций с массивами
import numpy
# библиотека scipy.special с сигмоидой
import scipy.special
# библиотека для графического отображения массивов
import matplotlib.pyplot


class Neural_Network:
    def __init__(self, layer_input, layer_latent, layer_out, learn):
        self.i_layer = layer_input  # количество узлов во входном, скрытом и выходном слое нейросети
        self.l_layer = layer_latent
        self.o_layer = layer_out
# Матрицы весовых коэффициентов(начальный вес) связей inp_lat (между входным и скрытым
# слоями) и lat_out (между скрытым и выходным слоями).
# Весовые коэффициенты связей между узлом i и узлом j следующего слоя обозначены как
# wll w21
# wl2 w22 и так далее
# С помощью numpy.random.normal весовые коэффициенты выбираются
# из нормального распределения с центром в нуле и со стандартным отклонением,
# величина которого обратно пропорциональна корню квадратному из количества входящих связей на узел
        self.inp_lat = numpy.random.normal(0.0, pow(self.l_layer, -0.5), (self.l_layer, self.i_layer))
        self.lat_out = numpy.random.normal(0.0, pow(self.o_layer, -0.5), (self.o_layer, self.l_layer))
# коэффициент обучения
        self.lr = learn
# использование сигмоиды в качестве функции активации ( принимает х возвращает сигмоиду)
        self.sigmoid = lambda x: scipy.special.expit(x)
        pass
# тренировка
    def train(self, list_inp, targ) :
# преобразование списка входных значений
# в двухмерный массив
        inputs = numpy.array(list_inp, 2).T
        targets = numpy.array(targ, 2).T
# рассчитаем входящие сигналы для скрытого слоя
        latent_input = numpy .dot (self .inp_lat, inputs)
# рассчитаем исходящие сигналы для скрытого слоя
        latent_output = self.sigmoid(latent_input)
# рассчитаем входящие сигналы для выходного слоя
        final_input = numpy.dot(self.lat_out, latent_output)
# рассчитаем исходящие сигналы для выходного слоя
        final_output = self.sigmoid(final_input)
# ошибки выходного слоя = целевое значение - фактическое значение
        output_errors = targets - final_output
# ошибки скрытого слоя - это ошибки output_errors,
# распределенные пропорционально весовым коэффициентам связей
# и рекомбинированные на скрытых узлах
        hidden_errors = numpy.dot(self.lat_out.T, output_errors)
# обновим весовые коэффициенты для связей между
# скрытым и выходным слоями
        self.lat_out += self.lr * numpy.dot((output_errors * final_output * (1.0 - final_output)), numpy.transpose(latent_output))
# обновим весовые коэффициенты для связей между
# входным и скрытым слоями
        self.inp_lat += self.lr * numpy.dot((hidden_errors * latent_output * (1.0 - latent_output)), numpy.transpose(inputs))

        pass
# Функция принимает в качестве аргумента входные данные нейронной сети а возвращает ее выходные данные
    def enterend(self, list_inp):
# преобразовываем список входных значений в двухмерный массив и
# транспонируем его( зачем это нужно будет в описании)
        inputs = numpy.array(list_inp, 2).T
# определим входящие сигналы для скрытого слоя
        latent_input = numpy.dot(self.inp_lat, inputs)
# определим исходящие сигналы для скрытого слоя
        latent_output = self.sigmoid(latent_input)
# определим входящие сигналы для выходного слоя
        final_input = numpy.dot(self.lat_out, latent_output)
# определим исходящие сигналы для выходного слоя
        final_output = self.sigmoid(final_input)

        return final_output

# количество входных, скрытых и выходных узлов
layer_input = 784
layer_latent = 100
layer_out =10
# коэффициент обучения равен 0,3
learn =0.2
# создаём экземпляр нейронной сети
n = Neural_Network(layer_input, layer_latent, layer_out, learn)
# загрузим в список тестовый набор данных CSV-файла набора MNIST(библиотека изображений)
training_data_file = open("mnist_dataset/mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
# тренировка нейронной сети
# переменная krugsort указывает, сколько раз тренировочный
# набор данных используется для тренировки сети
krugsort = 5
for _ in range(krugsort):
# перебрать все записи в тренировочном наборе данных
    for record in training_data_list:
# получим список значений
        all_values = record.split(',')
# масштабируем и сместим входные значения
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
# создадим целевые выходные значения (все равны 0,01, за исключением
# маркерного значения, равного 0,99)
        targets = numpy.zeros(layer_out) + 0.01
# all_values[0] - целевое маркерное значение для данной записи
        targets[int(all_values[0])] =0.99
        n.train(inputs, targets)
        pass
    pass

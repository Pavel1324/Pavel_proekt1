# библиотека для удобных операций с массивами
import numpy
# библиотека scipy.special с сигмоидой
import scipy.special

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
# использование сигмоиды в качестве функции активации ( принимает х возвращает сигмоиду), просто другой способ задания функций
        self.sigmoid = lambda x: scipy.special.expit(x)
        pass
# тренировка
    def train() :

        pass
# Функция принимает в качестве аргумента входные данные нейронной сети а возвращает ее выходные данные
    def enterend(self, list_inp):
        inputs = numpy.array(list_inp, 2).T  # преобразовываем список входных значений в двухмерный массив и транспонируем его( зачем это нужно будет в описании)
# определим входящие сигналы для скрытого слоя
        latent_input = numpy.dot(self.inp_lat, inputs)
# определим исходящие сигналы для скрытого слоя
        latent_output = self.sigmoid(latent_input)
# определим входящие сигналы для выходного слоя
        final_input = numpy.dot(self.lat_out, latent_output)
# определим исходящие сигналы для выходного слоя
        final_output = self.sigmoid(final_input)
        return final_output

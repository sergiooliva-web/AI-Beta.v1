import numpy as np
import pandas as pd
import argparse

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

class OurNeuralNetwork:
    def __init__(self):
        # Веса для первого скрытого слоя (3 нейрона, 2 входа)
        self.w1 = np.random.normal(size=(2, 3))
        self.b1 = np.random.normal(size=(1, 3))
        
        # Веса для второго скрытого слоя (3 нейрона, 3 входа)
        self.w2 = np.random.normal(size=(3, 3))
        self.b2 = np.random.normal(size=(1, 3))
        
        # Веса для третьего скрытого слоя (3 нейрона, 3 входа)
        self.w3 = np.random.normal(size=(3, 3))
        self.b3 = np.random.normal(size=(1, 3))
        
        # Веса для выходного слоя (1 нейрон, 3 входа)
        self.w4 = np.random.normal(size=(3, 1))
        self.b4 = np.random.normal(size=(1, 1))

    def feedforward(self, x):
        # Первый скрытый слой
        h1 = sigmoid(np.dot(x, self.w1) + self.b1)
        
        # Второй скрытый слой
        h2 = sigmoid(np.dot(h1, self.w2) + self.b2)
        
        # Третий скрытый слой
        h3 = sigmoid(np.dot(h2, self.w3) + self.b3)
        
        # Выходной слой
        o1 = sigmoid(np.dot(h3, self.w4) + self.b4)
        return o1

    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 5000

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # Прямое распространение
                h1 = sigmoid(np.dot(x, self.w1) + self.b1)
                h2 = sigmoid(np.dot(h1, self.w2) + self.b2)
                h3 = sigmoid(np.dot(h2, self.w3) + self.b3)
                o1 = sigmoid(np.dot(h3, self.w4) + self.b4)
                y_pred = o1

                # Вычисление градиентов
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Градиенты для выходного слоя
                d_ypred_d_w4 = np.dot(h3.T, d_L_d_ypred * deriv_sigmoid(o1))
                d_ypred_d_b4 = d_L_d_ypred * deriv_sigmoid(o1)

                # Градиенты для третьего скрытого слоя
                d_h3_d_w3 = np.dot(h2.T, (d_L_d_ypred * deriv_sigmoid(o1)).dot(self.w4.T) * deriv_sigmoid(h3))
                d_h3_d_b3 = (d_L_d_ypred * deriv_sigmoid(o1)).dot(self.w4.T) * deriv_sigmoid(h3)

                # Градиенты для второго скрытого слоя
                d_h2_d_w2 = np.dot(h1.T, ((d_L_d_ypred * deriv_sigmoid(o1)).dot(self.w4.T) * deriv_sigmoid(h3)).dot(self.w3.T) * deriv_sigmoid(h2))
                d_h2_d_b2 = ((d_L_d_ypred * deriv_sigmoid(o1)).dot(self.w4.T) * deriv_sigmoid(h3)).dot(self.w3.T) * deriv_sigmoid(h2)

                # Градиенты для первого скрытого слоя
                d_h1_d_w1 = np.dot(x.reshape(-1, 1), (((d_L_d_ypred * deriv_sigmoid(o1)).dot(self.w4.T) * deriv_sigmoid(h3)).dot(self.w3.T) * deriv_sigmoid(h2)).dot(self.w2.T) * deriv_sigmoid(h1))
                d_h1_d_b1 = (((d_L_d_ypred * deriv_sigmoid(o1)).dot(self.w4.T) * deriv_sigmoid(h3)).dot(self.w3.T) * deriv_sigmoid(h2)).dot(self.w2.T) * deriv_sigmoid(h1)

                # Обновление весов и смещений
                self.w4 -= learn_rate * d_ypred_d_w4
                self.b4 -= learn_rate * d_ypred_d_b4.sum(axis=0)

                self.w3 -= learn_rate * d_h3_d_w3
                self.b3 -= learn_rate * d_h3_d_b3.sum(axis=0)

                self.w2 -= learn_rate * d_h2_d_w2
                self.b2 -= learn_rate * d_h2_d_b2.sum(axis=0)

                self.w1 -= learn_rate * d_h1_d_w1
                self.b1 -= learn_rate * d_h1_d_b1.sum(axis=0)

            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))

def load_data(file_path):
    df = pd.read_csv(file_path, comment='#')  # Игнорирование строк с комментариями
    data = df[['weight', 'voice_pitch']].values
    labels = df['label'].astype(float).values  # Преобразование меток в числовой тип
    return data, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network on animal data.")
    parser.add_argument("data_file", type=str, help="Path to the CSV file containing the data.")
    args = parser.parse_args()

    # Загрузка данных из CSV-файла
    data, all_y_trues = load_data(args.data_file)

    # Тренируем нашу нейронную сеть!
    network = OurNeuralNetwork()
    network.train(data, all_y_trues)
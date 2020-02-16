import pandas as pd
import re
# from keras.datasets import mnist
import smote as smote

class InputParser:

    @staticmethod
    def parse_input_zoo_data(filename, header='infer'):
        # X_f,y_f = smote.Data_Extract(filename)
        # classes = y_f.tolist()
        # labels = y_f.tolist()
        # input_database = {
        #     0: X_f[:,:]
        # }

        # (X_train, y_train), (X_test, y_test) = mnist.load_data()
        #
        # d1, d2, d3 = X_train.shape
        # X_train_reshaped = X_train.reshape(d1, d2 * d3)
        # print(X_train_reshaped[:2000, :].shape)
        # y_train_half = y_train[:2000]
        # classes = y_train_half.tolist()
        # labels = y_train_half.tolist()
        # # print(labels)
        #
        # input_database = {
        #     0: X_train_reshaped[:2000, :]
        # }



        # input_data = pd.read_csv(filename, header=header)
        # print(input_data.head())
        # classes = input_data[30].tolist()
        # labels = input_data[30].tolist()
        # # 0: input_data.as_matrix([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        # input_database = {
        #     0: input_data.as_matrix([0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17,18,19,20,21,22,23,24,25,26,27,28,29])
        # }
        #
        #     (X_train, y_train), (X_test, y_test) = mnist.load_data()
        #
        #     d1, d2, d3 = X_train.shape
        #     X_train_reshaped = X_train.reshape(d1, d2 * d3)
        #     print(X_train_reshaped[:2000, :].shape)
        #     y_train_half = y_train[:2000]
        #     classes = y_train_half.tolist()
        #     labels = y_train_half.tolist()
        #     # print(labels)
        #
        #     input_database = {
        #         0: X_train_reshaped[:2000, :]
        #     }

        # (X_train, y_train), (X_test, y_test) = mnist.load_data()
        #
        # d1, d2, d3 = X_train.shape
        # X_train_reshaped = X_train.reshape(d1, d2 * d3)
        # print(X_train_reshaped[:2000, :].shape)
        # y_train_half = y_train[:2000]
        # classes = y_train_half.tolist()
        # labels = y_train_half.tolist()
        # # print(labels)
        #
        # input_database = {
        #     0: X_train_reshaped[:2000, :]
        # }

        # input_data = pd.read_csv(filename, header=header)
        # print(input_data.head())
        # classes = input_data[30].tolist()
        # labels = input_data[30].tolist()
        #
        # input_database = {
        #     0: input_data.as_matrix([0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17,18,19,20,21,22,23,24,25,26,27,28,29])
        # }

        input_data = pd.read_csv(filename, header=header)

        classes = input_data[17].tolist()
        labels = input_data[0].tolist()
        input_database = {
            0: input_data.as_matrix([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        }

        return input_database, labels, classes

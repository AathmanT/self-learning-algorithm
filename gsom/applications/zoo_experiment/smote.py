import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def nearest_neighbour(X, x):
    euclidean = np.ones(X.shape[0] - 1)
    additive = [None] * (1 * X.shape[1])
    additive = np.array(additive).reshape(1, X.shape[1])
    k = 0
    for j in range(0, X.shape[0]):
        if np.array_equal(X[j], x) == False:
            euclidean[k] = np.sqrt(sum((X[j] - x) ** 2))
            k = k + 1
    euclidean = np.sort(euclidean)
    weight = random.random()
    while (weight == 0):
        weight = random.random()
    additive = np.multiply(euclidean[:1], weight)
    return additive


def SMOTE_100(X):
    new = [None] * (X.shape[0] * X.shape[1])
    new = np.array(new).reshape(X.shape[0], X.shape[1])
    k = 0
    for i in range(0, X.shape[0]):
        additive = nearest_neighbour(X, X[i])
        for j in range(0, 1):
            new[k] = X[i] + additive[j]
            k = k + 1
    return new  # the synthetic samples created by SMOTe

def Data_Extract(filename):
    df = pd.read_csv('data/adult.csv')
    # df["income"] = df["income"].replace(">50K", "1")
    # df["income"] = df["income"].replace("<=50K", "0")


    # Loading of Selected Features into X
    X = df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]].values

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)
    X = X[:, 1:]
    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [10])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)
    X = X[:, 1:]
    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [26])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)
    X = X[:, 1:]
    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [32])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)
    X = X[:, 1:]
    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [46])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)
    X = X[:, 1:]
    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [51])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)
    X = X[:, 1:]
    # Loading of the Label into y
    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [-1])], remainder='passthrough')
    y1 = np.array(columnTransformer.fit_transform(df), dtype=np.str)

    y = y1[:, 1]
    y = np.where(y == "0.0", "0", y)
    y = np.where(y == "1.0", "1", y)
    # y = df.iloc[:, [-1]].values

    print(y)
    # feature scaling
    from sklearn.preprocessing import StandardScaler

    sc_X = StandardScaler()
    X = sc_X.fit_transform(X)
    print(X)
    unique, counts = np.unique(y, return_counts=True)
    minority_shape = dict(zip(unique, counts))['1']  # 2. Storing the minority class instances separately
    x1 = np.ones((minority_shape, X.shape[1]))
    k = 0
    for i in range(0, X.shape[0]):
        if y[i] == "1":
            x1[k] = X[i]
            k = k + 1  # 3. Applying 100% SMOTe
    sampled_instances = SMOTE_100(x1[:,:])  # Keeping the artificial instances and original instances together
    X_f = np.concatenate((X, sampled_instances), axis=0)
    y_sampled_instances = np.ones(minority_shape)
    y_f = np.concatenate((y, y_sampled_instances), axis=0)
    # X_f and y_f are final

    return X_f,y_f

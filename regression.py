#!/usr/bin/env python

from __future__ import print_function

from sklearn.svm import SVR
from sklearn.linear_model import ARDRegression
import numpy as np

SEPARATOR = ","


def load_data(f):
    header_line = f.readline().strip()
    names = header_line.split(SEPARATOR)
    data = []
    for l in f:
        line = l.strip()
        data.append([0.0 if v == "" else float(v) for v in line.split(SEPARATOR)])
    return names, np.array(data)


def learn_model(x_mat, y):
    #model = SVR(kernel='rbf')
    model = ARDRegression()
    model.fit(x_mat, y)
    return model


def filter_data(names, data, remove_list):
    indexes = []
    for remove in remove_list:
        idx = names.index(remove)
        indexes.append(idx)
    filtered_names = [n for n in names if n not in remove_list]
    return filtered_names, np.delete(data, indexes, 1)


def select_data(names, data, y_feature):
    y_idx = names.index(y_feature)
    y = data[:, y_idx]
    x_mat = np.delete(data, y_idx, 1)
    return x_mat, y


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data",
                        help="input csv with data to fit")
    parser.add_argument("-t", "--test",
                        help="test data")
    parser.add_argument("y_feature",
                        help="name of feature to predict")
    args = parser.parse_args()

    model = None
    with open(args.data) as f:
        names, data = load_data(f)
        x_mat, y = select_data(names, data, args.y_feature)
        model = learn_model(x_mat, y)

    with open(args.test) as f:
        names, data = load_data(f)
        x_mat, y = select_data(names, data, args.y_feature)
        print(model)
        y_hat = model.predict(x_mat)
        print(np.linalg.norm(y - y_hat))

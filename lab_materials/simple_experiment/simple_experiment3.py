import argparse
import numpy
import sklearn.metrics
import yaml
from yaml.loader import BaseLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


def put_in_bin(values, bins=None):
    if bins is None:
        bins = [(-2, -0.75), (-0.75, -0.25), (-0.25, 0), (0, 0.5), (0.5, 2.0)]
    out = numpy.zeros(len(bins), dtype=int)
    for v in values:
        for i, bin_idx in enumerate(bins):
            if bin_idx[0] < v <= bin_idx[1]:
                out[i] += 1
    return out


def load_data(path, subset="train"):
    path = Path(path)
    with open(path / '{}/X_{}.txt'.format(subset, subset), 'rt') as f:
        lines = [line.strip() for line in f.readlines()]

    data = [[float(v) for v in line.split(' ') if v]
            for line in lines]

    data = [put_in_bin(l) for l in data]

    with open(path / "{}/y_{}.txt".format(subset, subset)) as f:
        y = [int(l.strip()) for l in f.readlines()]

    return data, y


def train_model(data, target, model_type='logistic'):
    if model_type == 'logistic':
        model = LogisticRegression()
    else:
        model = KNeighborsClassifier()
    model.fit(data, target)
    return model


def evaluate_model(model, data, target):
    predicted = model.predict(data)
    return sklearn.metrics.accuracy_score(target, predicted)


def main(model_type):
    train_data, train_target = load_data("./UCI HAR Dataset")
    test_data, test_target = load_data("./UCI HAR Dataset", subset="test")

    train_data, valid_data, train_target, valid_target = train_test_split(
        train_data, train_target, test_size=0.2, random_state=0)

    model = train_model(train_data, train_target, model_type=model_type)
    print(evaluate_model(model, test_data, test_target))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()

    with open(args.config, 'rt') as f:
        params = yaml.load(f, Loader=BaseLoader)
    return params


if __name__ == "__main__":
    main(parse_args()['model_type'])

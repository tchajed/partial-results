#!/usr/bin/env python
# vim: et:tw=79:ts=4:sw=4

from __future__ import print_function, division

import yaml
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import ARDRegression

import subprocess
import os
from os.path import join, exists


class GraphFile(object):
    def __init__(self, prefix, graph_format="snap"):
        self.prefix = prefix
        self.graph_format = graph_format
    def args(self):
        """ Format this graph as a pair of arguments. """
        return ["--graph", self.prefix, "--format", self.graph_format]
    def __repr__(self):
        return "%s [%s]" % (self.prefix, self.graph_format)


class AbstractCli(object):
    program = None
    is_mpi = False
    def run_args(self):
        args = []
        if self.is_mpi:
            args.append("mpiexec")
        if "MPI_MACHINEFILE" in os.environ:
            hostfile = os.environ["MPI_MACHINEFILE"]
            args.extend(["--hostfile", hostfile])
        args.append(self.program)
        args.extend(self.args())
        return args
    def run(self):
        args = [str(arg) for arg in self.run_args()]
        print("running: ", " ".join(args))
        subprocess.call(args)


class SampleCli(AbstractCli):
    program = "sample"
    is_mpi = True

    def __init__(self, algorithm, graph, output):
        self.algorithm = algorithm
        self.graph = graph
        self.output = output
        if output.graph_format not in  ["tsv", "snap"]:
            raise ValueError("sample cli always outputs in tsv/snap format")
    def args(self):
        args = self.graph.args()
        if self.algorithm["name"] == "forest_fire":
            params = self.algorithm["parameters"]
            args.extend(["--prob", params["pf"]])
        else:
            raise ValueError("unknown algorithm " + self.algorithm["name"])
        args.extend(["--output", self.output.prefix])
        return args


class PartialResultsCli(AbstractCli):
    program = "partial_results"
    is_mpi = True

    def __init__(self, params, graph, prefix):
        self.params = params
        self.graph = graph
        self.prefix = prefix
    def args(self):
        args = self.graph.args()
        if self.params["engine"] not in ["synchronous"]:
            raise ValueError("unknown engine " + self.params["engine"])
        args.extend(["--engine", self.params["engine"]])
        args.extend(["--tol", self.params["tolerance"]])
        args.extend(["--feature_period", self.params["feature_period"]])
        args.extend(["--max_vertices", self.params["accuracy_k"]])
        args.extend(["--output", self.prefix])
        return args


class Features(object):
    def __init__(self, names, data):
        self.names = list(names)
        self.data = np.array(data)
    @classmethod
    def from_file(cls, fname):
        with open(fname, "r") as f:
            header_line = f.readline().strip()
# try to identify separator, falling back to comma
            sep = ","
            for try_sep in ["|", "\t"]:
                if header_line.find(try_sep) != -1:
                    sep = try_sep
                    break
            names = header_line.split(sep)
            data = []
            for l in f:
                line = l.strip()
                data_row = [0.0 if v == "" else float(v) for v in
                        line.split(sep)]
                data.append(data_row)
            return Features(names, data)
    def to_mat(self):
        return self.data
    def to_col(self):
        assert self.data.shape[1] == 1
        return self.data[:,0]
    def to_row(self):
        assert data.shape[0] == 1
        return data[0,:]
    def __getitem__(self, features):
        indexes = []
        for feat in features:
            idx = self.names.index(feat)
            if idx == -1:
                raise ValueError("unknown feature " + feat)
            indexes.append(idx)
        return Features(features, self.data[:, indexes])
    def __repr__(self):
        return "".join(["\t".join(self.names), "\n", repr(self.data)])


class Evaluation(object):
    def __init__(self, params):
        self.algorithm = params["learn"]["algorithm"]
        self.features = params["features"]
        self.target_feature = params["target_feature"]
        name = self.algorithm["name"]
        self.model = None
        if name == "SVR":
            model = SVR(**self.algorithm["parameters"])
        elif name == "ARDRegression":
            model = ARDRegression(**self.algorithm["parameters"])
        else:
            raise ValueError("unknown sklearn regression algorithm " + name)
        self.model = model
    def train(self, train_fname):
        features = Features.from_file(train_fname)
        x_mat = features[self.features].to_mat()
        y = features[(self.target_feature,)].to_col()
        self.model.fit(x_mat, y)
    def test_error(self, test_fname):
        features = Features.from_file(test_fname)
        x_mat = features[self.features].to_mat()
        y = features[self.target_feature,].to_col()
        y_hat = self.model.predict(x_mat)
        return np.linalg.norm(y - y_hat)


class Experiment(object):
    def __init__(self, o):
        """ Load an experiment from a dictionary of values.

        The format of this object should match that of the YAML configuration.

        """
        self.sample = o["sample"]
        self.run = o["run"]
        self.evaluate = o["evaluate"]
    @classmethod
    def from_yaml(cls, fname):
        """ Load an experiment from a YAML file. """
        with open(fname, "r") as f:
            o = yaml.load(f)
            return Experiment(o)
    def run_trial(self, graph, name, output_dir):
        if not exists(output_dir):
            os.makedirs(output_dir)
        sampled = GraphFile(join(output_dir, name + "-sampled"))
        sample_cli = SampleCli(self.sample["algorithm"], graph, sampled)
        sample_cli.run()

        sampled_dir = join(output_dir, name + "-sample-output")
        partial_results_cli = PartialResultsCli(self.run, sampled,
                sampled_dir)
# run twice, first just to generate pageranks
        partial_results_cli.run()
        partial_results_cli.run()

        true_dir = join(output_dir, name)
        true_partial_results_cli = PartialResultsCli(self.run, graph, true_dir)
        true_partial_results_cli.run()
        true_partial_results_cli.run()

        evaluation = Evaluation(self.evaluate)
        evaluation.train(join(sampled_dir, "features.csv"))
        error = evaluation.test_error(join(true_dir, "features.csv"))
        return error


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--graph",
            help="input graph prefix")
    parser.add_argument("--format",
            default="snap",
            help="input graph format")
    parser.add_argument("--experiment",
            help="experiment YAML configuration file")
    parser.add_argument("-n", "--name",
            help="experiment name")
    parser.add_argument("--output",
            help="output directory")
    args = parser.parse_args()

    experiment = Experiment.from_yaml(args.experiment)
    graph = GraphFile(args.graph, args.format)
    error = experiment.run_trial(graph, args.name, args.output)
    print(error)

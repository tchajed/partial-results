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
    """ Represent a graph on disk.

    Includes a:
    - location (specified by file prefix)
    - format (any of GraphLab's supported formats, such as snap or bintsv4)
    - an optional true pagerank prefix, specifying where to find precomputed
      final pageranks for this graph
    - an optional feature prefix, specifing where to find fully precomputed
      features for this graph

    """
    def __init__(self, prefix, graph_format="snap", pagerank_prefix=None,
            feature_prefix=None):
        self.prefix = prefix
        self.graph_format = graph_format
        self.pagerank_prefix = pagerank_prefix
        self.feature_prefix = feature_prefix
    def args(self):
        """ Format this graph as a pair of arguments. """
        args = ["--graph", self.prefix,
                "--format", self.graph_format]
        return args
    def __repr__(self):
        s = "%s [%s]" % (self.prefix, self.graph_format)
        if self.pagerank_prefix is not None:
            s += " [pagerank]"
        if self.feature_prefix is not None:
            s += " [features]"
        return s


class GraphLibrary(object):
    """ A collection of available graphs, indexed by name. """
    def __init__(self, graphs):
        self.graphs = graphs
    @classmethod
    def from_yaml(cls, fname):
        def load_graph(prefix, d):
            graph_prefix = join(prefix, d["prefix"])
            pagerank_prefix = None
            if d.has_key("pagerank_prefix"):
                pagerank_prefix = join(prefix, d.get("pagerank_prefix"))
            feature_prefix = None
            if d.has_key("feature_prefix"):
                feature_prefix = join(prefix, d.get("feature_prefix"))
            graph = GraphFile(graph_prefix,
                    graph_format=d.get("format", "snap"),
                    pagerank_prefix=pagerank_prefix,
                    feature_prefix=feature_prefix)
            name = d["name"]
            return (name, graph)
        with open(fname, "r") as f:
            o = yaml.load(f)
            prefix = o.get("prefix", "")
            graphs = o["graphs"]
            graph_tuples = [load_graph(prefix, graph) for graph in graphs]
            return GraphLibrary(dict(graph_tuples))
    def __getitem__(self, name):
        return self.graphs[name]
    def get(self, name):
        return self.graphs.get(name)


class AbstractCli(object):
    program = None
    is_mpi = False
    def run_args(self):
        args = []
        if self.is_mpi:
            args.extend(["mpiexec", "-mca", "btl", "^openib"])
        if "MPI_MACHINEFILE" in os.environ:
            hostfile = os.environ["MPI_MACHINEFILE"]
            args.extend(["--hostfile", hostfile])
        args.append(self.program)
        args.extend(self.args())
        return args
    def run(self):
        args = [str(arg) for arg in self.run_args()]
        print("running: ", " ".join(args))
        #raw_input('press enter to run...')
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
        if self.graph.pagerank_prefix is not None:
            args.extend(["--pagerank_prefix", self.graph.pagerank_prefix])
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

        if graph.feature_prefix is None:
            true_dir = join(output_dir, name)
            true_partial_results_cli = PartialResultsCli(self.run, graph, true_dir)
            true_partial_results_cli.run()
            if graph.pagerank_prefix is None:
                true_partial_results_cli.run()
        else:
            true_dir = graph.feature_prefix

        evaluation = Evaluation(self.evaluate)
        evaluation.train(join(sampled_dir, "features.csv"))
        error = evaluation.test_error(join(true_dir, "features.csv"))
        return error


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--graph",
            help="input graph name")
    parser.add_argument("--graph_library",
            default="graphs.yaml",
            help="graph database filename")
    parser.add_argument("--experiment",
            help="experiment YAML configuration file")
    parser.add_argument("-n", "--name",
            help="experiment name")
    parser.add_argument("--output",
            help="output directory")
    args = parser.parse_args()

    graph_lib = GraphLibrary.from_yaml(args.graph_library)
    graph = graph_lib[args.graph]

    experiment = Experiment.from_yaml(args.experiment)
    error = experiment.run_trial(graph, args.name, args.output)
    print(error)

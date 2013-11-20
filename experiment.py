# vim: et:tw=79:ts=4:sw=4
import subprocess
import yaml
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import ARDRegression

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


class GraphFile(object):
    def __init__(self, prefix, graph_format="snap"):
        self.prefix = prefix
        self.graph_format = graph_format
    def args(self):
        """ Format this graph as a pair of arguments. """
        return ["--graph", self.prefix, "--format", self.graph_format]


class AbstractCli(object):
    program = None
    is_mpi = False
    def run_args(self):
        args = []
        if self.is_mpi:
            args.append("mpiexec")
        args.append(self.program)
        args.extend(self.args())
        return args
    def run(self):
        subprocess.call([str(arg) for arg in self.run_args()])


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
    def __getitem__(self, features):
        indexes = []
        for feat in features:
            idx = self.names.index(feat)
            if idx == -1:
                raise ValueError("unknown feature " + feat)
            indexes.append(idx)
        return Features(features, self.data[:, indexes])


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
        raise ValueError("unknown sklearn regression algorithm " + name)
        self.model = model
    def train(self, train_fname):
        features = Features.from_file(train_fname)
        x_mat, y = features[self.features], features[target_feature]
        self.model.fit(x_mat, y)
    def test_error(self, test_fname):
        features = Features.from_file(test_fname)
        x_mat, y = features[self.features], features[target_feature]
        y_hat = self.model.predict(x_mat)
        return np.linalg.norm(y - y_hat)

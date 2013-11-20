# vim: et:tw=79:ts=4:sw=4
import subprocess
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


class GraphFile(object):
    def __init__(self, prefix, graph_format="snap"):
        self.prefix = prefix
        self.graph_format = graph_format
    def args(self):
        """ Format this graph as a pair of arguments. """
        return ["--graph", self.prefix, "--format", self.graph_format]


class AbstractCli(object):
    program = None
    def run(self):
        args = [program]
        args.extend(self.args())
        subprocess.call([str(arg) for arg in args])


class SampleCli(AbstractCli):
    program = "sample"

    def __init__(self, algorithm, graph):
        self.algorithm = algorithm
        self.graph - graph
    def args(self):
        args = self.graph.args()
        if self.algorithm["name"] == "forest_fire":
            params = self.algorithm["parameters"]
            args.extend(["--prob", params["pf"]])
        else:
            raise ValueError("unknown algorithm " + self.algorithm["name"])
        return args


class PartialResultsCli(AbstractCli):
    program = "partial_results"

    def __init__(self, params, graph):
        self.params = params
        self.graph = graph
    def args(self):
        args = self.graph.args()
        if self.params["engine"] not in ["synchronous"]:
            raise ValueError("unknown engine " + self.params["engine"])
        args.extend(["--engine", self.params["engine"]])
        args.extend(["--tol", self.params["tolerance"]])
        args.extend(["--feature_period", self.params["feature_period"]])
        args.extend(["--max_vertices", self.params["accuracy_k"]])


class Evaluate(object):
    def __init__(self, params):
        self.algorithm = params["learn"]["algorithm"]
        self.features = params["features"]
        self.target_feature = params["target_feature"]
    def base_model(self):
        name = self.algorithm["name"]
        if name == "SVR":
            return SVR(**self.algorithm["parameters"])
        elif name == "ARDRegression":
            return ARDRegression(**self.algorithm["parameters"])
        raise ValueError("unknown sklearn regression algorithm " + name)


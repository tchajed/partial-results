#!/usr/bin/env python

from itertools import islice

def load_features(f):
  header_line = f.readline().strip()
  feature_names = header_line.split(",")
  features_matrix = []
  for line in f:
    features = [float(s) for s in line.strip().split(",")]
    features_matrix.append(features)
  return (feature_names, features_matrix)

def select_inputs(feature_names, features, skip):
  inputs = []
  for name, v in zip(feature_names, features):
    if name not in skip:
      inputs.append(v)
  return inputs

def window_features(feature_names, feats, skip, window_size):
  feature_matrix = []
  past_inputs = [[None] * (len(feature_names) - len(skip))] * (window_size - 1)
  for features in feats:
    new_features = [f for f in features]
    for past_input in reversed(past_inputs):
      new_features.extend(past_input)
    feature_matrix.append(new_features)
    inputs = select_inputs(feature_names, features, skip)
    past_inputs.append(inputs)
    past_inputs = past_inputs[1:]
  new_feature_names = [n for n in feature_names]
  input_names = [n for n in feature_names if n not in skip]
  for i in xrange(1,window_size):
    for n in input_names:
      new_feature_names.append("%s-%d" % (n, i))
  return (new_feature_names, feature_matrix)

def write_features(feature_names, feature_matrix, f):
  f.write(",".join(feature_names))
  f.write("\n")
  for features in feature_matrix:
    str_features = ["" if v is None else str(v) for v in features]
    f.write(",".join(str_features))
    f.write("\n")
    
if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("filename",
      help="feature file to load")
  parser.add_argument("-n", "--window",
      type=int,
      default=3,
      help="window size")
  parser.add_argument("-o", "--output",
      help="output filename")
  args = parser.parse_args()

  with open(args.filename) as f:
    feature_names, feats = load_features(f)
    skip = set(["iter", "rank_e", "rmse"])
    windowed_feature_names, feature_matrix = \
        window_features(feature_names, feats, skip, args.window)
    with open(args.output, "w") as output_f:
      write_features(windowed_feature_names, feature_matrix, output_f)

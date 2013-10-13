#!/usr/bin/env python

from __future__ import print_function, division

import os
import gzip
from os.path import join, split, expanduser
import math
import numpy as np
import sys
from random import random
import itertools
import multiprocessing

class VertexFeatures(object):
  def __init__(self, line):
    parts = line.split()
    parts.pop(0) # vid
    self.in_edges = int(parts[0])
    self.out_edges = int(parts[1])
    pageranks = []
    while parts:
      active = parts.pop(0) == "1"
      pr = float(parts.pop(0))
      pageranks.append( (active, pr) )
    self._pageranks = pageranks
  def pageranks(self):
    return self._pageranks
  def final_pagerank(self):
    return self._pageranks[-1][1]
  def pagerank_list(self):
    return [active_pr[1] for active_pr in self._pageranks]
  def convergence_time(self):
    return max([ i for i,active_pr
      in enumerate(self._pageranks)
      if active_pr[0]])

  def __repr__(self):
    return "VertexFeatures(out_edges=%d, pageranks=%s)" \
      % (self.out_edges, self.pageranks())

def create_image(vertices):
  width = max([len(v.pageranks()) for v in vertices])
  height = len(vertices)
  return (width, height, create_image_data(vertices, width))

def create_image_data(vertices, max_length):
  # max_pagerank = max([max(v.pagerank_list()) for v in vertices])
  for v in vertices:
    row = []
    max_pagerank = max(v.pagerank_list()[1:])
    for active_pr in v.pageranks()[1:]:
      scaled_pr = math.pow(active_pr[1] / max_pagerank, 1/4)
      val = int(50 + 200 * scaled_pr)
      if active_pr[0]:
        color = (0, 0, val)
      else:
        color = (val, 0, 0)
      row.append(color)
    # pad each row to the same length with black
    while len(row) < max_length:
      row.append( (0,0,0) )
    yield row

def write_graph(data, f):
  data = np.array([x for x in data])
  H, xedges, yedges = np.histogram2d(data[:,0], data[:,1], 30)
  for row,ylabel in enumerate(yedges[:-1]):
    for col,xlabel in enumerate(xedges[:-1]):
      freq = H[col, row]
      f.write("%f %f %d\n" % (xlabel, ylabel, freq))
    f.write("\n")

def scatter_plot(vertices, x_fn, y_fn):
  for v in vertices:
    yield (x_fn(v), y_fn(v))

def convergence_graph(vertices, feature_fn):
  """
  Create data for a graph of convergence time.

  feature_fn should be a function that takes a vertex and returns a double to
  be used as the x-axis data.

  """
  return scatter_plot(vertices, feature_fn, lambda v: v.convergence_time())

def write_ppm(f, image):
  width, height, image_data = image
  print( (width, height) )
  f.write("P6\n")
  f.write("%d %d\n" % (width, height))
  f.write("255\n")
  for row in image_data:
    for color in row:
      f.write(bytes("%c%c%c" % color))

def sample_vertices(vertices, p):
  convergence_avg = 0.0
  pr_avg = 0.0
  for v in vertices:
    pr_avg += v.final_pagerank()
  pr_avg /= len(vertices)
  return [v for v in vertices
      if v.final_pagerank() > pr_avg or
      v.final_pagerank() < pr_avg and random() < p
      ]

def prefix_files(prefix):
  prefix_path, prefix_fname = split(expanduser(prefix))
  if not prefix_path:
    prefix_path = "."
  for fname in os.listdir(prefix_path):
    if fname.startswith(prefix_fname):
      yield join(prefix_path, fname)
      
def get_vertices(fname, q):
  open_function = open
  if fname.endswith(".gz"):
    open_function = gzip.open
  batch = []
  with open_function(fname, "r") as f:
    for line in f:
      batch.append(VertexFeatures(line))
      if len(batch) >= 100:
        q.put(batch)
        batch = []
  q.put(batch)
  q.put(None)

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("-o", "--output",
      help="prefix for output files",
      default="vertices")
  parser.add_argument("-p", "--prob",
      help="sampling percentage for low values",
      type=float,
      default=1.0)
  parser.add_argument("prefix",
      help="prefix of file with vertex history")
  parser.add_argument("-i", "--image",
      action="store_true",
      help="output an image")
  parser.add_argument("-g", "--graph",
      action="store_true",
      help="output scatterplot data")
  args = parser.parse_args()

  print("loading", end="")
  sys.stdout.flush()
  pool = multiprocessing.Pool(processes=4)
  files = list(prefix_files(args.prefix))
  vertices = []
  q = multiprocessing.Queue()
  for fname in files:
    p = multiprocessing.Process(target=get_vertices, args=(fname,q))
    p.start()
  num_finished = 0
  while num_finished < len(files):
    v = q.get()
    if v is not None:
      vertices.extend(v)
    else:
      num_finished += 1
      print(".", end="")
      sys.stdout.flush()
  print("\rloaded vertices")

  vertices = sample_vertices(vertices, args.prob)
  vertices.sort(key = lambda v: v.final_pagerank())
  print("sorted vertices")

  if args.image:
    image = create_image(vertices)
    with open(args.output + ".ppm", "w") as f:
      write_ppm(f, image)
    print("wrote image")

  if args.graph:
    with open(args.output + "-pagerank.data", "w") as f:
      data = convergence_graph(vertices,
        lambda v: v.final_pagerank())
      write_graph(data, f)
    print("wrote pagerank graph")

    with open(args.output + "-degree.data", "w") as f:
      data = convergence_graph(vertices,
        lambda v: v.out_edges)
      write_graph(data, f)
    print("wrote degree graph")

    with open(args.output + "-pagerank-vs-degree.data", "w") as f:
      write_graph(scatter_plot(vertices,
        lambda v: v.out_edges,
        lambda v: v.final_pagerank()), f)
    print("wrote pagerank vs degree graph")

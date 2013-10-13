#!/bin/sh

PREFIX="$1"
TITLE="$2"

./histogram.plt "$PREFIX-pagerank" "$TITLE" \
  "pagerank" "convergence iterations" "freq"
./histogram.plt "$PREFIX-degree" "$TITLE" \
  "outdegree" "convergence iterations" "freq"
./histogram.plt "$PREFIX-pagerank-vs-degree" "$TITLE" \
  "degree" "pagerank" "freq"


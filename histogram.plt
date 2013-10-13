#!/bin/sh

usage() {
  echo "$0 <prefix> <title> <xlabel> <ylabel> <zlabel>"
}

PREFIX="$1"
TITLE="$2"
XLABEL="$3"
YLABEL="$4"
ZLABEL="$5"

gnuplot << EOF
# adapted from Brighten Godfrey's example at
# http://youinfinitesnake.blogspot.com/2011/02/attractive-scientific-plots-with.html
set terminal svg size 640,480 #fname "Gill Sans" fsize 9 rounded dashed
set output "$PREFIX.svg"

# Line style for axes
set style line 80 lt 0
set style line 80 lt rgb "#808080"

# Line style for grid
set style line 81 lt 3  # dashed
set style line 81 lt rgb "#808080" lw 0.5  # grey

set grid back linestyle 81
set border 3 back linestyle 80 # Remove border on top and right.  These
             # borders are useless and make it harder
             # to see plotted lines near the border.
    # Also, put it in grey; no need for so much emphasis on a border.
set xtics nomirror
set ytics nomirror

set log cb
set mcbtics 10    # Makes logscale look good.

unset key

set xlabel "$XLABEL"
set ylabel "$YLABEL"
set cblabel "$ZLABEL"
set title "$TITLE"

set palette rgbformula -23,-28,-3

plot '$PREFIX.data' using 1:2:(\$3+0.01) with image


EOF


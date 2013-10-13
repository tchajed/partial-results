default: as-plots wiki-plots png

as: as-data as-plots as.png

wiki: wiki-data wiki-plots

as-data: history_viz.py
	time ./history_viz.py --output as --graph --image as-caida-history

as-plots:
	./plot-viz.sh "as" "AS-CAIDA"

wiki-data: history_viz.py
	time ./history_viz.py --output wiki --prob 0.001 --graph --image wiki-talk-history

wiki-plots:
	./plot-viz.sh "wiki" "Wiki Talk"

as-vertices.ppm: history_viz.py

%.png: %.ppm
	convert $< $@

PPM := $(wildcard *.ppm)
PNG := $(PPM:.ppm=.png)

png: $(PNG)

TERMINAL="pdfcairo color transparent size 8cm,7cm enhanced font \"Times, 12\""
#TERMINAL='cairolatex pdf input dashed transparent colortext size 8cm,7cm'
EXTENSION='pdf'
#EXTENSION='pdf'
gnuplot <<EOF
set terminal ${TERMINAL}
set output './kernel_approx.${EXTENSION}'
set size ratio 0.7
set key top right samplen 2 spacing 0.9 
#set logscale x
set title 'Kernel approximation ' font ",18" 
set xlabel 'D / d' font ",18" 
set ylabel 'MSE' font ",18"  offset 1, 0
#set yrange [0.0:0.007]
set style fill transparent solid 0.2 noborder
plot 'rff.dat' using 1:(column(2)-column(3)):(column(2)+column(3)) with filledcurves lc rgb "red" title '', \
     '' using 1:2 with lp lt 1 pt 7 ps .5 lc rgb "red" lw 3 title 'RFF', 'fastfood.dat' using 1:(column(2)-column(3)):(column(2)+column(3)) with filledcurves lc rgb "green" title '', \
     '' using 1:2 with lp lt 1 pt 7 ps .5 lc rgb "green" lw 3 title 'Fastfood', 'orff.dat' using 1:(column(2)-column(3)):(column(2)+column(3)) with filledcurves lc rgb "purple" title '', \
     '' using 1:2 with lp lt 1 pt 7 ps .5 lc rgb "purple" lw 3 title 'ORFF','sorff.dat' using 1:(column(2)-column(3)):(column(2)+column(3)) with filledcurves lc rgb "black" title '', \
     '' using 1:2 with lp lt 1 pt 7 ps .5 lc rgb "black" lw 3 title 'SORFF','krff.dat' using 1:(column(2)-column(3)):(column(2)+column(3)) with filledcurves lc rgb "blue" title '', \
     '' using 1:2 with lp lt 1 pt 7 ps .5 lc rgb "blue" lw 3 title 'KRFF'

EOF

set terminal pdfcairo enhanced color font "Helvetica,12"
set output "my_plot.pdf"
set style data dots
set nokey
set xrange [0: 1.82036]
set yrange [-24.77179 : -8.40635]
set arrow from  0.84488, -24.77179 to  0.84488,  -8.40635 nohead
set xtics ("M"  0.00000,"G"  0.84488,"K"  1.82036)
 plot "bismuth_final_band.dat"
unset output

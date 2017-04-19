#! /bin/bash
# constant
PlotInputLoc=/export/home/etikhonc/net/hci-storage01/userfolders/etikhonc/scratch/classification/MNIST_5000/
PlotOutputLoc=/export/home/etikhonc/workspace/classification/MNIST_5000/
PlotOutputName="train_performance"
InputFileExt=txt
OutputFileExt=png
delim=()
# read argument list
ARGS=("$@")
N=${#ARGS[@]}
for (( i=0; i<$N; i++ ))
do
	MODEL[i]="${ARGS[$i]}"
	nIt[i]=$(wc -l < "${PlotInputLoc}/${MODEL[i]}/train_loss.$InputFileExt")
	echo " - plot ${MODEL[i]}, ${nIt[i]} iterations"
	PlotOutputName="${PlotOutputName}_${MODEL[i]}"
done
echo "TOTAL $N files to process..."
echo "SAVE TO: ${PlotOutputLoc}/${PlotOutputName}"

gnuplot <<- EOF
MODELSTR="${MODEL[*]}"
set term $OutputFileExt large size 1200,1024
set output "$PlotOutputLoc/$PlotOutputName.$OutputFileExt"

set multiplot layout 3, 1 title "Training performance" font ",14"
set tmargin 3
#
set key outside box
set xlabel "Iteration"
set ylabel "Train Loss" font ",16"
# set style line 1 lt rgb "red" lw 3
# set style line 2 lt rgb "green" lw 3
set grid x2tics

plot for [model in MODELSTR] sprintf("$PlotInputLoc/%s/train_loss.$InputFileExt", model) using 1:2 every 50 with linespoints lw 2 title sprintf("%s", model)
#,\
 	 # for [model in MODELSTR] sprintf("$PlotInputLoc/%s/test_loss.$InputFileExt", model) using 1:2 every 10 with linespoints lw 2 title sprintf("%s-test", model)
#
set key outside box
set xlabel "Iteration"
set ylabel "Test Loss" font ",16"
# set style line 1 lt rgb "red" lw 3
# set style line 2 lt rgb "green" lw 3
set grid x2tics

plot for [model in MODELSTR] sprintf("$PlotInputLoc/%s/test_loss.$InputFileExt", model) using 1:2 every 100 with lines lw 2 title sprintf("%st", model)
#
set key outside box
set xlabel "Iteration"
set ylabel "Test accuracy" font ",16"
set style line 1 lt rgb "red" lw 3
set style line 2 lt rgb "green" lw 3
set grid x2tics
plot for [model in MODELSTR] sprintf("$PlotInputLoc/%s/test_acc.$InputFileExt", model) using 1:2 every 100 with lines lw 2 title sprintf("%s", model)
#
exit()

EOF

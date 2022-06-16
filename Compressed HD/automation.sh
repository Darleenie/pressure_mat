#!/bin/bash

for i in 1 2
do
    if [ "$i" == "1" ]
    then
        directory="daily_activity"
        dataset="daily_activity"
    else
        directory="HCM"
        dataset="valve"
    fi
    for j in 2 4 10 20 25 40 50 100
    do
        output="./results/plot$j$dataset.txt"
        echo "printing dots for $j $dataset dataset"
        python main.py $directory $dataset 1000 5 100 0 1 $j > $output
    done
done

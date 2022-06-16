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
    for j in 1000
    do
        output="./results/$j$dataset.txt"
        echo "testing dimRed for $j $dataset dataset"
        python main.py $directory $dataset $j 5 100 0 0 0 > $output
    done
done

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
    for j in 2 4 5 8 10 20 25 40 50 100 125 200 250 500
    do
        output="./results/compressed$j$dataset.txt"
        echo "testing compression for $j $dataset dataset"
        python main.py $directory $dataset 1000 5 100 0 1 $j > $output
    done
done

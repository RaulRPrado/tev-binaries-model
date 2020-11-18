#!/bin/bash

for size in '_small'
do
    for orb in 'an'
    do
        label="${orb}${size}_dref"
        echo "Submitting label $label"
        qsub -cwd -l s_cpu=48:00:00 -js 10 -N $label fit_psr.sh $label

        # for par in 'm' 'i' 'e' 'o' 'p'
        # do
        #     for dir in 'sup' 'inf'
        #     do
        #         label="${orb}_${par}_${dir}${size}_test"
        #         echo $label
        #         qsub -cwd -l s_cpu=48:00:00 -N $label fit_psr.sh $label
        #     done # dir
        # done # par
    done # orb
done # size

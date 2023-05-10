#!/bin/bash

# set the input parameters for each run
# 1.experimentName 2.weight1 3.weight2 4.mutationRate 5.crossoverRate

prrams=(
	"experiment_name="exper_hyperTune1" weight1=0.1 weight2=0.9 mutationRate=0.05 crossoverRate=0.6"
	"experiment_name="exper_hyperTune2" weight1=0.5 weight2=0.5 mutationRate=0.05 crossoverRate=0.6"
	"experiment_name="exper_hyperTune3" weight1=0.9 weight2=0.1 mutationRate=0.05 crossoverRate=0.6"
	"experiment_name="exper_hyperTune4" weight1=0.5 weight2=0.5 mutationRate=0.10 crossoverRate=0.6"
	"experiment_name="exper_hyperTune5" weight1=0.5 weight2=0.5 mutationRate=0.15 crossoverRate=0.6"
	"experiment_name="exper_hyperTune6" weight1=0.5 weight2=0.5 mutationRate=0.05 crossoverRate=0.4"
	"experiment_name="exper_hyperTune1" weight1=0.1 weight2=0.9 mutationRate=0.05 crossoverRate=0.8"
)

# iterate over the input parameters and run mainG.py
for i in "${params[@]}"
do
	python mainG.py $i
done


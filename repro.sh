#!/bin/bash

dataset=bc
startclean=true
nsample=50000
epochs=30
mcsamples=100
nreps=4

echo Starting experiment...

echo $vaeoutput

python src/create_missing.py --dataset $dataset --pmiss 0.2 --normalize01 0 -o missing.csv --oref ref.csv -n $nsample --istarget 1
python src/create_dataset.py -i missing.csv -o interim  --ref ref.csv --target target --dataset $dataset

vaeoutput=gain_samples.p
counter=1

echo Testing GAIN

python src/gain_mod.py -i missing.csv -o imputed.csv --it $epochs --dataset $dataset --samples 40
python src/analysis.py --dataset $dataset --new $startclean --serialized $vaeoutput --model GAIN
while [ $counter -le $nreps ]
do
	echo $counter
	python src/gain_mod.py -i missing.csv -o imputed.csv --it $epochs --dataset $dataset --samples 40
	python src/analysis.py --dataset $dataset --serialized $vaeoutput --model GAIN
	((counter++))
done
python src/results.py --filename analysis_scores_$dataset.p --dataset $dataset --model GAIN

prior=mean
vaeoutput=${dataset}_${prior}_testX_imputed_multiple.p
counter=1

echo Testing VAE MEAN

python src/vae_trainer.py -o $vaeoutput --it $epochs --dataset $dataset --prior $prior --mcsamples $mcsamples
python src/analysis.py --dataset $dataset --new $startclean --serialized $vaeoutput 
while [ $counter -le $nreps ]
do
	echo $counter
	python src/vae_trainer.py -o $vaeoutput --it $epochs --dataset $dataset --prior $prior --mcsamples $mcsamples
	python src/analysis.py --dataset $dataset --serialized $vaeoutput 
	((counter++))
done
python src/results.py --filename analysis_scores_$dataset.p --dataset $dataset --prior $prior --model VAE


prior=zero
vaeoutput=${dataset}_${prior}_testX_imputed_multiple.p
counter=1

echo Testing VAE ZERO

python src/vae_trainer.py -o $vaeoutput --it $epochs --dataset $dataset --prior $prior --mcsamples $mcsamples
python src/analysis.py --dataset $dataset --new $startclean --serialized $vaeoutput 
while [ $counter -le $nreps ]
do
	echo $counter
	python src/vae_trainer.py -o $vaeoutput --it $epochs --dataset $dataset --prior $prior --mcsamples $mcsamples
	python src/analysis.py --dataset $dataset --serialized $vaeoutput 
	((counter++))
done
python src/results.py --filename analysis_scores_$dataset.p --dataset $dataset --prior $prior --model VAE



prior=std
vaeoutput=${dataset}_${prior}_testX_imputed_multiple.p
counter=1

echo Testing VAE STD

python src/vae_trainer.py -o $vaeoutput --it $epochs --dataset $dataset --prior $prior --mcsamples $mcsamples
python src/analysis.py --dataset $dataset --new $startclean --serialized $vaeoutput 
while [ $counter -le $nreps ]
do
	echo $counter
	python src/vae_trainer.py -o $vaeoutput --it $epochs --dataset $dataset --prior $prior --mcsamples $mcsamples
	python src/analysis.py --dataset $dataset --serialized $vaeoutput 
	((counter++))
done
python src/results.py --filename analysis_scores_$dataset.p --dataset $dataset --prior $prior --model VAE


prior=epsilon
vaeoutput=${dataset}_${prior}_testX_imputed_multiple.p
counter=1

echo Testing VAE EPSILON

python src/vae_trainer.py -o $vaeoutput --it $epochs --dataset $dataset --prior $prior --mcsamples $mcsamples
python src/analysis.py --dataset $dataset --new $startclean --serialized $vaeoutput 
while [ $counter -le $nreps ]
do
	echo $counter
	python src/vae_trainer.py -o $vaeoutput --it $epochs --dataset $dataset --prior $prior --mcsamples $mcsamples
	python src/analysis.py --dataset $dataset --serialized $vaeoutput 
	((counter++))
done
python src/results.py --filename analysis_scores_$dataset.p --dataset $dataset --prior $prior --model VAE



prior=uniform
vaeoutput=${dataset}_${prior}_testX_imputed_multiple.p
counter=1

echo Testing VAE UNIFORM

python src/vae_trainer.py -o $vaeoutput --it $epochs --dataset $dataset --prior $prior --mcsamples $mcsamples
python src/analysis.py --dataset $dataset --new $startclean --serialized $vaeoutput 
while [ $counter -le $nreps ]
do
	echo $counter
	python src/vae_trainer.py -o $vaeoutput --it $epochs --dataset $dataset --prior $prior --mcsamples $mcsamples
	python src/analysis.py --dataset $dataset --serialized $vaeoutput 
	((counter++))
done
python src/results.py --filename analysis_scores_$dataset.p --dataset $dataset --prior $prior --model VAE


echo All done




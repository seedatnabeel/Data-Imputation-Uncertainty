# Data Imputation Uncertainty

This repo presents an approach to modelling uncertainty in data imputation

- It makes use of GAIN (Yoon et al) and a VAE
- Uses MC Dropout to obatin multiple samples for each imputation (i.e. multiple imputation)
- Presents a benchmarking framework of measures to compare uncertainty estimates for data imputation



## Features
* Software engineering best practices for reproducibility: unit testing of modules, object-oriented design, doc strings, command line interface, requirements given
* Results can easily be reproduced by running a Bash script


## Example usage:

A bash script has been provided to ease reproducing the results, as well as, for ease in running the code & different experiments

```
    bash repro.sh
```

To run unit tests (to validate any code changes don't result in silent failures due to breaks in the code):

From the home directory run (make sure nose2 is installed from the requirements.txt)

```
    python -m nose2 
```

Python command line execution can be seen in the bash script, should each component wish to be run seperately, for example

```
    
    python src/create_missing.py --dataset bc --pmiss 0.2 --normalize01 0 -o missing.csv --oref ref.csv -n 50000 --istarget 1
	python src/create_dataset.py -i missing.csv -o interim  --ref ref.csv --target target --dataset bc
	python src/gain_mod.py -i missing.csv -o imputed.csv --it 500 --dataset bc --samples 40
	python src/analysis.py --dataset bc --new True --serialized gain_samples.p --model GAIN
	python src/results.py --filename analysis_scores_bc.p --dataset bc --model GAIN

```

## TODO: Future Improvements

  * Greater testing coverage
  * Docker file for full environment reproducibility
  * Logging of training and hyper-parameters to a comparitive shared environment like Weights and Biases or MLFlow

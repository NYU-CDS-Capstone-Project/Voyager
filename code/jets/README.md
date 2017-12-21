# Graph neural nets for jet physics


## Instructions

### Requirements

- python 3
- pytorch
- scikit-learn

### Email results
Change the RECIPIENT default in constants.py to your email address. Then you will get emails with results and logfiles. Otherwise you won't.

### Data

You need to unzip the tars and put the raw pickle files into data/w-vs-qcd/pickles/raw (make this directory).
The training script will look for data in  data/w-vs-qcd/pickles/preprocessed and if it doesn't find it, make it from the raw stuff.

### Usage

Classification of W vs QCD jets:

```
# Training without Adversarial Network, for running on pileup, copy the pileup combination code from train_with_adversarial.py
python train.py [argparse args]
# Training with Adversarial Network - Hardcoded for Pileup 40, 50 and 60
python train_with_adversarial.py [argparse args]
# Test
python evaluation.py [argparse args]
```

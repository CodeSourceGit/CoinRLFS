# CoinRLFS

environment:

* torch 0.4.0
* python 3.6

# train model

* python main_train.py -d xxx -model xxx -p xxx -it xxx -ft xxx -od xxx


explanations:

* -d: choose dataset
* -model: choose downstream task
* -p: choose poisoned or not
* -it: choose to use Isolation Forest Trainer
* -ft: choose to use Random Forest Trainer
* -od: output file name

example:

* python main_train.py -d spam -model dt -p 1 -it 1 -ft 0 -od result.txt

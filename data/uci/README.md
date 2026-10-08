# Local UCI Regression Data (MC dropout benchmark)

The UCI regression benchmark of Hernandez-Lobato and Adams (2015), as
packaged by Gal and Ghahramani for "Dropout as a Bayesian Approximation"
(2016): each dataset's table plus its fixed train/test splits (20 random
90%/10% splits; 5 for protein) and the hyperparameter grid.

```bash
B=https://raw.githubusercontent.com/yaringal/DropoutUncertaintyExps/master/UCI_Datasets
for d in bostonHousing concrete energy kin8nm naval-propulsion-plant power-plant \
         protein-tertiary-structure wine-quality-red yacht; do
  mkdir -p data/uci/$d
  for f in data index_features index_target n_splits n_epochs n_hidden dropout_rates tau_values; do
    curl -sSL -o data/uci/$d/$f.txt $B/$d/data/$f.txt
  done
  n=$(cat data/uci/$d/n_splits.txt | tr -d ' \n')
  for i in $(seq 0 $((n - 1))); do
    curl -sSL -o data/uci/$d/index_train_$i.txt $B/$d/data/index_train_$i.txt
    curl -sSL -o data/uci/$d/index_test_$i.txt $B/$d/data/index_test_$i.txt
  done
done
```

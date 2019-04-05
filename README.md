##usage
1. train and evaluate the network with logging
2. plot log results

### train and evaluate with logging
```
$ python3 -u use_rnn.py 5 100 tur_words.txt > nn_5_100_tur.log
```

**NB!**

```
$ python3 use_rnn.py --help
usage: python3 THIS_SCRIPT.py HIDDEN_SIZE EPOCHS_NUM FILENAME
HIDDEN_SIZE is the integer number of units in the hidden layer
EPOCHS_NUM  is the number of training epochs
FILENAME (default: STDIN) is the relative path to vocabulary file to train on.
  pass STDIN as FILENAME to use stdin as the vocabulary file

```

### plot results
```
$ cat nn_5_100_tur.log | python3 plot_results
```
plots named `unit_${UNIT_IDX}.png` will appear in the folder


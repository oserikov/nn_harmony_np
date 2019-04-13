now working on:
1. train the RNN with _N_ hidden units in (single) hidden layer to predict next character in word given a current character.
1. given the _{<word<sub>i</sub>, feature<sub>i</sub>>}_ dataset (e.g. _{.., <"türlerinde", +front>, ...}_)
   1. feed the network word-by-word to receive the set _H={h<sub>ij</sub>}_. 
   where _h<sub>ij</sub>_ represents the hidden units activations for the _j_-th letter of _i_-th word
   1. train the supervised classifier to predict _feature_k_ given _{h<sub>k1</sub>, .., h<sub>kl</sub>} 
   (l = len(word<sub>k</sub>))_.
   1. via extracting the features importance from classifier **get the intuition 
   whether there are any units strongly connected with the phonological features**.


- [x] implement the RNN
- [X] encode words with unit activations (embeddings, aren't they). see `sandbox_script.py`
- [X] train the classifier (decision tree?) on the encoded words
  
  now it's not that bad showing smth interesting. see `.ipynb`.
- [ ] extract features importance data
- [ ] miracle, magic etc.


## usage
1. train and evaluate the network with logging
2. plot log results

### train and evaluate with logging
```
$ python3 -u use_rnn.py 5 100 data/tur_words.txt > nn_5_100_tur.log
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
$ cat nn_5_100_tur.log | python3 plot_results.py
```
plots named `unit_${UNIT_IDX}.png` will appear in the folder

Alternatively use 

```
$ cat nn_5_100_tur.log | python3 plot_results.py myprefix
```

and the plots will be named `myprefix_${UNIT_IDX}.png` 

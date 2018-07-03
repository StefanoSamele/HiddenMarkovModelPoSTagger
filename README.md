# Hidden Markov Model PoS Tagger
PoS Tagger for Italian Language using Hidden Markov Model trained on an italian version of the Universal Dependency Treebank
(https://github.com/UniversalDependencies/UD_Italian-ISDT).

## Resources

Inside the resources folder you can find the train, validation and test data sets. These three files are taken directly from
the link above; PoS (part of speech) tags are universal features used to distinguish lexical and grammatical properties of words.

## Utils

Inside utils folder you can find methods for loading data from files and saving/loading the model in the appropriate folder.

An example on how to train and save the model in the appropriate folder is contained in the *trainAndSaveModel.py* file; an example of
load and evaluation is contained in the *loadAndEvaluate.py* file.

## About HMM and PoS Tagging

Hidden Markov Model is a classical formulation for a well known problem: given an input sequence of element (the output of a markov process) we would like
to reconstruct the hidden states. In our case we would like to tag, i.e. assigning a specific label, to each word in a corpus from a pre-defined set of labels.
Main idea is that the probability of a word w_i to have a tag t_i (with i = 1....n) is described as:

```
P(t_i|w_i)= P(w_i|t_i)P(t_i|t_i-1)
```
where P(w_i|t_i) is the probability of seeing the w_i given the tag t_i; P(t_i|t_i-1) is the probability of seeing a tag t at time i given the tag t_i-1 at the previous time. Of course,
P(x|y) is the conditional probability of an event x given the event y.

Given a word w, the task is to find the tag which maximizes the above probability;
the problem is addressed in two steps: 
- Calculate all the possible probabilities;
- Maximize the sequence of possible tags with a dynamic programming technique through the Viterbi Algorithm.

The calculation of  P(w_i|t_i) and P(t_i|t_i-1) is simply based on the count of the occurrences in the corpus. A well known issue is the 
handling of unknown words. Here we follow what is presented in *A second-order Hidden Markov Model for part-of-speech tagging*, paragraph 2.2,
by Scott M. Thede and Mary P. Harper, published in *Proceedings of the 37th annual meeting of the Association for Computational Linguistics on Computational Linguistics*
(pages 175-182). The main idea is to give a word a tag according to his suffix. By this term we simply mean the final sequence of characters of a word.

We implemented this two steps solution inside the HmmTagger class; the two main methods are *train* and *viterbi*. 


## Results
We achieved an accuracy of 0.96448 on the test set. 

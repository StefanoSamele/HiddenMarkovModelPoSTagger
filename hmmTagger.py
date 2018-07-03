import numpy as np

from utils import conlluUtils


# HMM class
class HmmTagger:

    def __init__(self):

        self.corpus = list()

        # List of possible tags:
        self.tags = ["PROPN", "PUNCT",
                        "NOUN", "ADP", "DET", "ADJ","AUX", "VERB",
                        "PRON", "CCONJ", "NUM", "ADV", "INTJ",
                        "SCONJ", "X", "SYM", "PART"]

        # Number of words in the corpus
        self.total_corpus_length = 0

        # Number of sentences in the corpus
        self.number_of_sents = 0

        # List of all words
        self.words = list()

        # Dimension of the suffix and list of all possible suffixes with the specified dimension
        self.suffix_dim = 0
        self.suffixes = list()

        # A-priori probability matrix: p(t_i|t_j) for each tag t_i, t_j
        self.tags_tags = np.empty

        # Likelihood matrix: p(w_i|t_j) for each w_i and tag t_j
        self.words_tags = np.empty

        # Total number of tags for each type
        self.counted_tags = np.empty

        # Matrix of suffixes-tag occurrences
        self.suffixes_tags = np.empty

        # The following arrays check the tag occurrences for particular types of words
        # They will be used in case the word will not be found in the dictionary and it is
        # not possible to apply the suffix matrix
        self.words_hyphens = np.empty
        self.words_digits = np.empty
        self.words_capital = np.empty

        # Probability array of a tag to start a sentence
        self.pit = np.empty

        # Probability array of a tag to end a sentence
        self.pft = np.empty

    def corpus_initialization(self):
        # Corpus
        self.corpus = conlluUtils.get_train_data()

        # Number of words in the corpus
        self.total_corpus_length = len([item[0] for sublist in self.corpus for item in sublist if item[1] != '_'])

        # Number of sentences in the corpus
        self.number_of_sents = len(self.corpus)

        # List of all words
        self.words = list({item[0] for sublist in self.corpus for item in sublist if item[1] != '_'})

        # Dimension of the suffix and list of all possible suffixes with the specified dimension
        self.suffix_dim = 3
        self.suffixes = list({word[-self.suffix_dim:] for word in self.words if (len(word) > self.suffix_dim and
                                                                                 "-" not in word and
                                                                                 not any(d.isdigit() for d in word))})
        # Total number of suffixes
        num_suffixes = len(self.suffixes)

        # Total number of tags
        num_tags = len(self.tags)

        # Total number of words
        num_words = len(self.words)

        # A-priori probability matrix: p(t_i|t_j) for each tag t_i, t_j
        self.tags_tags = np.zeros((num_tags, num_tags))

        # Likelihood matrix: p(w_i|t_j) for each w_i and tag t_j
        self.words_tags = np.zeros((num_words + 1, num_tags))

        # Total number of tags for each type
        self.counted_tags = np.zeros(num_tags)

        # Matrix of suffixes-tag occurrences
        self.suffixes_tags = np.zeros((num_suffixes, num_tags))

        # The following arrays check the tag occurrences for particular types of words
        # They will be used in case the word will not be found in the dictionary and it is
        # not possible to apply the suffix matrix
        self.words_hyphens = np.zeros(len(self.tags))
        self.words_digits = np.zeros(len(self.tags))
        self.words_capital = np.zeros(len(self.tags))

        # Probability array of a tag to start a sentence
        self.pit = np.zeros(num_tags)

        # Probability array of a tag to end a sentence
        self.pft = np.zeros(num_tags)

        print("HMM tagger correctly initialized!")

    def load_from_outside(self, total_corpus_length, number_of_sents, words, suffix_dim, suffixes, counted_tags,
                          tags_tags, words_tags, suffixes_tags, words_hyphens, words_digits, words_capital, pit, pft):
        self.total_corpus_length = total_corpus_length
        self.number_of_sents = number_of_sents
        self.words = words
        self.suffix_dim = suffix_dim
        self.suffixes = suffixes

        self.tags_tags = tags_tags
        self.words_tags = words_tags
        self.counted_tags = counted_tags
        self.suffixes_tags = suffixes_tags
        self.words_hyphens = words_hyphens
        self.words_digits = words_digits
        self.words_capital = words_capital
        self.pit = pit
        self.pft = pft

    # Count each tag occurrence
    def count_tag(self):
        tag_list = [item[1] for sublist in self.corpus for item in sublist]
        tags_array = np.array(self.tags)
        indexes = [np.where(tags_array == t)[0] for t in tag_list if t != '_']
        unique, counts = np.unique(indexes, return_counts=True)
        return counts

    # Train method to fill the core matrix and arrays (counted_tags, coupled_tags, word_tags)
    def train(self):
        print("Begin HMM training")

        self.counted_tags = self.count_tag()

        for sent in self.corpus:
            for number, couple in enumerate(sent):
                if number == 0:

                    # Count only likelihood for words
                    word_index = self.words.index(couple[0])
                    tag_index = self.tags.index(couple[1])
                    self.words_tags[word_index][tag_index] += 1.0

                    # Count likelihood for suffix
                    if len(couple[0]) > self.suffix_dim and "-" not in couple[0] and\
                            not any(d.isdigit() for d in couple[0]):
                        suffix_index = self.suffixes.index(couple[0][-self.suffix_dim:])
                        self.suffixes_tags[suffix_index][tag_index] += 1.0

                    # Count likelihood for excluded type of words, i.e. those containing an hyphens or a digit
                    # Note that for beginning of sentence words counting how many begin with capital letter has no sense
                    if "-" in couple[0] and not any(d.isdigit() for d in couple[0]):
                        self.words_hyphens[tag_index] += 1.0

                    if any(d.isdigit() for d in couple[0]) and "-" not in couple[0]:
                        self.words_digits[tag_index] += 1.0

                    # Count PIT
                    self.pit[tag_index] += 1.0

                    old_tag_index = tag_index

                else:

                    word_index = self.words.index(couple[0])
                    tag_index = self.tags.index(couple[1])

                    # Count prior
                    self.tags_tags[old_tag_index][tag_index] += 1.0

                    # Count likelihood for word
                    self.words_tags[word_index][tag_index] += 1.0

                    # Count likelihood for suffix
                    if len(couple[0]) > self.suffix_dim and "-" not in couple[0] and\
                            not any(d.isdigit() for d in couple[0]) and\
                            not couple[0][0].isupper():
                        suffix_index = self.suffixes.index(couple[0][-self.suffix_dim:])
                        self.suffixes_tags[suffix_index][tag_index] += 1.0

                    # Count likelihood for excluded type of words
                    if "-" in couple[0] and not any(d.isdigit() for d in couple[0]) and\
                            not couple[0][0].isupper():
                        self.words_hyphens[tag_index] += 1.0

                    if any(d.isdigit() for d in couple[0]) and "-" not in couple[0] and\
                            not couple[0][0].isupper():
                        self.words_digits[tag_index] += 1.0

                    if couple[0][0].isupper() and not any(d.isdigit() for d in couple[0]) and "-" not in couple[0]:
                        self.words_capital[tag_index] += 1.0

                    # Update the old index with the new one for next round
                    old_tag_index = tag_index

            # Count PFT
            last_tag_index = self.tags.index(sent[-1][1])
            self.pft[last_tag_index] += 1.0

        # Normalize the frequencies
        self.words_tags = self.words_tags / self.counted_tags
        self.suffixes_tags = self.suffixes_tags / self.counted_tags
        self.words_hyphens = self.words_hyphens / self.counted_tags
        self.words_digits = self.words_digits / self.counted_tags
        self.words_capital = self.words_capital / self.counted_tags
        self.pit = self.pit / self.number_of_sents
        self.pft = self.pft / self.number_of_sents

        # For unknown words if strategy fails, assign a flat probability
        self.words_tags[-1, :] = 1.0 / len(self.tags)
        # self.words_tags[-1, :] = self.counted_tags / self.total_corpus_length

        transposed_counted_tags = np.reshape(self.counted_tags, (17, 1))
        self.tags_tags = self.tags_tags / transposed_counted_tags

        print("HMM trained!")

    def viterbi(self, T, sent):

        # Fixed number of tags
        N = 17

        # Create data structures
        back_pointer = np.zeros((N, T))
        viterbi = np.zeros((N, T))

        if sent[0] in self.words:
            row = self.words.index(sent[0])
            word_type = "normal"
        elif len(sent[0]) > self.suffix_dim and "-" not in sent[0] and not any(d.isdigit() for d in sent[0]):
            if sent[0] in self.suffixes:
                row = self.suffixes.index(sent[0][-self.suffix_dim:])
                word_type = "suffix"
            else:
                row = -1
                word_type = "normal"
        elif "-" in sent[0] and not any(d.isdigit() for d in sent[0]):
            word_type = "hyphens"
        elif any(d.isdigit() for d in sent[0]) and "-" not in sent[0]:
            word_type = "digit"
        else:
            row = -1
            word_type = "normal"

        for s in range(0, N):

            prob_initial_tag = self.pit[s]

            if word_type == "normal":
                prob_observation = self.words_tags[row][s]
            elif word_type == "suffix":
                prob_observation = self.suffixes_tags[row][s]
            elif word_type == "hyphens":
                prob_observation = self.words_hyphens[s]
            elif word_type == "digit":
                prob_observation = self.words_digits[s]
            else:
                raise ValueError('This should never ever ever happen: word_type out of known values')

            # The first state in the the viterbi matrix is given by the probability
            # of a tag to begin the sentence multiplied the state observation likelihood
            viterbi[s][0] = prob_initial_tag * prob_observation

        for t in range(1, T):

            if sent[t] in self.words:
                word_index = self.words.index(sent[t])
                word_type = "normal"
            elif len(sent[t]) > self.suffix_dim and "-" not in sent[t] and not any(d.isdigit() for d in sent[t]) and\
                    not sent[t][0].isupper:
                if sent[t][-self.suffix_dim:] in self.suffixes:
                    word_index = self.suffixes.index(sent[t][-self.suffix_dim:])
                    word_type = "suffix"
                else:
                    word_index = -1
                    word_type = "normal"
            elif "-" in sent[t] and not any(d.isdigit() for d in sent[t]) and not sent[t][0].isupper():
                word_type = "hyphens"
            elif any(d.isdigit() for d in sent[t]) and "-" not in sent[t] and not sent[t][0].isupper():
                word_type = "digit"
            elif sent[t][0].isupper() and not any(d.isdigit() for d in sent[t]) and "-" not in sent[t]:
                word_type = "capital"
            else:
                word_index = -1
                word_type = "normal"

            for s in range(0, N):

                if word_type == "normal":
                    b = self.words_tags[word_index, s]
                elif word_type == "suffix":
                    b = self.suffixes_tags[word_index, s]
                elif word_type == "hyphens":
                    b = self.words_hyphens[s]
                elif word_type == "digit":
                    b = self.words_digits[s]
                elif word_type == "capital":
                    b = self.words_capital[s]
                else:
                    raise ValueError('This should never ever ever happen: word_type out of known values')

                # The viterbi entry at time t and state s is calculated multiplying
                # the previous viterbi column for all the possible
                # transition probability of state s and the likelihood of the analyzed word with state s
                viterbi[s][t] = np.max(viterbi[:, t-1] * self.tags_tags[:, s]) * b
                back_pointer[s][t] = np.argmax(viterbi[:, t-1] * self.tags_tags[:, s])

        back_pointer_final = np.argmax(viterbi[:, T-1] * self.pft)

        tags = self.get_viterbi_path(back_pointer, back_pointer_final)

        return [self.tags[int(i)] for i in tags]

    def get_viterbi_path(self, back_pointer, back_pointer_final):

        index = back_pointer_final

        # Get the number of state (0 state is not included)
        l = back_pointer.shape[1]
        path = np.zeros(l)
        path[l-1] = back_pointer_final

        for t in range(0, l-1):
            new_index = back_pointer[index][l-1-t]
            index = int(new_index)
            path[l-2-t] = int(new_index)

        return path

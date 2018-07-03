from hmmTagger import HmmTagger
from utils import conlluUtils
from utils.filesystem_utils import load_model


tagger = HmmTagger()

total_corpus_length, number_of_sents, words, suffix_dim, suffixes, counted_tags, tags_tags, words_tags, suffixes_tags, \
    words_hyphens, words_digits, words_capital, pit, pft = load_model()


tagger.load_from_outside(total_corpus_length, number_of_sents, words, suffix_dim, suffixes, counted_tags, tags_tags,
                         words_tags, suffixes_tags, words_hyphens, words_digits, words_capital, pit, pft)

sent_list = conlluUtils.get_test_data()

predicted_pos_list = list(list())

print("Starting prediction...")

for sent in sent_list:
    tag_seq = tagger.viterbi(len(sent), 17, sent)
    predicted_pos_list.append((tag_seq, sent))

print("Prediction completed!")

sent_list_with_tags = conlluUtils.get_test_data(True)
pos_scores = 0
num_tag = 0

errors = list()

print("Starting evaluation...")

for i, sent in enumerate(sent_list_with_tags):
    correct_tags = list()
    sentence = list()
    for couple in sent:
        correct_tags.append(couple[1])
        sentence.append(couple[0])
    matches = [k for k, j in zip(correct_tags, predicted_pos_list[i][0]) if k == j]
    # Visual check with sub sampling
    # if i % 100 == 0:
    #    print(predicted_pos_list[i][1])
    #    print(correct_tags)
    #    print(predicted_pos_list[i][0])
    #    print("-------------------")
    pos_scores += len(matches)
    num_tag += len(correct_tags)
    if len(matches) != len(correct_tags):
        errors.append((predicted_pos_list[i][1], correct_tags, predicted_pos_list[i][0]))

print("Accuracy:", float(pos_scores)/num_tag)

count_error_4_smoothing = 0.0

count_error = 0.0

for el in errors:
    for index, word in enumerate(el[0]):
        if el[1][index] != el[2][index]:
            count_error += 1.0
            if word not in tagger.words:
                count_error_4_smoothing += 1.0

print("Number of errors:", count_error)

print("Unknown words errors percentage:", count_error_4_smoothing/len(errors)*100)
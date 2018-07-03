from conllu import parse


def get_train_data():
    return get_data('resources/it_isdt-ud-train.conllu', True)


def get_test_data(tag=False):
    return get_data('resources/it_isdt-ud-test.conllu', tag)


def get_validation_data(tag=False):
    return get_data('resources/it_isdt-ud-dev.conllu', tag)


def get_data(path, tag):
    # read train data in conllu format
    # IF TAG == TRUE return a list of list (each list represent a sentence) of tuples
    # each tuple contains the actual word and its POS TAG
    # IF TAG == FALSE return a list of list of words
    print("Read data from: " + path)
    with open(path) as f:
        read_data = f.read()
    f.closed

    # GET PRE-TAGGED TRAIN DATA
    conllu_data = parse(read_data)

    global_list_of_tuples = []
    list_of_tuples = []
    for sent in conllu_data:
        for dict in sent:
            if dict.get("upostag") != "_":
                if tag:
                    list_of_tuples.append((dict.get("lemma"), dict.get("upostag")))
                else:
                    list_of_tuples.append(dict.get("lemma"))
        global_list_of_tuples.append(list_of_tuples)
        list_of_tuples = []

    return global_list_of_tuples

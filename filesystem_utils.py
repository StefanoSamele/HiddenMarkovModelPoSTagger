import pickle


def save_model(total_corpus_length, number_of_sents, words, suffix_dim, suffixes, counted_tags, tags_tags, words_tags,
               suffixes_tags, words_hyphens, words_digits, words_capital, pit, pft):
    with open("model/words.file", "wb") as f:
        pickle.dump(words, f, pickle.HIGHEST_PROTOCOL)
    with open("model/suffixes.file", "wb") as f:
        pickle.dump(suffixes, f, pickle.HIGHEST_PROTOCOL)
    with open("model/counted_tags.file", "wb") as f:
        pickle.dump(counted_tags, f, pickle.HIGHEST_PROTOCOL)
    with open("model/tags_tags.file", "wb") as f:
        pickle.dump(tags_tags, f, pickle.HIGHEST_PROTOCOL)
    with open("model/words_tags.file", "wb") as f:
        pickle.dump(words_tags, f, pickle.HIGHEST_PROTOCOL)
    with open("model/suffixes_tags.file", "wb") as f:
        pickle.dump(suffixes_tags, f, pickle.HIGHEST_PROTOCOL)
    with open("model/words_hyphens.file", "wb") as f:
        pickle.dump(words_hyphens, f, pickle.HIGHEST_PROTOCOL)
    with open("model/words_digits.file", "wb") as f:
        pickle.dump(words_digits, f, pickle.HIGHEST_PROTOCOL)
    with open("model/words_capital.file", "wb") as f:
        pickle.dump(words_capital, f, pickle.HIGHEST_PROTOCOL)
    with open("model/pit.file", "wb") as f:
        pickle.dump(pit, f, pickle.HIGHEST_PROTOCOL)
    with open("model/pft.file", "wb") as f:
        pickle.dump(pft, f, pickle.HIGHEST_PROTOCOL)

    dim_config = {"total_corpus_length": total_corpus_length, "number_of_sents": number_of_sents, "suffix_dim": suffix_dim}

    with open("model/dim_config.file", "wb") as f:
        pickle.dump(dim_config, f, pickle.HIGHEST_PROTOCOL)

def load_model():
    with open("model/words.file", "rb") as f:
        words = pickle.load(f)
    with open("model/suffixes.file", "rb") as f:
        suffixes = pickle.load(f)
    with open("model/counted_tags.file", "rb") as f:
        counted_tags = pickle.load(f)
    with open("model/tags_tags.file", "rb") as f:
        tags_tags = pickle.load(f)
    with open("model/words_tags.file", "rb") as f:
        words_tags = pickle.load(f)
    with open("model/suffixes_tags.file", "rb") as f:
        suffixes_tags = pickle.load(f)
    with open("model/words_hyphens.file", "rb") as f:
        words_hyphens = pickle.load(f)
    with open("model/words_digits.file", "rb") as f:
        words_digits = pickle.load(f)
    with open("model/words_capital.file", "rb") as f:
        words_capital = pickle.load(f)
    with open("model/pit.file", "rb") as f:
        pit = pickle.load(f)
    with open("model/pft.file", "rb") as f:
        pft = pickle.load(f)
    with open("model/dim_config.file", "rb") as f:
        dim_config = pickle.load(f)

    return dim_config["total_corpus_length"], dim_config["number_of_sents"], words, dim_config["suffix_dim"], suffixes, \
           counted_tags, tags_tags, words_tags, suffixes_tags, words_hyphens, words_digits, words_capital, pit, pft



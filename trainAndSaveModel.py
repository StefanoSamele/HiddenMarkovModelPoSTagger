from hmmTagger import HmmTagger
from utils.filesystem_utils import save_model

tagger = HmmTagger()

tagger.corpus_initialization()

tagger.train()

save_model(tagger.total_corpus_length, tagger.number_of_sents, tagger.words, tagger.suffix_dim, tagger.suffixes,
           tagger.counted_tags, tagger.tags_tags, tagger.words_tags, tagger.suffixes_tags, tagger.words_hyphens,
           tagger.words_digits, tagger.words_capital, tagger.pit, tagger.pft)

print("Model saved!")
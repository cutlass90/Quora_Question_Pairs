import gensim
from gensim.models.keyedvectors import KeyedVectors

model = KeyedVectors.load_word2vec_format('/media/nazar/F64CA6774CA631F3/GoogleNews-vectors-negative300.bin', binary=True)

model.word_vec('love')
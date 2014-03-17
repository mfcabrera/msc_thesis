#!/usr/bin/env python

import gensim,  logging 
from  word2vec_util.io import FileSetencesGenerator
import sys 


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    sentences = FileSetencesGenerator(sys.argv[1])
    model = gensim.models.Word2Vec(sentences, min_count=10, size=150, workers=10, window=10)
    model.save("mymodel.bin")
    model.save_word2vec_format("mymodel_word2vec.bin")

    






#! -*- coding: utf-8 -*-
import logging
import sys
sys.path.append('./SIF/src')
import data_io, params, SIF_embedding
logging.basicConfig(level=logging.INFO)


def main(sentences, wordfile: str, weightfile: str, weightpara: float = 1e-3, rmpc: int = 1):
    # load word vectors
    (words, We) = data_io.getWordmap(wordfile)
    # load word weights
    word2weight = data_io.getWordWeight(weightfile, weightpara) # word2weight['str'] is the weight for the word 'str'
    weight4ind = data_io.getWeight(words, word2weight) # weight4ind[i] is the weight for the i-th word
    # load sentences
    x, m, _ = data_io.sentences2idx(sentences, words) # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
    w = data_io.seq2weight(x, m, weight4ind) # get word weights

    # set parameters
    params = params.params()
    params.rmpc = rmpc
    # get SIF embedding
    embedding = SIF_embedding.SIF_embedding(We, x, w, params) # embedding[i,:] is the embedding for sentence i


def test():
    # input
    wordfile = './text/jawiki.all_vectors.100d.txt'  # word vector file, can be downloaded from GloVe website
    weightfile = './text/vocab.txt'  # each line is a word and its frequency
    weightpara = 1e-3  # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
    rmpc = 1  # number of principal components to remove in SIF weighting scheme
    sentences = ['これ は いい 壺 だ', 'きょう の ニュース は なかなか いい']

    main(
        sentences,
        wordfile,
        weightfile)


if __name__ == '__main__':
    test()

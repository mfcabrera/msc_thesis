import numpy as np

def load_word2vec_vocab(path):
    """ loads a word2vec vocab from the file system.
    Returns a dictionary in where they keys are the word and the 
    values the number of times the word was found in the training vocabulary.
    
    Keyword arguments:
    path -- path where  the word2vec vocabulary file is located
    """
    
    vocab = []
    
    with(open(path)) as vocab_file:
        for line in vocab_file:
            word,count =   line.split()
            #vocab[word] = int(count)
            vocab.append(word)

    return vocab

# let's define a method to do this:
def get_doc_vector(vector,model):
    """ Returns the docter the word vector mean representation"""
    idx = vector.todense() == 1 
    doc_matrix = model.syn0[np.array(idx.flat)]
    
    return np.mean(doc_matrix,axis=0)


if __name__ == '__main__':
    from gensim.models import word2vec

    model =  word2vec.Word2Vec.load_word2vec_format("../../experiments/dewiki-GINI001.bin", binary=True)
    
    

    v = load_word2vec_vocab("../../experiments/gini.vocab")
    print("Vocabulary with %d  words" % len(v))
    print(v)

    print("Let's test something")
 

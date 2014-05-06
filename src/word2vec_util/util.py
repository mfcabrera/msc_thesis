import numpy as np
import json
from os import listdir
from os.path import isfile, join
from nltk.corpus import stopwords
import cPickle as pickle


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
def get_doc_vector(vector, word_vectors, mean=True):
    """ Returns the docter the word vector mean representation"""
    idx = vector.todense() == 1 
    doc_matrix = word_vectors[np.array(idx.flat)]
    if mean:
        return np.mean(doc_matrix, axis=0)
    else:
        return np.sum(doc_matrix, axis=0)
    


#Let's define a funciton to create 
def create_data_vectors(data, word_vectors, mean=True):
    """ Create data vectors from a binary term vector  
        data: a binary term vector representation of the documents. One row per document.
        word_mode: the word2vec wordvector model to get the wordvector from
    """
    data_dense  = np.zeros((data.shape[0],word_vectors.shape[1]))
    #get the document vectors
    for i,item  in enumerate(data):
        #print("What? %d" % i)
        data_dense[i] = get_doc_vector(item, word_vectors, mean=mean)
        
        #if(  (np.isfinite(data_dense[i]).all()) ):
            #print (documents[i])
            #print (data_dense[i])
    return data_dense


def load_model_descriptions(json_files):
    """ load model description from a list of json files"""
    
    parameters = dict()
    
    for parameter_file in json_files:
        with open(parameter_file,'r') as input:
            runs = json.load(input)
            
            for run_code in runs:
                #print "Reading meta info from  %s " % run_code
                output = "dewiki-" + run_code + ".bin"
                parameters[output] = runs[run_code]
        
    return parameters
    
    
def load_documents_and_labels(data_dir, categories):
    """ Load the document of the labels.
    data_dir: the directoy where the documents are stored. The documents should be stored
    in subdirectories - where the name of the directory is the label.
    categoris: is the list of the labels or categories that is going to be extracted from the data_dir
    directory. A subdirectoy of 'data_dir' should exists for each cateogry in 'category'.

    return the list of documents, the labels and a dictionary cat : list_of_files
    """
    files_cat = {}
    labels = []

    for cat in categories:
        path = join(data_dir,cat)
        files_cat[cat] = [ join(path,f) for f in listdir(path) if isfile(join(path,f)) ]
        labels.extend([cat] * len(files_cat[cat]))

    documents = []
    for cat in categories:
        documents.extend(files_cat[cat])
    

    documents = np.asarray(documents)
    labels = np.asarray(labels)

    return documents, labels, files_cat
    
    
def load_gini_stop_words(additional_stop_words_file):
    """ Load gini stop words for the German language and append the NLTK stop word list
    returns: a list of the gini stop words and the NLTK stop words with the special german character
    scaped. """
    
    gini_stop_words = pickle.load(open(additional_stop_words_file))
    nltk_stop_words = stopwords.words('german')

    final_stopwords = map(lambda x:  
                          x.replace('\xc3\xb6','oe')
                          .replace('\xc3\xbc','ue')
                          .replace('\xc3\xa4','ae') 
                          .replace('\xc3\x9f', 'ss')
                          , nltk_stop_words)
    




if __name__ == '__main__':
    from gensim.models import word2vec

    model =  word2vec.Word2Vec.load_word2vec_format("../../experiments/dewiki-GINI001.bin", binary=True)
    
    

    v = load_word2vec_vocab("../../experiments/gini.vocab")
    print("Vocabulary with %d  words" % len(v))
    print(v)

    print("Let's test something")
 



import numpy as np
import json
from os import listdir
from os.path import isfile, join


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
def get_doc_vector(vector, model):
    """ Returns the docter the word vector mean representation"""
    idx = vector.todense() == 1 
    doc_matrix = model.syn0[np.array(idx.flat)]
    
    return np.mean(doc_matrix,axis=0)


#Let's define a funciton to create 
def create_data_vectors(data, word_model):
    """ Create data vectors from a binary term vector  
        data: a binary term vector representation of the documents. One row per document.
        word_mode: the word2vec wordvector model to get the wordvector from
    """
    data_dense  = np.zeros((data.shape[0],word_model.syn0.shape[1]))
    #get the document vectors
    for i,item  in enumerate(data):
        #print("What? %d" % i)
        data_dense[i] = get_doc_vector(item,word_model)
        
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
    
    




if __name__ == '__main__':
    from gensim.models import word2vec

    model =  word2vec.Word2Vec.load_word2vec_format("../../experiments/dewiki-GINI001.bin", binary=True)
    
    

    v = load_word2vec_vocab("../../experiments/gini.vocab")
    print("Vocabulary with %d  words" % len(v))
    print(v)

    print("Let's test something")
 

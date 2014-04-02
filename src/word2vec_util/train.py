import numpy as np
from gensim.models import word2vec

from sklearn import metrics
from sklearn.svm import SVC

from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing  import MinMaxScaler


from .util import create_data_vectors

#let's define a function to define to train the model

def train_with_wordvectors(model_files, data, labels, train_index=None, test_index=None, kernel='linear', n_jobs=8):
    """ Train a counte word vector with a word2vec model for document classiification and return the accuracies on t        he testing set"""

    import copy
        
    C_range = 10.0 ** np.arange(-2, 9)    
    gamma_range = 10.0 ** np.arange(-5, 2)

    if(kernel == 'linear'):
        param_grid = dict(C=C_range)
    elif (kernel == 'rbf'):
        param_grid = dict(C=C_range,gamma=gamma_range)
    else:
        raise Exception('Invalid kernel type')
    
    
    predictions =  {}
    classifiers = {}
    accuracies = {}
    predictions_train = {}
    
    if(train_index is None): 
        # So training = testing = None
        train_index = np.arange(0,data.shape[0])
        
    if(test_index is None):        test_index = train_index
        
    #X_train = data[train_index]
    y_train = labels[train_index]

    #X_test = data[test_index]
    y_test = labels[test_index]
   
    for w_model in model_files :
        
        print("Training with %s " %  w_model)

        model =  word2vec.Word2Vec.load_word2vec_format(w_model, binary=True)
        data_dense = create_data_vectors(data,model)
        
        scaler = MinMaxScaler(copy=False)
        data_dense = scaler.fit_transform(data_dense)
        
        
        X_train = data_dense[train_index]
        X_test =  data_dense[test_index]
                   
       
        cv = StratifiedKFold(y=y_train, n_folds=2)
        grid = GridSearchCV(SVC(kernel=kernel,class_weight='auto' ), param_grid=param_grid, 
                            cv=cv,verbose=0,n_jobs=n_jobs)
        grid.fit(X_train, y_train)
        
        clf = grid.best_estimator_
        classifiers[w_model] =  copy.deepcopy(grid.best_estimator_)
        predictions[w_model] = clf.predict(X_test)
        predictions_train[w_model] = clf.predict(X_train)

        #print(y_test)
        #print(predictions[w_model])
        
        
        accuracies[w_model] =  metrics.classification_report(y_test  , predictions[w_model])   
        
        
        print("The best classifier is: ", classifiers[w_model])
        
        
    return classifiers,predictions,accuracies,predictions_train

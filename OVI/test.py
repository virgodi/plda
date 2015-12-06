import sys
import os.path
sys.path.append(os.path.join('..', 'util'))

import set_compiler
set_compiler.install()

import pyximport
pyximport.install()

import numpy as np
import lda_ovi
import time

if __name__ == '__main__':

    ''' LOADING THE LAS VEGAS DATA '''

    dtm = np.load('../../lv_dtm.npy').astype(int)
    vocablv = np.load('../../vocab.npy')

    ''' Initialising parameters '''

    ''' Batchsize '''
    S = 40 

    max_iter = 1000
    threshold = 0.00000001

    '''Hyperparameters'''
    tau = 512
    kappa = 0.7
    alpha = 1/40
    eta = 1/40
    num_topics = 40
    num_threads = 8
    

    '''Constructing the model'''
    lda = lda_ovi.LDA(num_topics, num_threads)
    print ''
    print 'Fitting an Online Variational Inference model on Las Vegas restaurant reviews with 40 topics:'
    '''Fitting the model'''
    np.random.seed(0)
    start = time.time()
    lda.fit_p(dtm,S, tau, kappa)
    end = time.time()-start
    
    ''' Printing Some Topics '''
    print 'Some topics:'
    print ''
    topic_word = lda.topics
    n_top_words = 10
    for i, topic_dist in enumerate(topic_word):
        if i in [2,4,6,10,15,23,27,28,30,37]:
            topic_words = np.array(vocablv)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
            print(u'Topic {}: {}'.format(i, ' '.join(topic_words)))
            print ''
    print '' 
    print 'Run time: %s sec with 8 threads' %(end)
    print ''


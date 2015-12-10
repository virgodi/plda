import sys
import os.path
sys.path.append('util')

import set_compiler
set_compiler.install()

import pyximport
pyximport.install()

import numpy as np
from OnlineVariationalInference.ovi import oviLDA
from CollapsedGibbsSampler.cgs import cgsLDA
from LDAutil import Evaluation

import time

if __name__ == '__main__':

    # LOADING THE LAS VEGAS DATA 

    dtm = np.load('../reuters/dtm.npy')
    vocablv = np.load('../reuters/vocab.npy')

    # Initialising parameters

    # Batchsize 

    num_topics = 20
    num_threads = 8
    

    # Constructing the model

    # UNCOMMENT THE NEXT LINE TO USE OVI:
    lda = oviLDA(num_topics, num_threads)

    # UNCOMMENT THE NEXT LINE TO USE CGS:
    # lda = cgsLDA(num_topics, num_threads)
    print ''
    print 'Fitting an LDA model on the Reuters dataset with 20 topics:'
    
    # Fitting the model
    np.random.seed(0)
    start = time.time()
    lda.fit(dtm)
    end = time.time()-start

    print 'Run time: %s sec with 8 threads' %(end)
    #Evaluation.print_topic(lda,vocablv, num_top_words=10)
    print ''
    print Evaluation.perplexity(lda,dtm)
    print ''
    print 'Perplexity on train dataset is %s' % lda.perplexity_train
    
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

    dtm = np.load('../../lv_dtm.npy').astype(int)
    vocablv = np.load('../../vocab.npy')

    # Initialising parameters

    # Batchsize 

    S = 40 

    max_iter = 1000
    threshold = 0.00000001

    # Hyperparameters

    tau = 512
    kappa = 0.7
    alpha = 1/40
    eta = 1/40
    num_topics = 40
    num_threads = 8
    

    # Constructing the model
    lda = oviLDA(num_topics, num_threads)
    print ''
    print 'Fitting an Online Variational Inference model on Las Vegas restaurant reviews with 40 topics:'
    
    # Fitting the model
    np.random.seed(0)
    start = time.time()
    lda.fit(dtm)
    end = time.time()-start

    print 'Run time: %s sec with 8 threads' %(end)
    Evaluation.print_topic(lda,vocablv, num_top_words=10)
    print ''
    print Evaluation.perplexity(lda,dtm)
    print ''
    print 'Perplexity on train dataset is %s' % lda.perplexity_train
    
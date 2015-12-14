import lda
import sys
sys.path.append('util')

import set_compiler
set_compiler.install()

import pyximport
pyximport.install()

import plda

from timer import Timer
import logging
import numpy as np

logger = logging.getLogger('lda')
logger.propagate = False

X = lda.datasets.load_reuters()
X_train = X[0:375]
X_test = X[375:395]
iterations = 1000
vocab = lda.datasets.load_reuters_vocab()
test = plda.LDA(method='cgs', num_topics=20)
n_top_words = 8

print str(iterations) + " Iterations"
for num_threads in [16]:
    for sync in [10]:
        test.set_sync_interval(sync)
        #run training data on two methods of parallel fit
        with Timer() as t:
            test.fit(X_train, num_threads, 0.1, 0.01)
            topic_word = test.K_V 
            for i, topic_dist in enumerate(topic_word):
                topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
                print('Topic {}: {}'.format(i, ' '.join(topic_words)))
        print "Train Copy K_V: {} threads, {} sync_step:".format(num_threads, sync) + str(t.interval)
        with Timer() as t:
            test.fit(X_train, num_threads, 0.1, 0.01, True)
            topic_word = test.K_V 
            for i, topic_dist in enumerate(topic_word):
                topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
                print('Topic {}: {}'.format(i, ' '.join(topic_words)))
        print "Train Lock K_V:{} threads, {} sync_step:".format(num_threads, sync) + str(t.interval)
        
        #run inference on two methods of parallel fit
        
        with Timer() as t:
            document_topic = test.inference(X_test, iterations, num_threads)
            print document_topic
        print "Test Copy K_V: {} threads, {} sync_step:".format(num_threads, sync) + str(t.interval)
        with Timer() as t:
            document_topic = test.inference(X_test, iterations, num_threads, True)
            print document_topic
        print "Test Lock K_V:{} threads, {} sync_step:".format(num_threads, sync) + str(t.interval)
        
        

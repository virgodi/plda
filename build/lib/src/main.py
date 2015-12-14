import lda
import sys
sys.path.append('util')

import set_compiler
set_compiler.install()

import pyximport
pyximport.install()

from CollapsedGibbsSampler.cgs import cgsLDA

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
test = cgsLDA(20, iterations=iterations)
n_top_words = 8

print str(iterations) + " Iterations"
for num_threads in [16]:
    test.set_num_threads(num_threads)
    for sync in [10]:
        test.set_sync_interval(sync)
        #run training data on two methods of parallel fit
        with Timer() as t:
            test.set_split_words(False)
            test.fit(X_train)
            topic_word = test.topics
            for i, topic_dist in enumerate(topic_word):
                topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
                print('Topic {}: {}'.format(i, ' '.join(topic_words)))
        print "Train Copy K_V: {} threads, {} sync_step:".format(num_threads, sync) + str(t.interval)
        with Timer() as t:
            test.set_split_words(True)
            test.fit(X_train)
            topic_word = test.topics 
            for i, topic_dist in enumerate(topic_word):
                topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
                print('Topic {}: {}'.format(i, ' '.join(topic_words)))
        print "Train Lock K_V:{} threads, {} sync_step:".format(num_threads, sync) + str(t.interval)
        
        #run inference on two methods of parallel fit
        
        with Timer() as t:
            test.set_split_words(False)
            document_topic = test.transform(X_test)
            print document_topic
        print "Test Copy K_V: {} threads, {} sync_step:".format(num_threads, sync) + str(t.interval)
        with Timer() as t:
            test.set_split_words(True)
            document_topic = test.transform(X_test)
            print document_topic
        print "Test Lock K_V:{} threads, {} sync_step:".format(num_threads, sync) + str(t.interval)
        
        

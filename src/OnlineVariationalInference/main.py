import sys
import os.path
sys.path.append(os.path.join('..', 'util'))

import set_compiler
set_compiler.install()

import pyximport
pyximport.install()

import numpy as np
import lda_vi_serial
import lda_vi_cython
from lda_vi import LDA_vi
import time
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix


if __name__ == '__main__':
    corpus_ap = False

    if corpus_ap:
        # ### AP dtm
        # Read the vocabulary
        vocabulary = np.loadtxt('../ap/vocab.txt', dtype=str)
        V = len(vocabulary)

        # Read the data
        # Output format is a list of document (dtm) with
        # document: array([[index1, count1], ... , [index2, count2]])

        # To build the sparse matrix
        counts = []
        row_ind = []
        col_ind = []

        with open('../ap/ap.dat', 'r') as f:
            for i, row in enumerate(f):
                # row format is:
                #    [M] [term_1]:[count] [term_2]:[count] ...  [term_N]:[count]
                row_raw = row.split(' ')
                M = int(row_raw[0])
                document = np.zeros((M, 2))

                row_ind += M*[i]
                for j, w in enumerate(row_raw[1:]):
                    document[j, :] = [int(u) for u in w.split(':')]
                counts += list(document[:, 1])
                col_ind += list(document[:, 0])

        # dtm size
        C = i + 1

        # Building the dtm matrix
        dtm = csc_matrix((counts, (row_ind, col_ind)), shape=(C, V))
        dtm = dtm.toarray().astype(int)
    else:
        #  ### HARD: Artificial dtm
        # 100 docs
        # 450 words
        # 9 artificial topics (for each bag of 5 words)
        # N_docs = 100
        # N_words = 450
        # num_topics = 9
        # dtm = np.zeros((N_docs, N_words), dtype=int)  # shape (docs, words)
        # vocabulary = np.arange(N_words)

        # block = 10 * np.ones((20, 50), dtype=int)
        # for i in range(num_topics):
        #     dtm[10*i:10*(i+2), 50*i:50*(i+1)] = block

    # #### Parameters
    #dtm = np.load('../../lv_dtm.npy').astype(int)
    #vocablv = np.load('../Lasvegas/lv_vocab10.npy')
    S = 10
    max_iter = 100
    tau = 512
    kappa = 0.7
    alpha = 1/40
    eta = 1/40
    threshold = 0.00000001
    num_threads = 1
    num_topics = 40
    ndoc, nvoc = dtm.shape

    # ### Opt
    np.random.seed(0)
    # Initialization
    model = LDA_vi(num_topics, num_threads)
    time1 = time.time()
    # lda_batch makes in place operations
    model.fit_s(dtm, S, tau=512, kappa=0.7)

    time1_stop = time.time() - time1
    print 'Opt Time is', time1_stop, ' s'

    # ### Parallel
    # Initialization
    num_threads = 1
    model_parallel1 = LDA_vi(num_topics, num_threads)
    time1 = time.time()
    # lda_batch makes in place operations
    model_parallel1.fit_p(dtm, S)
    time1_stop = time.time() - time1
    print 'Parallel Time with {} threads is '.format(num_threads), time1_stop, ' s'

    # Initialization
    num_threads = 4
    model_parallel2 = LDA_vi(num_topics, num_threads)
    time1 = time.time()
    # lda_batch makes in place operations
    model_parallel2.fit_p(dtm, S)
    time1_stop = time.time() - time1
    print 'Parallel Time with {} threads is '.format(num_threads), time1_stop, ' s'

    # Initialization
    num_threads = 8
    model_parallel3 = LDA_vi(num_topics, num_threads)
    time1 = time.time()
    # lda_batch makes in place operations
    model_parallel3.fit_p(dtm, S)
    time1_stop = time.time() - time1
    print 'Parallel Time with {} threads is '.format(num_threads), time1_stop, ' s'

    num_threads = 16
    model_parallel4 = LDA_vi(num_topics, num_threads)
    time1 = time.time()
    # lda_batch makes in place operations
    model_parallel4.fit_p(dtm, S)
    time1_stop = time.time() - time1
    print 'Parallel Time with {} threads is '.format(num_threads), time1_stop, ' s'

    # ### Serial

    # np.random.seed(0)
    # lambda_ = np.random.gamma(100., 1./100., (num_topics, nvoc))
    # time2 = time.time()
    # lambda_serial, gamma_serial = lda_vi_serial.lda_batch(lambda_, dtm, num_topics, S, 512, 0.7)
    # time2_stop = time.time() - time2
    # print 'Serial Time is', time2_stop, ' s'

    if corpus_ap:
        # dtm AP: printing the result
        print 'Cython topics'
        topic_word = model.topics  # model.components_ also works
        n_top_words = 10
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(vocablv)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
            print(u'Topic {}: {}'.format(i, ' '.join(topic_words)))

        print 'Parallel topics'
        topic_word = model_parallel1.topics  # model.components_ also works
        n_top_words = 10
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(vocablv)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
            print(u'Topic {}: {}'.format(i, ' '.join(topic_words)))

        print 'Parallel topics'
        topic_word = model_parallel2.topics  # model.components_ also works
        n_top_words = 10
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(vocablv)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
            print(u'Topic {}: {}'.format(i, ' '.join(topic_words)))

        print 'Parallel topics'
        topic_word = model_parallel3.topics  # model.components_ also works
        n_top_words = 10
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(vocablv)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
            print(u'Topic {}: {}'.format(i, ' '.join(topic_words)))

        print 'Serial topics'
        topic_word = lambda_serial  # model.components_ also works
        n_top_words = 10
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(vocablv)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
            print(u'Topic {}: {}'.format(i, ' '.join(topic_words)))

    # else:
    #     # Checking result

    #     plt.figure(1)
    #     # Cython
    #     plt.subplot(411)
    #     for i in xrange(num_topics):
    #         plt.plot(np.arange(N_words), model_parallel1.topics[i, :])
    #     plt.title('Opt')
    #     # Serial
    #     plt.subplot(412)
    #     for i in xrange(num_topics):
    #         plt.plot(np.arange(N_words), model_parallel2.topics[i, :])
    #     plt.title('Serial')
    #     plt.subplot(413)
    #     for i in xrange(num_topics):
    #         plt.plot(np.arange(N_words), model_parallel3.topics[i, :])
    #     plt.title('Parallel')
    #     plt.subplot(414)
    #     for i in xrange(num_topics):
    #         plt.plot(np.arange(N_words), model_parallel4.topics[i, :])
    #     plt.title('Parallel')
    #     plt.show()

    # raw_input()
    # for i in xrange(num_topics):
    #     plt.plot(np.arange(N_words), lambda_batch[i, :])
    #     plt.title('Batch')
    #     plt.show()
    #     raw_input()

    # raw_input()
    # for i in xrange(num_topics):
    #     plt.plot(np.arange(N_words), model.topic_word_[i, :])
    #     plt.title('Lda package')
    #     plt.show()
    #     raw_input()

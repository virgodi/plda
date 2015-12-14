import sys

import cgs_cython

sys.path.append('../util')
import matutil as mat

import numpy as np
import threading
from LDAutil import Evaluation


class cgsLDA:

    def __init__(self, num_topics, num_threads=4, iterations=500, damping=1,
                 sync_interval=1, alpha=None, beta=None, split_words=False):
        assert sync_interval <= iterations, "Cannot have sync_interval greater than iterations"
        self.topics = None
        self.num_topics = num_topics
        self.num_threads = num_threads
        self.iterations = iterations
        self.damping = damping
        self.sync_interval = sync_interval
        if alpha is None or alpha <= 0:
            self.alpha = 50./self.num_topics
        if beta is None or beta <= 0:
            self.beta = 0.1
        else:
            self.alpha = alpha
            self.beta = beta
        self.split_words = split_words

    def set_topics(self, n):
        self.num_topics = n

    def set_num_threads(self, t):
        self.num_threads = t

    def set_iterations(self, i):
        self.iterations = i

    def set_damping(self, d):
        self.damping = d

    def set_sync_interval(self, s):
        assert s <= self.iterations, "Cannot have sync_interval greater than iterations"
        self.sync_interval = s

    def set_alpha(self, a):
        self.alpha = a

    def set_beta(self, b):
        self.beta = b

    def set_split_words(self, sw):
        self.split_words = sw

    # baseline serial cython CGS
    def _sCGS(self, documents):
        # topic -> words distribution
        topics = np.zeros((self.num_topics, documents.shape[1]), dtype=np.float)
        # documents -> topic distribution
        gamma = np.zeros((documents.shape[0], self.num_topics), dtype=np.float)
        # sum of types per topic
        sum_K = np.zeros((self.num_topics), dtype=np.dtype("i"))
        # current topic for ith word in corpus
        curr_K = np.zeros((np.sum(documents)), dtype=np.dtype("i"))
        # sampling distributions
        sampling = np.zeros(
            (documents.shape[0], documents.shape[1], np.max(documents)), dtype=np.dtype("i"))
        # vectors for probability and topic counts
        p_K = np.zeros((self.num_topics), dtype=np.float)
        uniq_K = np.zeros((self.num_topics), dtype=np.dtype("i"))
        cgs_cython.CGS(documents, topics, gamma, sum_K, curr_K,
                       self.alpha, self.beta, self.iterations, sampling, p_K, uniq_K)
        self.topics = topics
        self.gamma = gamma
        self.sum_K = sum_K

    def fit(self, documents):
        '''
            Implement a parallel version of CGS over documents
            This requires storing a copy of sum_K and topics to avoid conflicts
            There's a synchronization period over which to reconcile the global
            sum_K/topics with the thread's local
            Additional feature is setting split_words = True
            This splits V into num_threads regions and provides locking mechanisms
            to update topics. This avoids the need to store the local topics and synchronize
            in case the corpus is very large, however the extra locking may hinder speedups
        '''        
        # topic -> words distribution
        topics = np.zeros((self.num_topics, documents.shape[1]), dtype=np.float)
        # documents -> topic distribution
        gamma = np.zeros((documents.shape[0], self.num_topics), dtype=np.float)
        # sum of types per topic
        sum_K = np.zeros((self.num_topics), dtype=np.dtype("i"))
        # sampling distributions
        sampling = np.zeros(
            (documents.shape[0], documents.shape[1], np.max(documents)), dtype=np.dtype("i"))

        self._pCGS(documents, topics, gamma, sum_K,
                  sampling)
        assert np.sum(sum_K) == np.sum(documents), "Sum_K not synced: {}, {}".format(
            np.sum(sum_K), np.sum(documents))
        assert np.sum(topics) == np.sum(documents), "topics not synced: {}, {}".format(
            np.sum(topics), np.sum(documents))

        self.topics = topics
        self.gamma = gamma
        self.sum_K = sum_K

        # Compute the perplexity of the trained model on the train data
        self.perplexity_train = Evaluation._log_likelihood(self, gamma, documents)

    def transform(self, documents):
        if self.topics is None:
            raise NameError('The model has not been trained yet')
            
        # create a new gamma that has additional rows for documents
        gamma = np.zeros(
            (documents.shape[0] + self.gamma.shape[0], self.gamma.shape[1]))
        gamma[documents.shape[0]:gamma.shape[0]] = self.gamma

        # create a copy of topics, sum_K, sampling to run CGS over
        topics = self.topics.copy()
        sum_K = self.sum_K.copy()
        sampling = np.zeros(
            (documents.shape[0], documents.shape[1], np.max(documents)), dtype=np.dtype("i"))
        self._pCGS(documents, topics, gamma, sum_K, sampling, False)

        # return the gamma over the first documents.shape[0] rows
        return gamma[0:documents.shape[0]]

    def _pCGS(self, documents, topics, gamma, sum_K, sampling, training = True):
        '''
            Function shared by the fit and transform functions to run parallelised CGS
        '''
        documents = documents.astype(np.dtype("i"))
        # vector of threads
        tList = [None]*self.num_threads
        # count array to synchronize
        copyCount = [0]
        # condition object to synchronize
        copyCondition = threading.Condition()

        wLocks = None
        if self.split_words:
            # array of locks over words
            wLocks = [None]*self.num_threads
            for i in range(self.num_threads):
                wLocks[i] = threading.Lock()
        for i in range(self.num_threads):
            tList[i] = threading.Thread(target=self._workerCGS, args=(
                wLocks, copyCondition, copyCount, i, documents, topics, gamma, sum_K, sampling, training))
            tList[i].start()
        for i in range(self.num_threads):
            tList[i].join()

    def _workerCGS(self, wLocks, copyCondition, copyCount, thread_num, documents, topics, gamma, sum_K, sampling, training):
        # vectors for probability and topic counts
        p_K = np.zeros((self.num_topics), dtype=np.float)
        uniq_K = np.zeros((self.num_topics), dtype=np.dtype("i"))

        # create a copy of sum_K and topics to work over
        t_sum_K = np.zeros(sum_K.shape, dtype=np.dtype("i"))
        t_topics = np.zeros(topics.shape, dtype=np.float) if wLocks is None else None

        # specify the boundaries of documents to work over
        d_interval = (documents.shape[0] - (documents.shape[0] % self.num_threads))/self.num_threads + \
            1 if documents.shape[
                0] % self.num_threads != 0 else documents.shape[0]/self.num_threads
        d_start = thread_num*d_interval
        d_end = min(documents.shape[0], d_start + d_interval)
        w_interval = (documents.shape[1] - (documents.shape[1] % self.num_threads))/self.num_threads + \
            1 if documents.shape[
                1] % self.num_threads != 0 else documents.shape[1]/self.num_threads
        # create a custom curr_K and that maps to the document boundaries
        curr_K = None
        if wLocks is None:
            curr_K = np.zeros(
                (np.sum(documents[d_start:d_end])), dtype=np.dtype("i"))
        else:
            max_region = 0
            for i in xrange(self.num_threads):
                region = np.sum(
                    documents[d_start:d_end, i*w_interval:(i+1)*w_interval])
                if region > max_region:
                    max_region = region
            curr_K = np.zeros((self.num_threads, max_region), dtype=np.dtype("i"))

        # initialize topics for each thread
        if wLocks is None:
            cgs_cython.init_topics(
                documents, t_topics, gamma, t_sum_K, curr_K, d_start, d_end, 0, documents.shape[1], training)
        else:
            for i in xrange(self.num_threads):
                word_group = (i+thread_num) % self.num_threads
                with wLocks[word_group]:
                    w_start = (word_group)*w_interval
                    w_end = min(documents.shape[1], w_start + w_interval)
                    cgs_cython.init_topics(
                        documents, topics, gamma, t_sum_K, curr_K[i], d_start, d_end, w_start, w_end, training)

        # have sum_K and topics be the sum of all thread-specific
        # t_sum_K's/t_topics's
        with copyCondition:
            mat.add1d(sum_K, t_sum_K)
            if wLocks is None:
                mat.add2d(topics, t_topics)
            self.copyConditionCheck(copyCount, copyCondition)

        # have t_sum_K/t_topics be a copy of the summed sum_K/topics
        mat.copy1d(sum_K, t_sum_K)
        if wLocks is None:
            mat.copy2d(topics, t_topics)
        # start the gibb sampling iterations
        for i in xrange(self.iterations/self.sync_interval):
            if wLocks is None:
                cgs_cython.CGS_iter(documents, t_topics, gamma, t_sum_K, curr_K, self.alpha, self.beta,
                                    sampling, p_K, uniq_K, d_start, d_end, 0, documents.shape[1], self.sync_interval, training)
            else:
                # work through each group of words
                for j in xrange(self.num_threads):
                    word_group = (j+thread_num) % self.num_threads
                    with wLocks[word_group]:
                        w_start = (word_group)*w_interval
                        w_end = min(
                            documents.shape[1], (word_group+1)*w_interval)
                        cgs_cython.CGS_iter(documents, topics, gamma, t_sum_K, curr_K[
                                            j], self.alpha, self.beta, sampling, p_K, uniq_K, d_start, d_end, w_start, w_end, self.sync_interval, training)

            # must synchronize sum_K and topics
            # this subtraction can be done in parallel as originals unmodified
            # and then wait for every thread to do that
            mat.subtract1d(t_sum_K, sum_K)
            if wLocks is None:
                mat.subtract2d(t_topics, topics)

            with copyCondition:
                self.copyConditionCheck(copyCount, copyCondition)

            # one at a time update sum_K and topics
            with copyCondition:
                mat.add1d(sum_K, t_sum_K)
                if wLocks is None:
                    mat.add2d(topics, t_topics)
                self.copyConditionCheck(copyCount, copyCondition)

            # at this point need to wait for all threads to update sum_K with their changes
            # once all threads reach this point it's safe to copy sum_K to
            # t_sum_K
            mat.copy1d(sum_K, t_sum_K)
            if wLocks is None:
                mat.copy2d(topics, t_topics)

    def copyConditionCheck(self, copyCount, copyCondition):
        copyCount[0] += 1
        if copyCount[0] == self.num_threads:
            copyCount[0] = 0
            copyCondition.notifyAll()
        else:
            copyCondition.wait()

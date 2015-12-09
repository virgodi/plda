import sys
import ovi_cython
import numpy as np
import threading
from LDAutil import Evaluation

class oviLDA:

    def __init__(self, num_topics, num_threads=4, batch_size=20, tau=512, kappa=0.7):
        self.num_topics = num_topics
        self.num_threads = num_threads
        self.batch_size = batch_size
        self.tau = tau
        self.kappa = kappa
        self.topics = None
        self.gamma = None
        self.perplexity_train = None

    def set_topics(self, n):
        self.num_topics = n

    def set_threads(self, t):
        self.num_threads = t

    def set_batch_size(self, b):
        self.batch_size = b

    def set_tau(self, tau):
        self.tau = tau

    def set_kappa(self, k):
        self.kappa = k

    def fit(self, dtm):
        '''
        Parallel version of the lda: the temporary topics are computed in
        parallel for each document inside a mini-batch

        '''
        # Initialisation
        dtm = dtm.astype(int)
        num_docs, num_words = dtm.shape
        topics = np.random.gamma(100., 1./100., (self.num_topics, num_words))
        gamma = np.ones((num_docs, self.num_topics))
        ExpELogBeta = np.zeros((self.num_topics, num_words))
        topics_int = np.zeros((self.num_threads, self.num_topics, num_words))

        num_batch = num_docs / self.batch_size
        batches = np.array_split(
            np.arange(num_docs, dtype=np.int32), num_batch)

        for it_batch in range(num_batch):
            ovi_cython.exp_digamma2d(topics, ExpELogBeta)

            docs_thread = np.array_split(batches[it_batch], self.num_threads)

            # vector of threads
            threads = [None]*self.num_threads

            for tid in range(self.num_threads):
                threads[tid] = threading.Thread(target=self._worker_estep,
                                                args=(docs_thread[tid], dtm,
                                                      topics_int[tid, :, :],
                                                      gamma, ExpELogBeta))
                threads[tid].start()

            for thread in threads:
                thread.join()

            # Synchronizing the topics_int
            topics_int_tot = np.sum(topics_int, axis=0)
            # Initialize the list of topics int for the next batch
            topics_int[:, :, :] = 0
            # M-step
            indices = (np.sum(dtm[batches[it_batch], :], axis=0) > 0).astype(
                np.int32)
            ovi_cython.m_step(topics, topics_int_tot, indices, num_docs,
                                 self.batch_size, self.tau, self.kappa, it_batch)

        self.topics = topics
        self.gamma = gamma

        # Compute the perplexity of the trained model on the train data
        self.perplexity_train = Evaluation._log_likelihood(self, gamma, dtm)


    def transform(self, dtm):
        '''
        Transform dtm into gamma according to the previously trained model.

        '''
        if self.topics is None:
            raise NameError('The model has not been trained yet')
        # Initialisation
        num_docs, num_words = dtm.shape
        np.random.seed(0)
        gamma = np.ones((num_docs, self.num_topics))
        ExpELogBeta = np.zeros((self.num_topics, num_words))
        topics_int = np.zeros((self.num_threads, self.num_topics, num_words))

        num_batch = num_docs / self.batch_size
        batches = np.array_split(
            np.arange(num_docs, dtype=np.int32), num_batch)

        for it_batch in range(num_batch):
            ovi_cython.exp_digamma2d(self.topics, ExpELogBeta)

            docs_thread = np.array_split(batches[it_batch], self.num_threads)

            # vector of threads
            threads = [None]*self.num_threads

            for tid in range(self.num_threads):
                threads[tid] = threading.Thread(target=self._worker_estep,
                                                args=(docs_thread[tid], dtm,
                                                      topics_int[tid, :, :],
                                                      gamma, ExpELogBeta))
                threads[tid].start()

            for thread in threads:
                thread.join()

        return gamma

    

    def _worker_estep(self, docs, dtm, topics_int_t, gamma, ExpELogBeta):
        # Local initialization
        num_words = dtm.shape[1]
        ExpLogTethad = np.zeros(self.num_topics)
        phi = np.zeros((self.num_topics, num_words))

        # Lambda_int is shared among the threads
        ovi_cython.e_step(docs, dtm, gamma, ExpELogBeta, ExpLogTethad, topics_int_t, phi,
                             self.num_topics)

import sys
import ovi_cython
import numpy as np
import threading


class oviLDA:

    def __init__(self, num_topics, num_threads=1):
        self.num_topics = num_topics
        self.num_threads = num_threads
        self.topics = None
        self.gamma = None
        self.perplexity_train = None

    def set_topics(self, n):
        self.num_topics = n

    def set_threads(self, t):
        self.num_threads = t

    def fit(self, dtm, batch_size, tau=512, kappa=0.7):
        '''
        Parallel version of the lda: the temporary topics are computed in
        parallel for each document inside a mini-batch

        '''
        # Initialisation
        num_docs, num_words = dtm.shape
        topics = np.random.gamma(100., 1./100., (self.num_topics, num_words))
        gamma = np.ones((num_docs, self.num_topics))
        ExpELogBeta = np.zeros((self.num_topics, num_words))
        topics_int = np.zeros((self.num_threads, self.num_topics, num_words))

        num_batch = num_docs / batch_size
        batches = np.array_split(
            np.arange(num_docs, dtype=np.int32), num_batch)

        for it_batch in range(num_batch):
            ovi_cython.exp_digamma2d(topics, ExpELogBeta)

            docs_thread = np.array_split(batches[it_batch], self.num_threads)

            # vector of threads
            threads = [None]*self.num_threads

            for tid in range(self.num_threads):
                threads[tid] = threading.Thread(target=self.worker_estep,
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
                                 batch_size, tau, kappa, it_batch)

        # Compute the perplexity of the trained model on the train data
        self.perplexity_train = self._log_likelihood(topics, gamma, dtm)

        self.topics = topics
        self.gamma = gamma

    def transform(self, dtm, batch_size, tau=512, kappa=0.7):
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

        num_batch = num_docs / batch_size
        batches = np.array_split(
            np.arange(num_docs, dtype=np.int32), num_batch)

        for it_batch in range(num_batch):
            ovi_cython.exp_digamma2d(self.topics, ExpELogBeta)

            docs_thread = np.array_split(batches[it_batch], self.num_threads)

            # vector of threads
            threads = [None]*self.num_threads

            for tid in range(self.num_threads):
                threads[tid] = threading.Thread(target=self.worker_estep,
                                                args=(docs_thread[tid], dtm,
                                                      topics_int[tid, :, :],
                                                      gamma, ExpELogBeta))
                threads[tid].start()

            for thread in threads:
                thread.join()

        return gamma

    def perplexity(self, dtm_test, batch_size, tau=512, kappa=0.7):
        '''
        Compute the log-likelihood of the documents in dtm_test based on the
        topic distribution already learned by the model
        '''
        gamma = self.transform(dtm_test, tau, kappa)
        return self._log_likelihood(self.topics, self.gamma, dtm_test)

    def _log_likelihood(self, topics, gamma, dtm):
        '''
        Compute the log-likelihood given the two distributions gamma and topics.
        '''
        # Normalizing the topics and gamma
        topics = topics/topics.sum(axis=1)[:, np.newaxis]
        gamma = gamma/gamma.sum(axis=1)[:, np.newaxis]

        if len(gamma.shape) == 1:
            doc_idx = np.nonzero(dtm)[0]
            doc_cts = dtm[doc_idx]
            return np.sum(np.log(np.dot(gamma[i, :],
                          topics[:, doc_idx]))*doc_cts)
        else:
            # Initialization
            num = 0
            denom = 0
            for i in range(gamma.shape[0]):
                doc_idx = np.nonzero(dtm[i, :])[0]
                doc_cts = dtm[i, doc_idx]
                num += np.sum(np.log(np.dot(gamma[i, :],
                              topics[:, doc_idx]))*doc_cts)
                denom += np.sum(doc_cts)
        return num/denom

    def print_topic(self, vocabulary, num_top_words=10):
        if self.topics is None:
            raise NameError('The model has not been trained yet')
        topic_word = self.topics
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(
                vocabulary)[np.argsort(topic_dist)][:-(num_top_words+1):-1]
            print(u'Topic {}: {}'.format(i, ' '.join(topic_words)))

    def worker_estep(self, docs, dtm, topics_int_t, gamma, ExpELogBeta):
        # Local initialization
        num_words = dtm.shape[1]
        ExpLogTethad = np.zeros(self.num_topics)
        phi = np.zeros((self.num_topics, num_words))

        # Lambda_int is shared among the threads
        ovi_cython.e_step(docs, dtm, gamma, ExpELogBeta, ExpLogTethad, topics_int_t, phi,
                             self.num_topics)

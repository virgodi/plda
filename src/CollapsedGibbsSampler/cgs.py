import sys

import cgs_cython

sys.path.append('../util')
import matutil as mat

import numpy as np
import threading

class cgsLDA:
    def __init__(self, num_topics, iterations = 500, damping = 1, sync_interval = 1):
        assert sync_interval <= iterations, "Cannot have sync_interval greater than iterations"
        self.num_topics = num_topics
        self.iterations = iterations
        self.damping = damping
        self.sync_interval = sync_interval
        
    def set_topics(self, n):
        self.num_topics = n
        
    def set_iterations(self, i):
        self.iterations = i
        
    def set_damping(self, d):
        self.damping = d
        
    def set_sync_interval(self, s):
        assert s <= self.iterations, "Cannot have sync_interval greater than iterations"
        self.sync_interval = s
        
    #baseline serial cython CGS
    def sCGS(self, documents, alpha=None, beta=None):
        if alpha is None or alpha <= 0:
            alpha = 50./self.num_topics
        if beta is None or beta <= 0:
            beta = 0.1
        
        # topic -> words distribution
        K_V = np.zeros((self.num_topics,documents.shape[1]), dtype=np.float)
        # documents -> topic distribution
        D_K = np.zeros((documents.shape[0], self.num_topics), dtype=np.float)
        # sum of types per topic
        sum_K = np.zeros((self.num_topics), dtype=np.dtype("i"))
        # current topic for ith word in corpus
        curr_K = np.zeros((np.sum(documents)), dtype=np.dtype("i"))
        # sampling distributions
        sampling = np.zeros((documents.shape[0], documents.shape[1], np.max(documents)), dtype=np.dtype("i"))
        # vectors for probability and topic counts
        p_K = np.zeros((self.num_topics), dtype=np.float)
        uniq_K = np.zeros((self.num_topics), dtype=np.dtype("i"))
        cgs_cython.CGS(documents, K_V, D_K, sum_K, curr_K, alpha, beta, self.iterations, sampling, p_K, uniq_K)
        self.K_V = K_V
        self.D_K = D_K
        
        
    '''
        Implement a parallel version of CGS over documents
        This requires storing a copy of sum_K and K_V to avoid conflicts
        There's a synchronization period over which to reconcile the global
        sum_K/K_V with the thread's local
        Additional feature is setting split_words = True
        This splits V into num_threads regions and provides locking mechanisms
        to update K_V. This avoids the need to store the local K_V and synchronize
        in case the corpus is very large, however the extra locking may hinder speedups
    '''
    def fit(self, documents, num_threads = 4, alpha=None, beta=None, split_words = False):
        if alpha is None or alpha <= 0:
            alpha = 50./self.num_topics
        if beta is None or beta <= 0:
            beta = 0.1
            
        # topic -> words distribution
        K_V = np.zeros((self.num_topics,documents.shape[1]), dtype=np.float)
        # documents -> topic distribution
        D_K = np.zeros((documents.shape[0], self.num_topics), dtype=np.float)
        # sum of types per topic
        sum_K = np.zeros((self.num_topics), dtype=np.dtype("i"))
        # sampling distributions
        sampling = np.zeros((documents.shape[0], documents.shape[1], np.max(documents)), dtype=np.dtype("i"))
        
        self.pCGS(documents, self.iterations, K_V, D_K, sum_K, sampling, num_threads, alpha, beta, split_words)
        assert np.sum(sum_K) == np.sum(documents), "Sum_K not synced: {}, {}".format(np.sum(sum_K), np.sum(documents))
        assert np.sum(K_V) == np.sum(documents), "K_V not synced: {}, {}".format(np.sum(K_V), np.sum(documents))
        
        self.K_V = K_V
        self.D_K = D_K
        self.sum_K = sum_K
        self.alpha = alpha
        self.beta = beta
        
    def inference(self, documents, iterations = 500, num_threads = 4, split_words = False):
        #create a new D_K that has additional rows for documents
        D_K = np.zeros((documents.shape[0] + self.D_K.shape[0], self.D_K.shape[1]))
        D_K[documents.shape[0]:D_K.shape[0]] = self.D_K
        
        #create a copy of K_V, sum_K, sampling to run CGS over
        K_V = self.K_V.copy()
        sum_K = self.sum_K.copy()
        sampling = np.zeros((documents.shape[0], documents.shape[1], np.max(documents)), dtype=np.dtype("i"))
        self.pCGS(documents, iterations, K_V, D_K, sum_K, sampling, num_threads, self.alpha, self.beta, split_words)
        
        #return the D_K over the first documents.shape[0] rows
        return D_K[0:documents.shape[0]]
        
    '''
        This is a function that's shared by the fit and inference functions to run parallelised CGS
    '''
    def pCGS(self, documents, iterations, K_V, D_K, sum_K, sampling, num_threads, alpha, beta, split_words):
        #vector of threads
        tList = [None]*num_threads
        #count array to synchronize
        copyCount = [0]
        #condition object to synchronize
        copyCondition = threading.Condition()
        
        wLocks = None
        if split_words:
            #array of locks over words
            wLocks = [None]*num_threads
            for i in range(num_threads):
                wLocks[i] = threading.Lock()
        for i in range(num_threads):
            tList[i] = threading.Thread(target=self.workerCGS, args=(iterations, wLocks, copyCondition, copyCount, i, num_threads, documents, K_V, D_K, sum_K, alpha, beta, sampling))
            tList[i].start()
        for i in range(num_threads):
            tList[i].join()
            
    def workerCGS(self, iterations, wLocks, copyCondition, copyCount, thread_num, num_threads, documents, K_V, D_K, sum_K, alpha, beta, sampling):
        # vectors for probability and topic counts
        p_K = np.zeros((self.num_topics), dtype=np.float)
        uniq_K = np.zeros((self.num_topics), dtype=np.dtype("i"))
        
        #create a copy of sum_K and K_V to work over
        t_sum_K = np.zeros(sum_K.shape, dtype=np.dtype("i"))
        t_K_V = np.zeros(K_V.shape, dtype=np.float) if wLocks is None else None
        
        #specify the boundaries of documents to work over
        d_interval = (documents.shape[0] - (documents.shape[0]%num_threads))/num_threads + 1 if documents.shape[0]%num_threads != 0 else documents.shape[0]/num_threads
        d_start = thread_num*d_interval
        d_end = min(documents.shape[0], d_start + d_interval)
        w_interval = (documents.shape[1] - (documents.shape[1]%num_threads))/num_threads + 1 if documents.shape[1]%num_threads != 0 else documents.shape[1]/num_threads
        #create a custom curr_K and that maps to the document boundaries
        curr_K = None
        if wLocks is None:
            curr_K = np.zeros((np.sum(documents[d_start:d_end])), dtype=np.dtype("i"))
        else:
            max_region = 0
            for i in xrange(num_threads):
                region = np.sum(documents[d_start:d_end, i*w_interval:(i+1)*w_interval])
                if region > max_region:
                    max_region = region
            curr_K = np.zeros((num_threads, max_region), dtype=np.dtype("i"))
            
        #initialize topics for each thread
        if wLocks is None:
            cgs_cython.init_topics(documents, t_K_V, D_K, t_sum_K, curr_K, d_start, d_end, 0, documents.shape[1])
        else:
            for i in xrange(num_threads):
                word_group = (i+thread_num)%num_threads
                with wLocks[word_group]:
                    w_start = (word_group)*w_interval
                    w_end = min(documents.shape[1], w_start + w_interval)
                    cgs_cython.init_topics(documents, K_V, D_K, t_sum_K, curr_K[i], d_start, d_end, w_start, w_end)
                
        #have sum_K and K_V be the sum of all thread-specific t_sum_K's/t_K_V's
        with copyCondition:
            mat.add1d(sum_K, t_sum_K)
            if wLocks is None: mat.add2d(K_V, t_K_V)
            self.copyConditionCheck(copyCount, num_threads, copyCondition) 
        
        #have t_sum_K/t_K_V be a copy of the summed sum_K/K_V
        mat.copy1d(sum_K, t_sum_K)
        if wLocks is None: mat.copy2d(K_V, t_K_V)
        #start the gibb sampling iterations
        for i in xrange(iterations/self.sync_interval):
            if wLocks is None:
                cgs_cython.CGS_iter(documents, t_K_V, D_K, t_sum_K, curr_K, alpha, beta, sampling, p_K, uniq_K, d_start, d_end, 0, documents.shape[1], self.sync_interval)
            else:
                #work through each group of words
                for j in xrange(num_threads):
                    word_group = (j+thread_num)%num_threads
                    with wLocks[word_group]:
                        w_start = (word_group)*w_interval
                        w_end = min(documents.shape[1], (word_group+1)*w_interval)
                        cgs_cython.CGS_iter(documents, K_V, D_K, t_sum_K, curr_K[j], alpha, beta, sampling, p_K, uniq_K, d_start, d_end, w_start, w_end, self.sync_interval)
            
            #must synchronize sum_K and K_V              
            #this subtraction can be done in parallel as originals unmodified and then wait for every thread to do that
            mat.subtract1d(t_sum_K, sum_K)
            if wLocks is None: mat.subtract2d(t_K_V, K_V)
          
            with copyCondition:
                self.copyConditionCheck(copyCount, num_threads, copyCondition)
                
            #one at a time update sum_K and K_V
            with copyCondition:
                mat.add1d(sum_K, t_sum_K)
                if wLocks is None: mat.add2d(K_V, t_K_V)
                self.copyConditionCheck(copyCount, num_threads, copyCondition)
                
            #at this point need to wait for all threads to update sum_K with their changes
            #once all threads reach this point it's safe to copy sum_K to t_sum_K
            mat.copy1d(sum_K, t_sum_K)
            if wLocks is None: mat.copy2d(K_V, t_K_V)
         
    def copyConditionCheck(self, copyCount, num_threads, copyCondition):
        copyCount[0] +=1
        if copyCount[0] == num_threads:
            copyCount[0] = 0
            copyCondition.notifyAll()
        else:
            copyCondition.wait()
            
    
        
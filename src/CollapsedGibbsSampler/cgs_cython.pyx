import numpy as np
cimport numpy as np
cimport cython
from cython.operator cimport preincrement as inc, predecrement as dec
from libc.stdlib cimport rand, RAND_MAX

'''
Initial topic sampling for Gibbs sampling
'''
def init_topics(int[:, ::1] documents, double[:, ::1] K_V, double[:, ::1] D_K, int[::1] sum_K, int[::1] curr_K, int d_start, int d_end, int w_start, int w_end):
    with nogil:
        _init_topics(documents, K_V, D_K, sum_K, curr_K, d_start, d_end, w_start, w_end)

@cython.boundscheck(False)
cdef inline void _init_topics(int[:, ::1] documents, double[:, ::1] K_V, double[:, ::1] D_K, int[::1] sum_K, int[::1] curr_K, int d_start, int d_end, int w_start, int w_end) nogil:
    cdef size_t i,j,k
    # number of topics
    cdef int K = K_V.shape[0]
    cdef int topic = 0
    cdef int count = 0
    
    for i in xrange(d_start, d_end):
        for j in xrange(w_start, w_end):
            for k in xrange(documents[i,j]):
                topic = count % K
                curr_K[count] = topic
                inc(sum_K[topic])
                inc(K_V[topic, j])
                inc(D_K[i, topic])
                inc(count)

'''
CGS with dynamic sampling with specified documents/words and iterations
'''
def CGS_iter(int[:, ::1] documents, double[:, ::1] K_V, double[:, ::1] D_K, int[::1] sum_K, int[::1] curr_K, double alpha, double beta, int[:, :, ::1] sampling, double[::1] p_K, int[::1] uniq_K, int d_start, int d_end, int w_start, int w_end, int iterations):
    with nogil:
        _CGS_iter(documents, K_V, D_K, sum_K, curr_K, alpha, beta, sampling, p_K, uniq_K, d_start, d_end, w_start, w_end, iterations)

@cython.boundscheck(False)
cdef inline void _CGS_iter(int[:, ::1] documents, double[:, ::1] K_V, double[:, ::1] D_K, int[::1] sum_K, int[::1] curr_K, double alpha, double beta, int[:, :, ::1] sampling, double[::1] p_K, int[::1] uniq_K, int d_start, int d_end, int w_start, int w_end, int iterations) nogil:
    cdef size_t i,j,k,p,it
    # number of types
    cdef int V = documents.shape[1]
    # number of topics
    cdef int K = K_V.shape[0]
    cdef double cumulative_p, randnum
    cdef int num_samples, topic, count
    
    for it in xrange(iterations):
        count = 0
        for i in xrange(d_start, d_end):
            for j in xrange(w_start, w_end):
                #check if document has type j, if so go to sampling procedure
                if documents[i,j] > 0:
                    num_samples = documents[i,j]
                    if num_samples > 2:
                        #more than 2 occurrences, so do dynamic sampling procedure
                        cumulative_p = 0
                        for p in xrange(num_samples):
                            cumulative_p += sampling[i,j,p]
                        #choose random number between 0 and sum of samples
                        #if cumulative_p is 0, must initialize the sampling vector
                        if cumulative_p == 0:
                            sampling[i,j,num_samples-1] = 1
                            cumulative_p = 1
                        randnum = rand()/float(RAND_MAX)*cumulative_p
                        num_samples = -1
                        cumulative_p = 0
                        while cumulative_p < randnum and num_samples < documents[i,j]:
                            cumulative_p += sampling[i,j,inc(num_samples)]
                    #only looping through num_samples times, reset uniq_k
                    for k in xrange(K):
                        uniq_K[k] = 0
                    for k in xrange(num_samples):
                        #retrieve current topic and decrement
                        topic = curr_K[count]
                        dec(sum_K[topic])
                        dec(K_V[topic,j])
                        dec(D_K[i,topic])
                        
                        cumulative_p = 0
                        for p in xrange(K):
                            cumulative_p += (D_K[i, p] + alpha)*(K_V[p,j] + beta)/(sum_K[p] + V*beta)
                            p_K[p] = cumulative_p
                        randnum = rand()/float(RAND_MAX)*cumulative_p
                        topic = get_index(p_K, randnum, K)
                        
                        #assign new topic
                        curr_K[count] = topic
                        inc(sum_K[topic])
                        inc(K_V[topic,j])
                        inc(D_K[i, topic])
                        inc(count)
                        
                        #increment index in uniq_k
                        inc(uniq_K[topic])
                    #make sure to keep count consistent with documents table
                    count += documents[i,j] - num_samples
                    #find number of unique topics used and increment index
                    num_samples = 0
                    for k in xrange(K):
                        if uniq_K[k] != 0:
                            inc(num_samples)
                    inc(sampling[i,j,num_samples])

'''
Initial Serial Collapsed Gibbs Sampling
- Dynamic sampling (http://jmlr.csail.mit.edu/proceedings/papers/v13/xiao10a/xiao10a.pdf)
'''
@cython.boundscheck(False)
cpdef void CGS(int[:, ::1] documents, double[:, ::1] K_V, double[:, ::1] D_K, int[::1] sum_K, int[::1] curr_K, double alpha, double beta, unsigned int n, int[:, :, ::1] sampling, double[::1] p_K, int[::1] uniq_K) nogil:
    cdef size_t i, j, k, it, p
    # number of documents
    cdef int D = documents.shape[0]
    # number of types
    cdef int V = documents.shape[1]
    # number of topics
    cdef int K = K_V.shape[0]
    cdef double cumulative_p
    cdef double randnum
    cdef int num_samples, count, topic    
        
    #initialize random topics to start
    count = 0
    for i in xrange(documents.shape[0]):
        for j in xrange(documents.shape[1]):
            for k in xrange(documents[i,j]):
                topic = count % K
                curr_K[count] = topic
                inc(sum_K[topic])
                inc(K_V[topic, j])
                inc(D_K[i, topic])
                inc(count) 
    
    #proceed to Gibbs sampling
    for it in range(n):
        count = 0
        for i in xrange(D):
            for j in xrange(V):
                #check if document has type j, if so go to sampling procedure
                if documents[i,j] > 0:
                    num_samples = documents[i,j]
                    if num_samples > 2:
                        #more than 2 occurrences, so do dynamic sampling procedure
                        cumulative_p = 0
                        for p in xrange(num_samples):
                            cumulative_p += sampling[i,j,p]
                        #choose random number between 0 and sum of samples
                        #if cumulative_p is 0, must initialize the sampling vector
                        if cumulative_p == 0:
                            sampling[i,j,num_samples-1] = 1
                            cumulative_p = 1
                        randnum = rand()/float(RAND_MAX)*cumulative_p
                        num_samples = -1
                        cumulative_p = 0
                        while cumulative_p < randnum and num_samples < documents[i,j]:
                            cumulative_p += sampling[i,j,inc(num_samples)]
                    #only looping through num_samples times, reset uniq_k
                    for k in xrange(K):
                        uniq_K[k] = 0
                    for k in xrange(num_samples):
                        #retrieve current topic and decrement
                        topic = curr_K[count]
                        dec(sum_K[topic])
                        dec(K_V[topic,j])
                        dec(D_K[i,topic])
                        
                        cumulative_p = 0
                        for p in xrange(K):
                            cumulative_p += (D_K[i, p] + alpha)*(K_V[p,j] + beta)/(sum_K[p] + V*beta)
                            p_K[p] = cumulative_p
                        randnum = rand()/float(RAND_MAX)*cumulative_p
                        topic = get_index(p_K, randnum, K)
                        
                        #assign new topic
                        curr_K[count] = topic
                        inc(sum_K[topic])
                        inc(K_V[topic,j])
                        inc(D_K[i, topic])
                        inc(count)
                        
                        #increment index in uniq_k
                        inc(uniq_K[topic])
                    #make sure to keep count consistent with documents table
                    count += documents[i,j] - num_samples
                    #find number of unique topics used and increment index
                    num_samples = 0
                    for k in xrange(K):
                        if uniq_K[k] != 0:
                            inc(num_samples)
                    inc(sampling[i,j,num_samples])                    
    
    #normalize K_V and D_K
    for i in xrange(K):
        cumulative_p = 0
        for j in xrange(V):
            K_V[i,j] += beta
            cumulative_p += K_V[i,j]
        for j in xrange(V):
            K_V[i,j] /= cumulative_p
    for i in xrange(D):
        cumulative_p = 0
        for j in xrange(K):
            D_K[i,j] += alpha
            cumulative_p += D_K[i,j]
        for j in xrange(K):
            D_K[i,j] /= cumulative_p  
        
    
@cython.boundscheck(False)
cdef int get_index(double[::1] p_K, double randnum, int K) nogil:
    cdef int mink = 0
    cdef int maxk = K-1
    cdef int midk
    while mink < maxk:
        midk = mink + (maxk-mink)/2
        if p_K[midk] < randnum:
            mink = midk + 1
        else:
            maxk = midk
    return maxk
    
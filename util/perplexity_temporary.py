import numpy as np

'''
These are tempory perplexity function until I create the class and uniform everything
To make it work you need to have an inference function. It should output a tuple (topics, assignments)
'''

def perplexity_online(lda,newdocs,tau,kappa):

    new = inference(lda,newdocs,tau,kappa)
    
    topics = new[0]
    gammas = new[1]

    topics = topics/topics.sum(axis=1)[:, np.newaxis]

    gammas = gammas/gammas.sum(axis=1)[:,np.newaxis]

    num = 0
    denom = 0
    
    for i in range(gammas.shape[0]):
        doc_idx = np.nonzero(newdocs[i,:])[0]
        doc_cts = newdocs[i,doc_idx]
        num += np.sum(np.log(np.dot(gammas[i,:],topics[:,doc_idx]))*doc_cts)
        denom += np.sum(doc_cts)

    return num/denom

def perplexity_online(lda,newdocs:

    new = inference(lda,newdocs)
    
    topics = new[0]
    gammas = new[1]

    topics = topics/topics.sum(axis=1)[:, np.newaxis]

    gammas = gammas/gammas.sum(axis=1)[:,np.newaxis]

    num = 0
    denom = 0
    
    for i in range(gammas.shape[0]):
        doc_idx = np.nonzero(newdocs[i,:])[0]
        doc_cts = newdocs[i,doc_idx]
        num += np.sum(np.log(np.dot(gammas[i,:],topics[:,doc_idx]))*doc_cts)
        denom += np.sum(doc_cts)

    return num/denom
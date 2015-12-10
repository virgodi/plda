import numpy as np


def perplexity(model, dtm_test):
    '''
    Compute the log-likelihood of the documents in dtm_test based on the
    topic distribution already learned by the model
    '''
    gamma = model.transform(dtm_test)
    return _log_likelihood(model, gamma, dtm_test)

def _log_likelihood(model, gamma, dtm):
    '''
    Compute the log-likelihood given the two distributions gamma and topics.
    '''
    # Normalizing the topics and gamma
    topics = model.topics
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
            num += np.sum(np.log(np.dot(gamma[i, :],topics[:, doc_idx]))*doc_cts)
            denom += np.sum(doc_cts)
    return num/denom

def print_topic(model, vocabulary, num_top_words=10):
    '''
    Printing topics
    '''
    if model.topics is None:
        raise NameError('The model has not been trained yet')
    topic_word = model.topics
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(
            vocabulary)[np.argsort(topic_dist)][:-(num_top_words+1):-1]
        print(u'Topic {}: {}'.format(i, ' '.join(topic_words)))
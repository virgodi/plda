import numpy as np
import matplotlib.pyplot as plt
from scipy.special import psi


def digamma(data):
    if (len(data.shape) == 1):
        return psi(data) - psi(np.sum(data))
    return psi(data) - psi(np.sum(data, axis=1))[:, np.newaxis]


def lda_batch(lambda_, dtm, ntopic, batch_size=10, tau=512, kappa=0.7):
    '''
    Online variational inference lda with mini-batch
    '''
    ndoc, nvoc = dtm.shape
    nu = 1./ntopic
    alpha = 1./ntopic

    # Trick to have iso initialization
    # lambda_ = np.random.gamma(100., 1./100., (ntopic, nvoc))
    gamma = np.random.gamma(100., 1./100., (ndoc, ntopic))
    print lambda_[0, 0]

    numbatch = ndoc / batch_size
    batches = np.array_split(range(ndoc), numbatch)

    for it_batch in range(numbatch):
        ELogBeta = digamma(lambda_)
        ExpELogBeta = np.exp(ELogBeta)

        lambda_int = np.zeros(lambda_.shape)

        for d in batches[it_batch]:
            # print d
            ids = np.nonzero(dtm[d, :])[0]
            counts = dtm[d, ids]
            ExpELogBetad = ExpELogBeta[:, ids]

            gammad = gamma[d, :]

            # print gammad

            for inner_it in range(1000):

                oldgammad = gammad

                ElogTethad = digamma(gammad)
                ExpLogTethad = np.exp(ElogTethad)
                phi = ExpELogBetad * ExpLogTethad[:, np.newaxis]
                phi = phi / (phi.sum(axis=0)+0.00001)

                gammad = alpha + np.dot(phi, counts)

                # print gammad

                if np.mean((gammad-oldgammad)**2) < 0.0000001:
                    break

            # print inner_it
            gamma[d, :] = gammad

            lambda_int[:, ids] += counts[np.newaxis, :] * phi

        indices = np.unique(np.nonzero(dtm[batches[it_batch], :])[1])

        rt = (tau + it_batch)**(- kappa)

        lambda_[:, indices] = (1 - rt) * lambda_[:, indices] + rt * \
            ndoc * (nu + lambda_int[:, indices]) / len(batches[it_batch])

    return lambda_, gamma

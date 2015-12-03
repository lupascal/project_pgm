
"""
obtain a tight lower bound on the log-likelihood of an LDA model for a 
document, as described in 
Blei, Ng, and Jordan. Latent Dirichlet Allocation (2003).
two variational parameters, one for a multinomial distribution, and one
for a dirichlet distribution, are optimized with respect to the log-likelihood
"""

import numpy as np
from scipy.special import psi



def variational_inference(word_incidences, dirich_param, word_prob_given_topic):
    """
    obtain a tight lower bound on the log-likelihood of an LDA model for a 
    document, as described in 
    Blei, Ng, and Jordan. Latent Dirichlet Allocation (2003).
    two variational parameters, one for a multinomial distribution, and one
    for a dirichlet distribution, are optimized with respect to the 
    log-likelihood.
    arguments:
    word_incidences: an array containing the number of times each word in the
    vocabulary appears in the document.
    dirich_param: an estimate of the parameter of the dirichlet distribution
    which generates the parameter for the (multinomial) probability 
    distribution over topics in the document.
    word_prob_given_topic: an array of size (nb topics, vocabulary size) which
    gives the (estimated) probability that a given topic will generate a 
    certain word.
    returns:
    the variational parameter for the dirichlet distribution,
    the variational parameter for the multinomial distribution
    """

    [nb_topics, voc_size] = np.shape(word_prob_given_topic)

    var_dirich = dirich_param + np.sum(word_incidences) / nb_topics
    var_multinom = np.ones([voc_size, nb_topics]) / nb_topics

    stop = var_inf_stop()

    while(not stop(var_dirich)):

        var_multinom = np.transpose(word_prob_given_topic) \
                       * np.exp(psi(var_dirich))
            
        var_multinom = (var_multinom 
                        / np.sum(var_multinom, axis = 1)[:, np.newaxis])

        var_dirich = dirich_param + np.sum(word_incidences * 
                                           np.transpose(var_multinom), axis = 1)

    return var_dirich, var_multinom
                                                    



class var_inf_stop(object):
    """stopping criterion for variational inference"""

    def __init__(self, threshold = 1e-3, max_iter = None):

        self.current_estimate_ = np.empty(0)
        self.threshold_ = threshold
        self.max_iter_ = max_iter
        if(max_iter):
            self.iter_ = 0

    def __call__(self, new_estimate):

        if(not np.size(self.current_estimate_)):
            self.current_estimate_ = new_estimate
            return False

        if(self.max_iter_):
            self.iter_ += 1
            if(self.iter_ > self.max_iter_):
                return True

        diff = np.linalg.norm(self.current_estimate_ - new_estimate)
        print diff

        self.current_estimate_ = new_estimate

        return diff < self.threshold_

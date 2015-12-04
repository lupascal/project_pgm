
"""
obtain a tight lower bound on the log-likelihood of an LDA model for a 
document, as described in 
Blei, Ng, and Jordan. Latent Dirichlet Allocation (2003).
two variational parameters, one for a multinomial distribution, and one
for a dirichlet distribution, are optimized with respect to the log-likelihood
"""

import numpy as np
from scipy.special import psi



def maximise_log_likelihood(corpus, list_var_dirich, list_var_multinom, voc_size):
    """
    corresponds to the M-step of Parameter estimation (paragraph 5.3)
    arguments:
        (corpus) list of word_incidences (array containing the number of times each word in the
    vocabulary appears in the document).
        (list of gamma) = list_var_dirich, list of variational parameters for the dirichlet distribution,
        (phi) = list_var_multinom, list of variational parameters for the multinomial distribution
        voc_size = vocabulary size
    returns
        (alpha) dirich_param: an estimate of the parameter of the dirichlet distribution
        which generates the parameter for the (multinomial) probability
        distribution over topics in the document.
        (beta) word_prob_given_topic: an array of size (nb topics, vocabulary size) which
        gives the (estimated) probability that a given topic will generate a
        certain word.
    """
    # compute dirich_param (alpha)
    [num_docs, num_topics] = np.shape(list_var_dirich)
    dirich_param = np.array(num_docs, 1)
    
    stop = var_inf_stop()
    while(not stop(error)):
        # compute H^{-1}(dirich_param) g(dirich_param)
        hessian_gradient = compute_hessian_gradient(dirich_param, list_var_dirich, num_docs)
        # dirich_param <- dirich_param - H^{-1}(dirich_param) g(dirich_param)
        dirich_param = dirich_param - hessian_gradient
        error = np.norm(hessian_gradient)


    # compute word_prob_given_topic (beta)
    word_prob_given_topic = np.matrix(num_topics, voc_size)
    for i in range(0, num_docs):
        # for each document and its corresponding var_dirich (phi)
        document = corpus[i]
        var_dirich = list_var_dirich[i]
        for word in document:
            word_prob_given_topic[word[0]] = word_prob_given_topic[word[0]] + word[1]*var_dirich

    return dirich_param, word_prob_given_topic



# Compute the gradient for Newton-Raphson (in order to compute dirich_param)
def compute_gradient_dirichlet(dirich_param, list_var_dirich, num_docs):
    grad_L = num_docs*(psi(np.sum(dirich_param)) - psi(dirich_param))
    for var_dirich in list_var_dirich:
        grad_L = grad_L + (psi(var_dirich) - psi(np.sum(var_dirich)))
    return grad_L


# Compute diagonal terms of the hessian matrix
def compute_hessian_dirichlet_diag(dirich_param, num_docs):
    return (num_docs*polygamma(1, dirich_param))


# Compute the vector z of the hessian matrix
def compute_hessian_dirichlet_z(dirich_param, num_docs):
    return (polygamma(1, np.sum(dirich_param))*np.ones(num_docs, 1))


# Compute H^(-1) * g where H = hessian and g = gradient for Newton-Raphson (in order to compute dirich_param)
# input = dirich_param (alpha), list_var_dirich (list of gamma), num_docs
def compute_hessian_gradient(dirich_param, list_var_dirich, num_docs):
    grad_L = compute_gradient_dirichlet(dirich_param, list_var_dirich, num_docs)
    h = compute_hessian_dirichlet_diag(dirich_param, num_docs)
    z = compute_hessian_dirichlet_z(dirich_param, num_docs)
    c = np.sum(grad_L/h)/(1/z+np.sum(1/h))
    hessian_gradient = (h - c)/grad_L
    return hessian_gradient




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

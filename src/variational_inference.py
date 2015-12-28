"""
see
Blei, Ng, and Jordan. Latent Dirichlet Allocation (2003).
"""

import numpy as np
from scipy.special import psi, polygamma, gamma
from scipy.misc import logsumexp


# EM algorithm
def latent_dirichlet_allocation(corpus, nb_topics, voc_size):
    # initialization
    (dirich_param, word_proba_given_topic) \
        = initialize_params(corpus, nb_topics, voc_size)
    converged = var_inf_stop(max_iter = 10)

    while(not converged(dirich_param)):
        # E-step
        var_dirch = []
        var_multinom = []
        
        for document in corpus:
            var_dirich_document, var_multinom_document = variational_inference(document, 
                log_dirich_param, word_logprob_given_topic, 
                save_log_likelihoods = False)
            var_dirich.append(var_dirich_document)
            var_multinom.append(var_multinom_document)
        
        # M-step
        (dirich_param, word_proba_given_topic, var_dirich) \
            = maximization_step(corpus, dirich_param, 
                                      word_proba_given_topic, dirich_param)
        
        print dirich_param
        #print compute_log_likelihood(corpus[0], dirich_param,
        #                             word_proba_given_topic,
        #                             var_dirich, var_multinom)
        
    return dirich_param, word_proba_given_topic, var_dirich
    


def initialize_params(corpus, nb_topics, voc_size):
    dirich_param = np.random.rand()
    word_proba_given_topic = np.random.rand(nb_topics, voc_size)
    word_proba_given_topic /= np.sum(word_proba_given_topic, 
                                     axis = 1)[:, np.newaxis]
    return dirich_param, word_proba_given_topic



# corresponds to the E-step
def variational_inference(document, log_dirich_param, word_logprob_given_topic,
                          save_log_likelihoods = False):

    incident_words, word_incidences = np.transpose(document)

    subvoc_size = np.size(incident_words, axis = 0)

    nb_topics = np.size(word_logprob_given_topic, axis = 0)

    var_dirich = np.exp(log_dirich_param) + np.sum(word_incidences) / nb_topics
    log_var_dirich_document = np.log(var_dirich)
        
    var_multinom = np.ones([subvoc_size, nb_topics]) / nb_topics
    
    log_var_multinom = np.zeros((subvoc_size, nb_topics)) - np.log(nb_topics)

    log_likelihood = None
    
    stop = var_inf_stop(threshold = 1e-3, max_iter = 30)

    log_likelihoods = []

    while(not stop(log_likelihood)):
        
        log_var_multinom = np.transpose(
            word_logprob_given_topic[:, incident_words]) + psi(var_dirich)

        log_var_multinom -= logsumexp(var_multinom, axis = 1)[:, np.newaxis]
        
        log_var_dirich_document = log_dirich_param + logsumexp(
            np.log(word_incidences) + np.transpose(np.exp(log_var_multinom_document)),
            axis = 1)

        log_likelihood = compute_log_likelihood(
            document, np.exp(log_dirich_param),
            np.exp(word_logprob_given_topic[:, incident_words]),
            np.exp(log_var_dirich_document),
            np.exp(log_var_multinom_document),
            word_incidences)

        if(save_log_likelihoods):
            log_likelihoods.append(log_likelihood)
            
        print 'log likelihood: %g' % log_likelihood

    print '\n'

    if(save_log_likelihoods):
        return np.exp(log_var_dirich_document), np.exp(var_multinom_document), log_likelihoods
    
    return np.exp(log_var_dirich_document), np.exp(var_multinom_document)
    
    
# corresponds to the M-step
def maximization_step(corpus, old_dirich, old_word_proba,
                            convergence_threshold = .1, old_dirich_param):
    """
    corresponds to the M-step of Parameter estimation (paragraph 5.3)
    arguments:
    (corpus) list of word_incidences (array containing the number of times 
    each word in the vocabulary appears in the document).
    (list of gamma) = list_var_dirich, list of variational parameters for 
    the dirichlet distribution,
    (phi) = list_var_multinom, list of variational parameters for the 
    multinomial distribution 
    voc_size = vocabulary size 
    returns
    (alpha) dirich_param: an estimate of the parameter of the dirichlet 
    distribution which generates the parameter for the (multinomial) 
    probability
    distribution over topics in the document.
    (beta) word_prob_given_topic: an array of size (nb topics, vocabulary 
    size) which gives the (estimated) probability that a given topic will
    generate a certain word.
    """
    num_docs = len(corpus)
    num_topics = np.shape(old_word_proba)[0]

    # compute word_prob_given_topic (beta)
    word_proba_given_topic = np.zeros(np.shape(old_word_proba))
    var_dirich = np.empty([num_docs, num_topics])

    for (index, document) in enumerate(corpus):
        # for each document and its corresponding var_multinom (phi)

        (var_dirich[index,:], var_multinom) = variational_inference(
            document, np.log(old_dirich), np.log(old_word_proba))

        np.transpose(word_proba_given_topic)[document[:,0]] \
            += document[:,1][:,np.newaxis] * var_multinom

         
    normalizing_constant = np.sum(word_proba_given_topic, axis = 1)
    assert(normalizing_constant.all())
    word_proba_given_topic /= normalizing_constant[:,np.newaxis]

    # compute dirich_param (alpha)
    dirich_param = old_dirich_param
    
    error = -1
    nb_iter = 0
    while(nb_iter < 10 and (convergence_threshold < error or error < 0)):
        nb_iter += 1
        # compute gradient and hessian of the log-likelihood wrt dirich_param
        gradient = compute_gradient_wrt_dirich_param(dirich_param, var_dirich, num_docs, num_topics)
        hessian = compute_hessian_wrt_dirich_param(dirich_param, num_docs, num_topics)
        # dirich_param <- dirich_param - H^{-1}(dirich_param) g(dirich_param)
        coefficient = gradient/hessian
        dirich_param = dirich_param - coefficient
        
        error = np.abs(coefficient)
        print error


    return dirich_param, word_proba_given_topic, var_dirich


# Compute the gradient of the expected log-likelihood wrt dirich_param
def compute_gradient_wrt_dirich_param(dirich_param, var_dirich, num_docs, num_topics):
    gradient = num_docs*num_topics*(psi(num_topics*dirich_param) - psi(dirich_param))
    gradient += np.sum(psi(var_dirich) 
                     - psi(np.sum(var_dirich, axis = 0)), axis = 0)
    return gradient

# Compute the hessian of the expected log-likelihood wrt dirich_param
def compute_hessian_wrt_dirich_param(dirich_param, num_docs, num_topics):
    hessian = (num_docs*num_topics
              * (num_topics*polygamma(1, num_topics*dirich_param)
              - polygamma(1, num_topics*dirich_param) ) )
    return hessian

# var_multinom = np.transpose(word_prob_given_topic[:,incident_words]) \
        #                * np.exp(psi(var_dirich))
        # var_multinom = (var_multinom 
        #                 / np.sum(var_multinom, axis = 1)[:, np.newaxis])        
        # var_dirich = dirich_param + np.sum(
        #     word_incidences * np.transpose(np.exp(var_multinom)), axis = 1)

def compute_log_likelihood(document, dirich_param, word_proba_given_topic,
                           var_dirich, var_multinom, word_incidences):
    log_likelihood = (
        np.log(gamma(np.sum(dirich_param)))
        - np.sum(np.log(gamma(dirich_param)))
        + np.sum((dirich_param - 1) * (psi(var_dirich)
                                       - psi(np.sum(var_dirich))))
        + np.sum(var_multinom * (psi(var_dirich) - psi(np.sum(var_dirich))))
        
        + np.sum(np.log(word_proba_given_topic)
                 * np.transpose(var_multinom)
                 * word_incidences)

        - np.log(gamma(np.sum(var_dirich))) + np.sum(np.log(var_dirich))
        - np.sum((var_dirich - 1) * psi(var_dirich)
                 - psi(np.sum(var_dirich)))
        -np.sum(var_multinom * np.log(var_multinom))
    )

    return log_likelihood


def compute_corpus_log_likelihood(corpus, dirich_param, word_proba_given_topic,
                           var_dirich, var_multinom, word_incidences):
    log_likelihood = (
        np.log(gamma(np.sum(dirich_param)))
        - np.sum(np.log(gamma(dirich_param)))
        + np.sum((dirich_param - 1) * (psi(var_dirich)
                                       - psi(np.sum(var_dirich))))
        + np.sum(var_multinom * (psi(var_dirich) - psi(np.sum(var_dirich))))
        
        + np.sum(np.log(word_proba_given_topic)
                 * np.transpose(var_multinom)
                 * word_incidences)

        - np.log(gamma(np.sum(var_dirich))) + np.sum(np.log(var_dirich))
        - np.sum((var_dirich - 1) * psi(var_dirich)
                 - psi(np.sum(var_dirich)))
        -np.sum(var_multinom * np.log(var_multinom))
    )

    return log_likelihood



class var_inf_stop(object):
    """stopping criterion for variational inference"""

    def __init__(self, threshold = 1e-6, max_iter = None):
        self.previous_log_likelihood_ = None
        self.threshold_ = threshold
        self.max_iter_ = max_iter
        if(max_iter):
            self.iter_ = 0

    def __call__(self, new_log_likelihood):

        if(self.previous_log_likelihood_ == None):
            self.previous_log_likelihood_ = new_log_likelihood
            return False

        if(self.max_iter_):
            self.iter_ += 1
            if(self.iter_ > self.max_iter_):
                return True

        diff = np.abs(new_log_likelihood - self.previous_log_likelihood_)

        self.previous_log_likelihood_ = new_log_likelihood

        return diff < self.threshold_

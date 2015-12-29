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
    corpus_log_likelihood = None

    while(not converged(corpus_log_likelihood)):
        # M-step (we compute the E-step in the M-step)
        (dirich_param, word_proba_given_topic, var_dirich, corpus_log_likelihood) \
            = maximization_step(corpus, dirich_param, word_proba_given_topic)

        #print 'log likelihood: %g' %corpus_log_likelihood
    
    return dirich_param, word_proba_given_topic, var_dirich


# initialization
def initialize_params(corpus, nb_topics, voc_size):
    dirich_param = np.random.rand()
    word_proba_given_topic = np.random.rand(nb_topics, voc_size)
    word_proba_given_topic /= np.sum(word_proba_given_topic, 
                                     axis = 1)[:, np.newaxis]
    return dirich_param, word_proba_given_topic


########################################################################################

# corresponds to the E-step
def variational_inference(document, log_dirich_param, word_logprob_given_topic,
                          save_log_likelihoods = False):

    incident_words, word_incidences = np.transpose(document)

    subvoc_size = np.size(incident_words, axis = 0)
    nb_topics = np.size(word_logprob_given_topic, axis = 0)
    
    # initialization of phi (var_multinom_document)
    log_var_multinom_document = np.zeros((subvoc_size, nb_topics)) - np.log(nb_topics)
    
    # initialization of gamma (var_dirich_document)
    var_dirich_document = np.ones(nb_topics)*(np.exp(log_dirich_param) + np.sum(word_incidences) / nb_topics)
    #var_dirich_document = np.ones(nb_topics)*(np.exp(log_dirich_param) + subvoc_size/nb_topics)
    #log_var_dirich_document = np.log(var_dirich_document)
    #print 'log_var_dirich_document:', log_var_dirich_document
    
    log_likelihood = None
    stop = var_inf_stop(threshold = 1e-3, max_iter = 10)
    log_likelihoods = []

    while(not stop(log_likelihood)):
        # compute new var_multinom_document
        log_var_multinom_document = np.transpose(
            word_logprob_given_topic[:, incident_words]) + psi(var_dirich_document)

        log_var_multinom_document -= logsumexp(log_var_multinom_document, axis = 1)[:, np.newaxis]
        
        # compute new var_dirich_document
        print var_dirich_document
        var_dirich_document = np.exp(log_dirich_param) + np.sum(
            word_incidences*np.transpose(np.exp(log_var_multinom_document)),
            axis = 1)
        
        # compute log_likelihood
        log_likelihood = compute_log_likelihood(
            word_incidences, np.exp(log_dirich_param),
            np.exp(word_logprob_given_topic[:, incident_words]),
            var_dirich_document,
            np.exp(log_var_multinom_document),
            nb_topics)
        
        if (save_log_likelihoods):
            log_likelihoods.append(log_likelihood)

    print '--- log likelihood: %g' %log_likelihood
    
    if (save_log_likelihoods):
        return var_dirich_document, np.exp(log_var_multinom_document), log_likelihoods
    
    return var_dirich_document, np.exp(log_var_multinom_document), log_likelihood


# compute the log-likehood for one document
def compute_log_likelihood(word_incidences, dirich_param, word_proba_given_topic,
                           var_dirich, var_multinom, nb_topics):
    
    log_likelihood = (np.log(gamma(nb_topics*dirich_param))
                      - nb_topics*np.log(gamma(dirich_param))
                      + (dirich_param-1)*np.sum(psi(var_dirich)
                                                     - psi(np.sum(var_dirich)))
                      
                      + np.sum(var_multinom * (psi(var_dirich) - psi(np.sum(var_dirich))))
                      
                      + np.sum(np.log(word_proba_given_topic)
                               * np.transpose(var_multinom)
                               * word_incidences)
                      
                      - np.log(gamma(np.sum(var_dirich))) 
                      + np.sum(np.log(gamma(var_dirich)))
                      - np.sum((var_dirich - 1) * psi(var_dirich)
                               - psi(np.sum(var_dirich)))
                    
                      -np.sum(var_multinom * np.log(var_multinom))
                      )
    
    #print '--- log-likelihood ---'
    #print '1-a = ', (np.log(gamma(nb_topics*dirich_param)))
    #print '1-b = ', (- nb_topics*np.log(gamma(dirich_param)))
    #print '1-c = ', (dirich_param-1)*np.sum(psi(var_dirich) - psi(np.sum(var_dirich)))
    
    #print '2-a = ', np.sum(var_multinom * (psi(var_dirich) - psi(np.sum(var_dirich))))
    #print '2-b = ', np.sum(np.log(word_proba_given_topic) * np.transpose(var_multinom)
    #                           * word_incidences)
    
    #print '3-a = ', - np.log(gamma(np.sum(var_dirich))) # -infini
    #print '3-b = ', np.sum(np.log(gamma(var_dirich)))  # (+infini)
    #print '3-c = ', - np.sum((var_dirich - 1) * psi(var_dirich) - psi(np.sum(var_dirich)))

    #print '4 = ', -np.sum(var_multinom * np.log(var_multinom))
    
    #print 'var_dirich = ', var_dirich
    #print 'sum_var_dirich = ', np.sum(var_dirich)
    #print gamma(np.sum(var_dirich))  # infini

    return log_likelihood


########################################################################################

# corresponds to the M-step
# arguments: corpus, old_dirich (alpha), old_word_proba (beta)
# returns new dirich_param, new word_prob_given_topic
def maximization_step(corpus, old_dirich, old_word_proba, convergence_threshold = .1):

    num_docs = len(corpus)
    nb_topics = np.shape(old_word_proba)[0]

    # compute word_prob_given_topic (beta)
    word_proba_given_topic = np.zeros(np.shape(old_word_proba))
    
    # var_dirich
    sum_psi_var_dirich = 0 # will be used for the gradient of L wrt dirich_param
    var_dirich = np.empty([num_docs, nb_topics])
    
    # corpus log_likelihood    
    corpus_log_likelihood = 0
    
    # for each document and its corresponding var_dirich (gamma) and var_multinom (phi)
    for (index, document) in enumerate(corpus):
        # E-step
        (var_dirich[index,:], var_multinom, log_likelihood) = variational_inference(
            document, np.log(old_dirich), np.log(old_word_proba))

        # update word_proba_given_topic (beta) of the M-step
        np.transpose(word_proba_given_topic)[document[:,0]] \
            += document[:,1][:,np.newaxis] * var_multinom
        
        # update sum_psi_var_dirich
        sum_psi_var_dirich += np.sum(psi(var_dirich[index,:]) 
                     - psi(np.sum(var_dirich[index,:])))
        
        # update corpus_log_likelihood
        corpus_log_likelihood += log_likelihood
    
    # normalization of word_proba_given_topic
    normalizing_constant = np.sum(word_proba_given_topic, axis = 1)
    assert(normalizing_constant.all())
    word_proba_given_topic /= normalizing_constant[:,np.newaxis]

    # M-step: compute dirich_param (alpha)
    dirich_param = old_dirich
    error = -1
    nb_iter = 0
    while(nb_iter < 10 and (convergence_threshold < error or error < 0)):
        nb_iter += 1
        # compute gradient and hessian of the log-likelihood wrt dirich_param
        gradient = num_docs*nb_topics*(psi(nb_topics*dirich_param) 
                    - psi(dirich_param)) + sum_psi_var_dirich
        hessian = compute_hessian_wrt_dirich_param(dirich_param, num_docs, nb_topics)
        # dirich_param <- dirich_param - H^{-1}(dirich_param) g(dirich_param)
        coefficient = gradient/hessian
        dirich_param = dirich_param - coefficient
        
        error = np.abs(coefficient)
        print error
    
    return dirich_param, word_proba_given_topic, var_dirich, corpus_log_likelihood


# Compute the gradient of the expected log-likelihood wrt dirich_param
#def compute_gradient_wrt_dirich_param(dirich_param, var_dirich, num_docs, num_topics):
#    gradient = num_docs*num_topics*(psi(num_topics*dirich_param) - psi(dirich_param))
#    gradient += np.sum(psi(var_dirich) 
#                     - psi(np.sum(var_dirich, axis = 0)), axis = 0)
#    return gradient

# Compute the hessian of the expected log-likelihood wrt dirich_param
def compute_hessian_wrt_dirich_param(dirich_param, num_docs, num_topics):
    hessian = (num_docs*num_topics
              * (num_topics*polygamma(1, num_topics*dirich_param)
              - polygamma(1, num_topics*dirich_param) ) )
    return hessian

########################################################################################

# var_multinom = np.transpose(word_prob_given_topic[:,incident_words]) \
        #                * np.exp(psi(var_dirich))
        # var_multinom = (var_multinom 
        #                 / np.sum(var_multinom, axis = 1)[:, np.newaxis])        
        # var_dirich = dirich_param + np.sum(
        #     word_incidences * np.transpose(np.exp(var_multinom)), axis = 1)


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

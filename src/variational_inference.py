"""
see
Blei, Ng, and Jordan. Latent Dirichlet Allocation (2003).
"""

import numpy as np
from scipy.special import psi, polygamma, gammaln
from scipy.misc import logsumexp

# EM algorithm
def latent_dirichlet_allocation(corpus, nb_topics, voc_size):
    # initialization
    (dirich_param, word_logproba_given_topic) \
        = initialize_params(corpus, nb_topics, voc_size)
    converged = var_inf_stop(threshold = 1e-4, max_iter = 100)
    corpus_log_likelihood = None
    log_likelihoods = []
    
    while(not converged(corpus_log_likelihood)):
        # M-step (we compute the E-step in the M-step)
        (dirich_param, word_logproba_given_topic, corpus_log_likelihood) \
            = maximization_step(corpus, dirich_param, word_logproba_given_topic)
        
        log_likelihoods.append(corpus_log_likelihood)
        print 'corpus log likelihood: %g' %corpus_log_likelihood
        
    return dirich_param, word_logproba_given_topic, log_likelihoods


# initialization
def initialize_params(corpus, nb_topics, voc_size):
    dirich_param = np.random.rand()
    word_logproba_given_topic = np.log(np.random.rand(nb_topics, voc_size))
    word_logproba_given_topic -= logsumexp(word_logproba_given_topic, 
                                     axis = 1)[:, np.newaxis]
    return dirich_param, word_logproba_given_topic


######################################################################

# corresponds to the E-step
def variational_inference(document, dirich_param, word_logprob_given_topic,
                          verbose = False):

    (incident_words, word_incidences, subvoc_size, nb_topics, log_var_multinom,
     var_dirich, log_likelihood, log_likelihoods, stop) = _var_inf_init(
         document,dirich_param, word_logprob_given_topic)
    
    while(not stop(log_likelihood)):
        #log_var_multinom_old = log_var_multinom
        
        # compute new var_multinom
        log_var_multinom = np.transpose(
            word_logprob_given_topic[:, incident_words]) \
            + psi(var_dirich)

        log_var_multinom -= logsumexp(log_var_multinom,
                                               axis = 1)[:, np.newaxis]

        var_dirich = dirich_param + np.sum(
            word_incidences * np.transpose(np.exp(log_var_multinom)),
            axis = 1)

        # var_dirich = dirich_param + np.sum(
        #     word_incidences * np.transpose(np.exp(log_var_multinom)
        #                                - np.exp(log_var_multinom_old)))

        log_likelihood = compute_log_likelihood(
            word_incidences, dirich_param,
            word_logprob_given_topic[:, incident_words],
            var_dirich,
            np.exp(log_var_multinom))
        
        log_likelihoods.append(log_likelihood)

        if (verbose): print 'log likelihood: %g' %log_likelihood
    
    return (var_dirich, log_var_multinom,
            log_likelihoods)
    


def _var_inf_init(
        document,dirich_param, word_logprob_given_topic):

    incident_words, word_incidences = np.transpose(document)

    subvoc_size = np.size(incident_words, axis = 0)
    nb_topics = np.size(word_logprob_given_topic, axis = 0)
    
    log_var_multinom = np.zeros((subvoc_size, nb_topics)) \
                                - np.log(nb_topics)

    var_dirich = np.ones(nb_topics)*(
        dirich_param + np.sum(word_incidences) / nb_topics)

    log_likelihood = None
    
    stop = var_inf_stop(threshold = 1e-6, max_iter = 20)
    log_likelihoods = []

    return (incident_words, word_incidences, subvoc_size,
            nb_topics, log_var_multinom, var_dirich, log_likelihood,
            log_likelihoods, stop)



# compute the log-likehood for one document

def compute_log_likelihood(word_incidences, dirich_param,
                           word_logproba_given_topic,
                           var_dirich, var_multinom):

    nb_topics = np.size(word_logproba_given_topic, axis = 0)

    log_likelihood = (gammaln(nb_topics*dirich_param)
                      - nb_topics*gammaln(dirich_param)
                      + (dirich_param-1)*np.sum(psi(var_dirich)
                                                     - psi(np.sum(var_dirich)))
                      
                      + np.sum(var_multinom * (psi(var_dirich) 
                                  - psi(np.sum(var_dirich))))
                      
                      + np.sum(word_logproba_given_topic
                               * np.transpose(var_multinom)
                               * word_incidences)
                      
                      - gammaln(np.sum(var_dirich))
                      + np.sum(gammaln(var_dirich))
                      - np.sum((var_dirich - 1) * psi(var_dirich)
                               - psi(np.sum(var_dirich)))
                    
                      -np.sum(var_multinom * np.log(var_multinom))
                      )
    
    # for debug
    if (np.isnan(log_likelihood) or np.isinf(log_likelihood)):
        
        a1 = gammaln(nb_topics*dirich_param)
        a2 = - nb_topics*gammaln(dirich_param)
        a3 = + (dirich_param-1)*np.sum(psi(var_dirich) - psi(np.sum(var_dirich)))
                      
        b1 = + np.sum(var_multinom * (psi(var_dirich) - psi(np.sum(var_dirich))))                
        b2 = + np.sum(word_logproba_given_topic * np.transpose(var_multinom)
                               * word_incidences)
                      
        c1 = - gammaln(np.sum(var_dirich))
        c2 = + np.sum(gammaln(var_dirich))
        c3 = - np.sum((var_dirich - 1) * psi(var_dirich) - psi(np.sum(var_dirich)))
                    
        d1 = -np.sum(var_multinom * np.log(var_multinom))
        
        if (np.isnan(a1) or np.isinf(a1)) :
            print 'a1 = ', a1 
        if (np.isnan(a2) or np.isinf(a2)) :
            print 'a2 = ', a2
        if (np.isnan(a3) or np.isinf(a3)) :
            print 'a3 = ', a3
        if (np.isnan(b1) or np.isinf(b1)) :
            print 'b1 = ', b1 
        if (np.isnan(b2) or np.isinf(b2)) :
            print 'b2 = ', b2
        if (np.isnan(c1) or np.isinf(c1)) :
            print 'c1 = ', c1
        if (np.isnan(c2) or np.isinf(c2)) :
            print 'c2 = ', c2
        if (np.isnan(c3) or np.isinf(c3)) :
            print 'c3 = ', c3
        if (np.isnan(d1) or np.isinf(d1)) :
            print 'd1 = ', d1

    return log_likelihood


########################################################################################

# corresponds to the M-step
# arguments: corpus, old_dirich (alpha), old_word_proba (beta)
# returns new dirich_param, new word_prob_given_topic
def maximization_step(corpus, old_dirich, log_old_word_proba,
                      convergence_threshold = .1):

    num_docs = len(corpus)
    nb_topics = np.shape(log_old_word_proba)[0]

    # compute word_prob_given_topic (beta)
    word_logproba_given_topic = np.zeros(np.shape(log_old_word_proba))
    
    # var_dirich
    sum_psi_var_dirich = 0 # will be used for the gradient of L wrt dirich_param
    
    # corpus log_likelihood    
    corpus_log_likelihood = 0
    
    # for each document and its corresponding var_dirich (gamma) and var_multinom (phi)
    for (index, document) in enumerate(corpus):
        # E-step
        (var_dirich, log_var_multinom, log_likelihoods) = variational_inference(
            document, old_dirich, log_old_word_proba, verbose = False)
        log_likelihood = log_likelihoods[-1]
        
        # update corpus_log_likelihood
        assert (not np.isnan(log_likelihood)), \
                'nan encountered in variational inference'
        
        if (not (np.isinf(log_likelihood))):
            # update word_proba_given_topic (beta) of the M-step
            np.transpose(word_logproba_given_topic)[document[:,0]] \
                = logsumexp([
                    np.transpose(word_logproba_given_topic)[document[:,0]],
                    np.log(document[:,1][:,np.newaxis]) + log_var_multinom],
                            axis = 0)
        
            # update sum_psi_var_dirich
            sum_psi_var_dirich += np.sum(psi(var_dirich) - psi(np.sum(var_dirich)))
            
            corpus_log_likelihood += log_likelihood

        else: print 'warning: inf encountered in variational inference'
        
    # normalization of word_proba_given_topic
    normalizing_constant = logsumexp(word_logproba_given_topic, axis = 1)
    assert(normalizing_constant.all())
    word_logproba_given_topic -= normalizing_constant[:,np.newaxis]

    # M-step: compute dirich_param (alpha)
    dirich_param = old_dirich
    error = -1
    nb_iter = 0
    while(nb_iter < 20 and (convergence_threshold < error or error < 0)):
        nb_iter += 1
        # compute gradient and hessian of the log-likelihood wrt dirich_param
        gradient = num_docs*nb_topics*(psi(nb_topics*dirich_param) 
                    - psi(dirich_param)) + sum_psi_var_dirich
        hessian = compute_hessian_wrt_dirich_param(dirich_param, num_docs, nb_topics)
        # dirich_param <- dirich_param - H^{-1}(dirich_param) g(dirich_param)
        coefficient = gradient/hessian
        dirich_param = dirich_param - coefficient
        
        error = np.abs(coefficient)
        #print 'error: %g' % error
    
    return (dirich_param, word_logproba_given_topic, corpus_log_likelihood)


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

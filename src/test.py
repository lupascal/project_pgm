from os import path
import numpy as np
from matplotlib import pyplot as plt
import doc_preprocessing as dp
reload(dp)
import variational_inference as vi
reload(vi)

# path_to_reuters = path.expanduser(
#       '/home/student/probabilistic_graphical_models/project_pgm/reuters21578/')

path_to_reuters = path.expanduser(
 '~/Documents/MVA/proba_graph_models/project/reuters_21578')

def test_variational_inference(voc = None, docs = None,
                               max_files = None, doc_num = None, n_topics = 20,
                               dirich_param = .5,
                               log_word_proba_given_topic = None,
                               **kwargs):

    if(voc == None or docs == None):
        #voc, docs = dp.build_voc(dp.find_reuters_files(path_to_reuters)[:max_files])
        voc, docs = dp.build_voc([path.join(path_to_reuters,
                                            'reut2-000.sgm')])
        print voc.keys()[:10]

    voc_size = len(voc)

    if doc_num == None:
        doc_count = len(docs)
        print 'doc_count: %d' % doc_count
        doc_num = np.random.randint(doc_count)

    if log_word_proba_given_topic == None:
        word_proba_given_topic = np.random.rand(n_topics * voc_size).reshape(
            (n_topics, voc_size))
        word_proba_given_topic /= np.sum(word_proba_given_topic,
                                         axis = 1).reshape((-1,1))
        log_word_proba_given_topic = np.log(word_proba_given_topic)
    
    
    # test for a document d
    var_dirich, var_multinom, log_likelihoods = vi.variational_inference(
        docs[doc_num], dirich_param, log_word_proba_given_topic, **kwargs)
    
    plt.figure(1)
    plt.plot(log_likelihoods)   
    plt.xlabel('iterations')
    plt.ylabel('expected log-likelihood')
    plt.title('expected log-likelihood for a document d, k = ' + str(n_topics))

    
    # test for a corpus
    (dirich_param, word_logproba_given_topic, corpus_log_likelihood) = vi.latent_dirichlet_allocation(docs, n_topics, voc_size)
    plt.figure(2)
    plt.plot(corpus_log_likelihood)    
    plt.xlabel('iterations')
    plt.ylabel('expected log-likelihood')
    plt.title('expected log-likelihood for a corpus, k = ' + str(n_topics))
    plt.show()
    
    top_words = topic_top_words(word_logproba_given_topic, voc, num_words = 10)    
    
    print top_words    
    
    #return (dirich_param, word_logproba_given_topic, top_words)


# display the ten first words for each topic
def topic_top_words(word_logproba_given_topic, voc, num_words = 10):
    order = np.argsort(word_logproba_given_topic, axis = 1)[:,:num_words]
    reversed = dp.reverse_dict(voc)
    top_words = [[reversed[index] for index in topic] for topic in order]
    return top_words


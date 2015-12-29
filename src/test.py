from os import path
import numpy as np
from matplotlib import pyplot as plt
import doc_preprocessing as dp
reload(dp)
import variational_inference as vi
reload(vi)

# path_to_reuters = path.expanduser(
#      '/home/student/probabilistic_graphical_models/project_pgm/src/reuters21578/')

path_to_reuters = path.expanduser(
  '~/Documents/MVA/proba_graph_models/project/reuters_21578')

def test_variational_inference(voc = None, docs = None,
                               max_files = None, doc_num = None, n_topics = 30,
                               dirich_param = .5,
                               log_word_proba_given_topic = None,
                               **kwargs):

    if(voc == None or docs == None):
        #voc, docs = dp.build_voc(dp.find_reuters_files(path_to_reuters)[:max_files])
        voc, docs = dp.build_voc([path.join(path_to_reuters,
                                            'reuters_short.sgm')])
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

    var_dirich, var_multinom, log_likelihoods = vi.variational_inference(
        docs[doc_num], dirich_param, log_word_proba_given_topic, **kwargs)

    plt.plot(log_likelihoods)

    lda_result = vi.latent_dirichlet_allocation(docs, n_topics, voc_size)
    
    return (lda_result, voc)

def topic_top_words(word_logproba_given_topic, voc, num_words = 10):

    order = np.argsort(word_logproba_given_topic, axis = 1)[:,:num_words]
    reversed = dp.reverse_dict(voc)
    top_words = [[reversed[index] for index in topic] for topic in order]
    return top_words
        

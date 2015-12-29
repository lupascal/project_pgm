from os import path
import numpy as np
from matplotlib import pyplot as plt
import doc_preprocessing as dp
import variational_inference as vi
reload(vi)

path_to_reuters = path.expanduser(
    '/home/student/probabilistic_graphical_models/project_pgm/src/reuters21578/')

#path_to_reuters = path.expanduser(
#    '~/Documents/MVA/proba_graph_models/project/reuters_21578')

def test_variational_inference(voc = None, docs = None,
                               max_files = None, doc_num = None, n_topics = 30,
                               dirich_param = None,
                               word_proba_given_topic = None):

    if(voc == None or docs == None):
        voc, docs = dp.build_voc(dp.find_reuters_files(path_to_reuters)[:max_files])
        
    voc_size = len(voc)

    if doc_num == None:
        doc_count = len(docs)
        doc_num = 0 #np.random.randint(doc_count)
        
    if dirich_param == None:
        log_dirich_param = np.log(1)

    if word_proba_given_topic == None:
        log_word_proba_given_topic = np.zeros((n_topics, voc_size)) \
                                     - np.log(voc_size) 
        #word_proba_given_topic = np.random.rand(n_topics, voc_size)
        #word_proba_given_topic /= np.sum(word_proba_given_topic, axis = 1)[:, np.newaxis]
        #log_word_proba_given_topic = np.log(word_proba_given_topic)        
        
    var_dirich, var_multinom, log_likelihoods = vi.variational_inference(
        docs[doc_num], log_dirich_param, log_word_proba_given_topic,
        save_log_likelihoods = True)

    plt.plot(log_likelihoods)
    
    #vi.latent_dirichlet_allocation(docs, n_topics, voc_size)

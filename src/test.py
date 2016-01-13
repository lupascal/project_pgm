from os import path
import numpy as np
from matplotlib import pyplot as plt
import doc_preprocessing as dp
reload(dp)
import variational_inference as vi
reload(vi)
import utils

path_to_project = '/home/student/probabilistic_graphical_models/project_pgm/'
#path = '~/Documents/MVA/proba_graph_models/project/'

path_to_reuters = path.expanduser(path_to_project + 'reuters21578/')

def results_dir():
    return path.expanduser(path_to_project + 'results/')


class Dirich_features_logger(utils.Logger):
    def __init__(self, **kwargs):
        utils.Logger.__init__(self, self_dir_name = 'lda_features', **kwargs)
        self.lda_file_path = path.join(self.dir_path, 'lda_features.npy')

    def call_hook_(self, dirich_features):
        with open(self.lda_file_path, 'w') as lda_file:
            np.save(lda_file, dirich_features)


    
def test_variational_inference(voc = None, docs = None,
                               max_files = None, doc_num = None, n_topics = 20,
                               dirich_param = .5,
                               log_word_proba_given_topic = None,
                               **kwargs):

    description = {'n_topics': n_topics,
                   'dirich_param': dirich_param}
    
    if(voc == None or docs == None):
        files_list = [path.join(path_to_reuters, 'reut2-000.sgm'), 
                      path.join(path_to_reuters, 'reut2-001.sgm'),
                      path.join(path_to_reuters, 'reut2-002.sgm'),
                      path.join(path_to_reuters, 'reut2-003.sgm'),
                      path.join(path_to_reuters, 'reut2-004.sgm'),
                      path.join(path_to_reuters, 'reut2-005.sgm'),
                      path.join(path_to_reuters, 'reut2-006.sgm'),
                      path.join(path_to_reuters, 'reut2-007.sgm'),
                      path.join(path_to_reuters, 'reut2-008.sgm'),
                      path.join(path_to_reuters, 'reut2-009.sgm')]
        print files_list        
        description['data_files_list'] = files_list
        voc, docs = dp.build_voc(files_list)
        print voc.keys()[:10]

    voc_size = len(voc)
    description['voc_size'] = voc_size
    
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


    # # test for a document d
    # var_dirich, var_multinom, log_likelihoods = vi.variational_inference(
    #     docs[doc_num], dirich_param, log_word_proba_given_topic, **kwargs)

    # plt.figure(1)
    # plt.plot(log_likelihoods)
    # plt.xlabel('iterations')
    # plt.ylabel('expected log-likelihood')
    # plt.title('expected log-likelihood for a document d, k = '
    # + str(n_topics))

    # test for a corpus
    logger = Dirich_features_logger(root_results_dir = results_dir(),
                                    description = description)
    
    (dirich_param, word_logproba_given_topic, corpus_log_likelihood) \
        = vi.latent_dirichlet_allocation(docs, n_topics, voc_size,
                                         max_iter = 200, var_inf_max_iter = 200,
                                         logger = logger)
    # plt.figure(2)
    # plt.plot(corpus_log_likelihood)
    # plt.xlabel('iterations')
    # plt.ylabel('expected log-likelihood')
    # plt.title('expected log-likelihood for a corpus, k = ' + str(n_topics))
    # plt.show()

    # top_words = topic_top_words(
    # word_logproba_given_topic, voc, num_words = 10)
    # print top_words

    #return (dirich_param, word_logproba_given_topic, top_words)


# display the ten first words for each topic
def topic_top_words(word_logproba_given_topic, voc, num_words = 10):
    order = np.argsort(word_logproba_given_topic, axis = 1)[:,:num_words]
    reversed = dp.reverse_dict(voc)
    top_words = [[reversed[index] for index in topic] for topic in order]
    return top_words


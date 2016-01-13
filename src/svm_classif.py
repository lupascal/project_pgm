from os import path
import re
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn import datasets
import utils

def svm_results_dir():
    return path.expanduser(
        '~/Documents/MVA/proba_graph_models/project/results_tmp')

def svm_load_wrapper(file_path):
    print 'load svm file'
    with open(file_path, 'rb') as data_file:
        features, labels = datasets.load_svmlight_file(data_file)
        print type(features)
        return features

def extract_training_and_test(features_file_name, labels_file_name):
    
    load_f = np.loadtxt
    if(re.match(r'.*\.npy', features_file_name) != None): load_f = np.load
    if(re.match(r'.*\.svm', features_file_name) != None):
        load_f = svm_load_wrapper
        
    features = load_f(features_file_name)
    labels = np.loadtxt(labels_file_name)
    n_samples = np.size(labels)
    # print ('nb features: %d\nnb_samples: %d'
    #        % (np.size(features, axis = 1), n_samples))

    return (features, labels, n_samples)


def choose_train_indices(n_samples, train_percent):
    n_train = n_samples * train_percent // 100
    n_train = max(n_train, 20)
    train_indices = np.random.choice(np.arange(n_samples), n_train,
                                     replace = False)
    test_indices = np.setdiff1d(np.arange(n_samples), train_indices)
    return train_indices, test_indices

    
def train_classif(features, labels):
    classifier = svm.SVC(kernel = 'rbf', gamma = 1)
    classifier.fit(features, labels)
    return classifier


def test_classif(classif, features, true_labels):
    n_test_samples = np.size(true_labels)
    computed_labels = classif.predict(features)
    differences = computed_labels - true_labels
    n_errors = np.sum(np.abs(differences))
    n_false_positives = np.sum((1 - true_labels) * computed_labels)
    n_false_negatives = np.sum(true_labels * (1 - computed_labels))
    n_true_positives = np.sum(true_labels * computed_labels)
    n_true_negatives = np.sum((1 - true_labels) * (1 - computed_labels))
    print 'proportion of errors: %g' % (1.0 * n_errors / n_test_samples)
    del(computed_labels)
    assert(n_false_negatives + n_false_positives == n_errors)
    return {'false_positives': n_false_positives,
            'false_negatives': n_false_negatives,
            'true_positives': n_true_positives,
            'true_negatives': n_true_negatives,
            'n_test_samples': n_test_samples}


def classif_experiment(features_file_name, labels_file_name,
                       train_percent_list = [1, 5, 10, 20], n_rep = 5,
                       results_files_prefix = '',
                       results_dir = svm_results_dir()):

    (features, labels, n_samples) = extract_training_and_test(
        features_file_name, labels_file_name)

    results_dir = utils.make_new_dir(results_dir, 'svm_experiment')
    with open(path.join(results_dir, 'description.txt'), 'w') as desc_file:
        desc_file.write('features file: %s\n' % features_file_name)
        desc_file.write('labels file: %s\n' % labels_file_name)
        desc_file.write('n samples: %d\n' % n_samples)
        desc_file.write('n positives: %d\n' % np.sum(labels))
        
    for train_percent in train_percent_list:
        file_name_prefix = '%s_train_%d_percent' % (results_files_prefix,
                                                    train_percent)
        
        file_name_prefix = utils.find_good_name(results_dir, file_name_prefix)
        
        fp_file_name = utils.find_good_name(
            results_dir, '%s_false_positives.txt' % file_name_prefix)
        fn_file_name = utils.find_good_name(
            results_dir, '%s_false_negatives.txt' % file_name_prefix)
        tp_file_name = utils.find_good_name(
            results_dir, '%s_true_positives.txt' % file_name_prefix)
        tn_file_name = utils.find_good_name(
            results_dir, '%s_true_negatives.txt' % file_name_prefix)
        with open(fp_file_name, 'a') as fp_file:
            with open(fn_file_name, 'a') as fn_file:
                with open(tp_file_name, 'a') as tp_file:
                    with open(tn_file_name, 'a') as tn_file:
                        
                        for rep in xrange(n_rep):
                            print ('%d percent of train, repetition %d'
                                   % (train_percent, rep))
                
                            train_indices, test_indices = choose_train_indices(
                                n_samples, train_percent)
            
                            classif = train_classif(
                                features[train_indices,:],
                                labels[train_indices])
            
                            perf = test_classif(
                                classif, features[test_indices,:],
                                labels[test_indices])
                
                            fp_file.write('%d\n' % perf['false_positives'])
                            fn_file.write('%d\n' % perf['false_negatives'])
                            tp_file.write('%d\n' % perf['true_positives'])
                            tn_file.write('%d\n' % perf['true_negatives'])
                            


def plot_classif_results(file_name_prefix):
    means = []
    variances = []
    minima = []
    maxima = []
    for percent in [1,5,10,20]:
        res = np.loadtxt(
            path.join(svm_results_dir(),
                      '%s_train_%d_percent_0.txt'
                      % (file_name_prefix, percent)))

        acc = 1 - res
        means.append(np.mean(acc))
        variances.append(np.var(acc))
        minima.append(np.min(acc))
        maxima.append(np.max(acc))

    errors = np.array(maxima) - np.array(minima)
    plt.errorbar([1,5,10,20], means, yerr = errors / 2)
    ax = plt.gcf().gca()
    ax.set_xbound(0, 25)
    ax.set_ybound(.75, .9)
    return np.array(means)

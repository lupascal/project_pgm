import numpy as np
import re
import mmap
import os




word_or_doc_end_ = re.compile(r'(\b\w+)|(</BODY>)')

stop_words_ = frozenset(['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by',
                         'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its',
                         'of', 'on', 'said', 'that', 'the', 'to', 'was',
                         'were', 'will', 'with', 'reuter'])


def build_voc(file_names, **kwargs):
    voc = {}
    voc_size = 0
    docs_found = 0
    documents = []

    for file_name in file_names:

        with open(file_name, 'r') as file_handle:
            
            mf = mmap.mmap(file_handle.fileno(), 0, access = mmap.ACCESS_READ) 
            pos = 0
            file_over = False

            while(not file_over):
                           
                open_tag = re.search(r'<BODY>', mf[pos:])

                if(open_tag):
                    docs_found += 1
                    file_handle.seek(pos + open_tag.end())
                    (voc_size, pos, doc_rep) = add_to_voc(
                        file_handle, voc, voc_size, **kwargs)
                    documents += [doc_rep]
                else:
                    file_over = True

    return voc, documents


    

def add_to_voc(file_handle, voc, voc_size, stop_words = stop_words_,
               to_lower = False, stop_case_sensitive = False):

    document_voc = {}

    while(True):
        line = file_handle.readline()
        words = re.findall(word_or_doc_end_, line)
        for (word, end_tag) in words:
            if(word):
                assert(not end_tag)
                cmp_word = word
                
                if(not stop_case_sensitive):
                    cmp_word = str.lower(cmp_word)
                    
                if(not cmp_word in stop_words):
                    if(to_lower):
                        word = str.lower(word)

                    if(not voc.has_key(word)):
                        voc[word] = voc_size
                        voc_size += 1
                        
                    if(document_voc.has_key(word)):
                        document_voc[word] += 1
                    else:
                        document_voc[word] = 1
            elif(end_tag):
                document_rep = np.array([(voc[word], document_voc[word]) 
                                         for word in document_voc])
                return(voc_size, file_handle.tell(), document_rep)

    return(voc_size, file_handle.tell(), document_rep)



def find_reuters_files(directory = os.getcwd()):
    all_files = os.listdir(directory)
    reuters_files = []
    for file_name in all_files:
        if(re.match(r'reut2\-\d\d\d\.sgm', file_name)):
            reuters_files += [os.path.join(directory, file_name)]

    reuters_files.sort()
    return reuters_files


def frequencies(voc, docs):
    freq = np.zeros(len(voc))
    for doc in docs:
        freq[doc[:,0]] += doc[:,1]
    return freq

def order_by_freq(voc, docs):
    freq = frequencies(voc, docs)
    return np.argsort(freq)[::-1]


def most_frequent(voc, docs, nb = 20, to_lower = False):
    order = order_by_freq(voc, docs)
    frequent_words = []
    if(to_lower):
        for i in range(nb):
            frequent_words += [
                str.lower(word) for word in voc if voc[word] == order[i]]
    else:
        for i in range(nb):
            frequent_words += [word for word in voc if voc[word] == order[i]]
            
    return frequent_words

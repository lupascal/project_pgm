from operator import itemgetter
import numpy as np
import doc_preprocessing as dp

def prepare_data(reuters_files, data_file_name, voc_file_name):
    voc = dp.build_voc(reuters_files)

    with open(data_file_name, 'w') as data_file:
        for doc in voc[1]:
            data_file.write('%d ' % np.size(doc, axis = 0))

            for word in doc:
                data_file.write('%d:%d ' % (word[0], word[1]))

            data_file.write('\n')

    sorted_voc = sorted(voc[0].items(), key = itemgetter(1))

    with open(voc_file_name, 'w') as voc_file:
        for item in sorted_voc:
            voc_file.write('%s\n' % item[0])

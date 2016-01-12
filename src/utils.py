import os
from os import path
import time
import re
import numpy as np



def find_good_name(parent_dir, name):
    stripped_name_match = re.match(r'(\.?[^\.]*)(\..*)?', name)
    stripped_name = stripped_name_match.group(1)
    extension = stripped_name_match.group(2)
    if (extension == None): extension = ''
    
    parent_dir = os.path.expanduser(parent_dir)
    suffix = 0
    name_pattern = re.compile(
        '^.*' + stripped_name + '_(\d+)' + extension + '$')

    for other_name in os.listdir(parent_dir):
        m = name_pattern.match(other_name)
        if(m):
            nb = int(m.group(1))
            if(suffix <= nb): suffix = nb + 1

    good_name = '%s_%d%s' % (stripped_name, suffix, extension)
    total_path = os.path.join(parent_dir, good_name)
    return total_path

    
def make_new_dir(parent_dir, dir_name):

    good_dir_path = find_good_name(parent_dir, dir_name)
    os.mkdir(good_dir_path)
    return good_dir_path



class Logger(object):
    def __init__(self, root_results_dir,
                 log_period = 1,
                 self_dir_name = 'results',
                 description = None,
                 message = None):
        
        self.dir_path = make_new_dir(root_results_dir, self_dir_name)
        self.log_period = log_period
        self.time = 0
        self.message = message
        
        self.desc_file_path = path.join(self.dir_path, 'description.txt')
        with open(self.desc_file_path, 'w') as desc_file:
            desc_file.write('time begin : %s\n' % time.ctime())
            if(description != None):
                self.write_description(desc_file, description)


    def write_description(self, desc_file, description):
        for (key, val) in description.iteritems():
            desc_file.write('%s : %s\n' % (str(key), str(val)))
        
    def __call__(self, *args):
        self.time += 1
        if(self.message != None):
            msg = self.message(self.time)
            if(msg != None): print msg

        self.call_hook_(*args)

    def done(self, *args):
        with open(self.desc_file_path, 'a') as desc_file:
            desc_file.write('log called %d times\n' % self.time)
            desc_file.write('time end : %s' % time.ctime())
            
        self.done_hook_(*args)

    def write_array(self, file_name, array):
        if(not re.match(r'.*\.npy', file_name)): file_name += '.npy'
        file_path = find_good_name(self.dir_path, file_name)
        with open(file_path, 'w') as file_handle:
            np.save(file_handle, array)

    def write(self, file_name, text):
        file_path = find_good_name(self.dir_path, file_name)
        with open(file_path, 'w') as file_handle:
            file_handle.write(text)
            

    def call_hook_(self, *args): pass
    def done_hook_(self, *args): pass

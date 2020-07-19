import argparse
import os
import tensorflow as tf

from tqdm import tqdm

def count_examples(tfrecord_file):
    '''Returns count of number of examples in a tfrecord'''
    count = 0
    for _ in tf.python_io.tf_record_iterator(tfrecord_file):
        count += 1
        
    return count


def list_files(src_loc, file_extension):
    '''List files in src_loc and returns files which have the valid file_extension'''
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(src_loc):
        for file in f:
            if file.endswith(file_extension):
                files.append(os.path.join(r, file))
    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Counts number of examples.')
    parser.add_argument('--tfrecords_loc', help='Location of tfrecords.')
    args = parser.parse_args()
    
    tfrecords_loc = args.tfrecords_loc
    tfrecords = list_files(src_loc=tfrecords_loc, file_extension='.tfrecords')
    count = 0
    
    for tfrecord in tqdm(tfrecords):
        temp = count_examples(tfrecord)
        if temp == 0:
            print("FILE:", tfrecord, " has no examples. Please check.")
        count += temp 
        
    print('Number of examples:', count)

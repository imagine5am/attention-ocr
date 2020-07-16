import os
import random

from numpy.random import choice
from math import ceil, log

def list_tfrecords(path, file_extension='.tfrecords'):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if file_extension in file:
                files.append(os.path.join(r, file))
    # files = sorted([x[::-1] for x in files])
    # files = [x[::-1] for x in files]
    return sorted(files)


def generator(list_):
    for item in list_:
        yield item
        

def output_to_file(filename, list_):
    fout = open(filename, "w")
    for item in list_:
        fout.write(item)
        fout.write("\n")
    fout.close()
    
    
def choose_synth_rec(npr_recs, npr_count, synth_recs, synth_count, npr_gen, synth_gen, ordered_files):
    synth_options = [npr_gen, synth_gen]
    
    npr_recs_left = len(npr_recs) - npr_count
    synth_recs_left = len(synth_recs) - synth_count
    npr_prob = npr_recs_left / (npr_recs_left + synth_recs_left)
    synth_prob = synth_recs_left / (npr_recs_left + synth_recs_left)
    # print(npr_prob, synth_prob)
    
    if npr_recs_left and synth_recs_left:
        synth_gen_choice = choice(synth_options, p=[npr_prob, synth_prob])
        npr_count += synth_gen_choice == npr_gen
        synth_count += synth_gen_choice == synth_gen
        ordered_files.append(next(synth_gen_choice))
    elif synth_recs_left:
        synth_count += 1
        ordered_files.append(next(synth_gen))
    elif npr_recs_left:
        npr_count += 1
        ordered_files.append(next(npr_gen))
    else:
        print('There is some problem.')
        print('npr_recs_left:', str(npr_recs_left), 'synth_recs_left:', str(synth_recs_left))
        
    return npr_count, synth_count


def mix_synth_real():
    npr_loc = '/mnt/data/Rohit/ACMData/5aSynthVideosE2E/TrainingDataRecordsvideosNPR/'
    synth_loc = '/mnt/data/Rohit/ACMData/5aSynthVideosE2E/TrainingDataRecordsvideos/'
    real123_loc = '/mnt/data/Rohit/ACMData/tftrainallFinal/mixed_data/mix1_ready/'
    
    npr_recs = list_tfrecords(npr_loc)
    synth_recs = list_tfrecords(synth_loc)
    real123_recs = list_tfrecords(real123_loc)
    
    num_synth_recs = len(npr_recs) + len(synth_recs)
    num_real_recs = len(real123_recs)
    
    num_total_recs = num_synth_recs + num_real_recs
    
    # Assuming number of real tfrecords < number of synthetic records.
    target_ratio =  num_synth_recs / num_real_recs
    
    npr_gen = generator(npr_recs)
    synth_gen = generator(synth_recs)
    real123_gen = generator(real123_recs)
    
    ordered_files = [next(real123_gen)]
    npr_count = 0
    synth_count = 0
    real_count = 1
    
    print('synth_recs:', str(len(synth_recs)), 'npr_recs:', str(len(npr_recs)), 'real123_recs:', str(num_real_recs))
    print('target_ratio:', str(target_ratio))
    for i in range(num_total_recs - 1):
        if real_count < num_real_recs and npr_count + synth_count < num_synth_recs:
            if (synth_count + npr_count) / real_count < target_ratio:
                npr_count, synth_count = choose_synth_rec(npr_recs, npr_count, synth_recs, synth_count, npr_gen, synth_gen, ordered_files)
            else:
                real_count += 1
                ordered_files.append(next(real123_gen))
        
        elif real_count < num_real_recs:
            real_count += 1
            ordered_files.append(next(real123_gen))
  
        elif npr_count + synth_count < num_synth_recs:
            npr_count, synth_count = choose_synth_rec(npr_recs, npr_count, synth_recs, synth_count, npr_gen, synth_gen, ordered_files)
        else:
            print('There is some problem. iteration_num:', i, '/', num_total_recs)
            print('real_count:', str(real_count), 'npr_count:', str(npr_count), 'synth_count:', str(synth_count))
            print('num_real_recs:', str(num_real_recs), 'num_synth_recs:', str(num_synth_recs))
        # print('real_count:', str(real_count), 'npr_count:', str(npr_count), 'synth_count:', str(synth_count))
    
    if not next(real123_gen, None): print('real123_gen has no files.')
    if not next(npr_gen, None): print('npr_gen has no files.')
    if not next(synth_gen, None): print('synth_gen has no files.')
    
    return ordered_files
    

def sprinkle_icdar15(ordered_files):
    icdar15_loc = '/mnt/data/Rohit/ACMData/4aicdarcomp/datasetoverlappingF/train_tf_records/'
    icdar15_cropped_loc = '/mnt/data/Rohit/ACMData/4aicdarcomp/datasetoverlappingF/crop_train_tf_records/'   
    icdar15_recs = list_tfrecords(icdar15_loc)
    icdar15crop_recs = list_tfrecords(icdar15_cropped_loc)
    icdar15_recs += icdar15crop_recs
    random.shuffle(icdar15_recs)
    icdar15_gen = generator(icdar15_recs)
    num_icdar15_recs = len(icdar15_recs)
    
    target_ratio = len(ordered_files) / num_icdar15_recs
    result = [next(icdar15_gen)]
    icdar15_count = 1
    
    ordered_files_gen = generator(ordered_files)
    ordered_files_count = 0
    num_ordered_files = len(ordered_files)
    
    for i in range(len(ordered_files) + num_icdar15_recs - 1):
        if ordered_files_count < num_ordered_files and icdar15_count < num_icdar15_recs:
            if (ordered_files_count) / icdar15_count < target_ratio:
                result.append(next(ordered_files_gen))
                ordered_files_count += 1
            else:
                result.append(next(icdar15_gen))
                icdar15_count += 1
        
        elif ordered_files_count < num_ordered_files:
            result.append(next(ordered_files_gen))
            ordered_files_count += 1
  
        elif icdar15_count < num_icdar15_recs:
            result.append(next(icdar15_gen))
            icdar15_count += 1
        else:
            print('There is some problem. iteration_num:', i, '/', num_ordered_files+num_icdar15_recs)
            print('ordered_files_count:', str(ordered_files_count), 'icdar15_count:', str(icdar15_count))
            print('num_icdar15_recs:', str(num_icdar15_recs), 'num_ordered_files:', str(num_ordered_files))
    
    if not next(ordered_files_gen, None): print('ordered_files_gen has no files.')
    if not next(icdar15_gen, None): print('icdar15_gen has no files.')
    
    return result
    

def overlay_files(gen1, num1, gen2, num2):
    target_ratio = num1 / num2
    result = [next(gen2)]
    
    count_1 = 0
    count_2 = 1
    
    for i in range(num1 + num2 - 1):
        if count_1 < num1 and count_2 < num2:
            if count_1 / count_2 < target_ratio:
                result.append(next(gen1))
                count_1 += 1
            else:
                result.append(next(gen2))
                count_2 += 1
        elif count_1 < num1:
            result.append(next(gen1))
            count_1 += 1
        elif count_2 < num2:
            result.append(next(gen2))
            count_2 += 1
        else:
            print('There is some problem in iteration', str(i), '.')
            
    if not next(gen1, None): print('gen1 has no files.')
    if not next(gen2, None): print('gen2 has no files.')
    
    return result
    
    
def get_real_files_mix():  
    icdar15_loc = '/mnt/data/Rohit/ACMData/4aicdarcomp/datasetoverlappingF/train_tf_records/'
    icdar15_cropped_loc = '/mnt/data/Rohit/ACMData/4aicdarcomp/datasetoverlappingF/crop_train_tf_records/'   
    icdar15_recs = list_tfrecords(icdar15_loc)
    icdar15crop_recs = list_tfrecords(icdar15_cropped_loc)
    icdar15_recs += icdar15crop_recs
    random.shuffle(icdar15_recs)
    icdar15_gen = generator(icdar15_recs)
    num_icdar15_recs = len(icdar15_recs)
    
    real123_loc = '/mnt/data/Rohit/ACMData/tftrainallFinal/mixed_data/mix1_ready/'
    real123_recs = list_tfrecords(real123_loc)
    real123_gen = generator(real123_recs)
    num_real_recs = len(real123_recs)
    
    real_files = overlay_files(real123_gen, num_real_recs, icdar15_gen, num_icdar15_recs)
    
    return real_files


def get_synth_files_mix():
    npr_loc = '/mnt/data/Rohit/ACMData/5aSynthVideosE2E/TrainingDataRecordsvideosNPR/'
    synth_loc = '/mnt/data/Rohit/ACMData/5aSynthVideosE2E/TrainingDataRecordsvideos/'
    npr_recs = list_tfrecords(npr_loc)
    synth_recs = list_tfrecords(synth_loc)
    npr_gen = generator(npr_recs)
    synth_gen = generator(synth_recs)
    
    return overlay_files(npr_gen, len(npr_recs), synth_gen, len(synth_recs))
              
   
def rename_files(files):
    num_files = len(files)
    trailing_zeros = ceil(log(num_files, 10))
    result = []
    for idx, file in enumerate(files):
        head_tail = os.path.split(file)
        file = head_tail[1]
        file = str(str(idx).zfill(trailing_zeros)) + '_' + file
        dir = head_tail[0]
        result.append(os.path.join(dir, file))
    return result

   
if __name__ == "__main__":
    # ordered_files = mix_synth_real()
    # ordered_files = sprinkle_icdar15(ordered_files)
    # 
    # output_to_file('systematic_mix.txt', ordered_files)
    
    real_files = get_real_files_mix()
    synth_files = get_synth_files_mix()
    ordered_files = overlay_files(generator(real_files), len(real_files), generator(synth_files), len(synth_files))
    renamed_ordered_files = rename_files(ordered_files)
    
    result = []
    for src, dest in zip(ordered_files, renamed_ordered_files):
        os.rename(src, dest)
        result.append(src + ' -> ' + dest)
        
    output_to_file('rename_result.txt', result)

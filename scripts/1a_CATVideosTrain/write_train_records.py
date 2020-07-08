EXAMPLES_PER_FILE = 100
CLIPS_PER_VIDEO = 10
CLIP_LEN = 8
CHARSET = {'a': 0, 'b': 1, 'c': 2}
FOURCC = VideoWriter_fourcc(*'MP42')

from math import ceil, log

def check_rotation(video_file):
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(video_file)

    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotateCode = None
    try:
        if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
            rotateCode = cv2.ROTATE_90_CLOCKWISE
        elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
            rotateCode = cv2.ROTATE_180
        elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
            rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE
    except Exception as e:
        print(e)
        
    return rotateCode


def encode_utf8_string(text='abc', charset=CHARSET, length=5, null_char_id=82):
    char_ids_padded = []
    char_ids_unpadded = []
    for i in range(0, len(text)):
        char_ids_unpadded.append(charset[text[i]])
        char_ids_padded.append(charset[text[i]])
    for i in range(len(text), length):
        char_ids_padded.append(null_char_id)
    return char_ids_padded, char_ids_unpadded


def read_charset(filename, null_character=u'\u2591'):
    """Reads a charset definition from a tab separated text file.

    charset file has to have format compatible with the FSNS dataset.

    Args:
      filename: a path to the charset file.
      null_character: a unicode character used to replace '<null>' character. the
        default value is a light shade block .

    Returns:
      a dictionary with keys equal to character codes and values - unicode
      characters.
    """
    pattern = re.compile(r'(\d+)\t(.+)')
    charset = {}
    with tf.gfile.GFile(filename) as f:
        for i, line in enumerate(f):
            m = pattern.match(line)
            if m is None:
                # logging.warning('incorrect charset file. line #%d: %s', i, line)
                print("incorrect charset file at", i, line)
                continue
            code = int(m.group(1))
            print(m.group(2))
            char = m.group(2).decode('utf-8')
            if char == '<nul>':
                char = null_character
            charset[char] = code  # charset[code] = char
    return charset


def read_video(line):
    video_file = line[0]
    cap = cv2.VideoCapture(video_file)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    fromFrame = int(float(line[3])*fps) if len(line) > 3 else 0
    toFrame = int(float(line[4])*fps) if len(line) > 4 else None
    cap.set(cv2.CAP_PROP_POS_FRAMES, fromFrame-1)
    
    rotateCode = check_rotation(video_file)
    
    frame_num = 0
    video = []
    while(cap.isOpened()):
        if (to_frame is not None) and (frame_num > (to_frame - from_frame)): break
        
        status, frame = cap.read()
        
        if status:
            if rotateCode:
                frame = cv2.rotate(frame, rotateCode)
            
            h, w, c = frame.shape
            if not (h == 256 and w == 480): 
                frame = cv2.resize(frame, (480, 256), interpolation=cv2.INTER_AREA)

            video.append(frame)    
        else:
            break
        
    cap.release()
    
    return video


def write_video(label, video, video_no, transform)
    out_file = "CATVideosTrain/Videos/"+ transform + str(video_no)+ "_realSigns_" + label.replace(" ","_").replace("/","_by_").encode("utf-8") + ".avi"
    writer = VideoWriter(out_file, FOURCC, float(fpsSigns),(480, 256))
    for frame in video: writer.write(frame)
    writer.release()
        

def create_clips(video, overlap=2, step_size=1, discard_extra=True):
    retVal = []

    for i in range(0, len(video), (CLIP_LEN-overlap)*step_size):
        for j in range(step_size):
            clip = video[i+j: CLIP_LEN*step_size+i: step_size]
            if discard_extra:
                retVal.append(clip)
            elif len(clip) > 4:
                clip += [np.zeros((260, 480, 3), dtype=np.int32)] * (CLIP_LEN - len(clip))
                retVal.append(clip)
    return retVal


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_featureL(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _int64_featureMask(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value.reshape(-1)))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_example(label, clip, transformation=7, video_no, clip_num, maxlength=180, null_char_id=1, flag_synth=1):
    char_ids_padded, char_ids_unpadded = encode_utf8_string(text=label, length=maxlength, null_char_id=null_char_id)
    
    im_h = cv2.hconcat(clip)
    charMasks1 = np.array([])
    wordMasks1 = np.array([])
    lineMasks1 = np.array([])
    print(im_h.shape,charMasks1.shape,wordMasks1.shape,lineMasks1.shape)
    
    h, w, c = im_h.shape
    if not(h == 256 and w == 480*CLIP_LEN):
        print("RESIZING BEFORE TFRECORD WRITING", label, im_h.shape)
        im_h = cv2.resize(im_h, (480*CLIP_LEN, 256), interpolation=cv2.INTER_AREA)
    
    _, img = cv2.imencode('.png', im_h)
    
    example = tf.train.Example(features=tf.train.Features(
        feature={'image/format': _bytes_feature(b"png"), 
                 'image/encoded': _bytes_feature(img.tostring()),
                 'image/class': _int64_featureL(char_ids_padded),
                 'image/unpadded_class': _int64_featureL(char_ids_unpadded),
                 'image/height': _int64_feature(im_h.shape[0]),
                 'image/width': _int64_feature(im_h.shape[1]),
                 'image/flag_synth': _int64_feature(flag_synth),
                 'image/orig_width': _int64_feature(480),
                 'image/text': _bytes_feature(label.encode("utf-8")),
                 'image/charBB': _int64_featureMask((charMasks1 == 1)),
                 'image/wordBB': _int64_featureMask((wordMasks1 == 1)),
                 'image/lineBB': _int64_featureMask((lineMasks1 == 1)),
                 'image/transformation': _int64_feature(transformation),
                 'image/video_no': _int64_feature(video_no), 
                 'image/frame_no': _int64_feature(clip_num)}))
    return example


def write_tfrecord(clip_num, tfrecord_num, examples):
    out_file = 'real_'+str(i).zfill(ceil(log(CLIPS_PER_VIDEO)))+'_'+str(tfrecord_num).zfill(3)+'.tfrecords'
    
    writer = tf.python_io.TFRecordWriter(out_file)
    for example in examples:
        writer.write(example.SerializeToString())
    writer.close()
    
    
if __name__ == "__main__":
    CHARSET = read_charset('charset_size=134.txt')
    
    tfrecords = {}
    for i in range(CLIPS_PER_VIDEO):
        tfrecords[i] = {'filename': 'real_'+str(i)+'_000.tfrecords',
                        'count': 0,
                        'examples': []
                        }
        
    with open('2ShubhamValclean', 'r') as fp:
        for line_idx, line in enumerate(fp):
            line = line.strip().split("\t")
            
            video_label = line[1].decode('utf-8')
            transform = line[2]+'_' if len(line) > 2 else ''
            
            video = read_video(line)
            
            if video:
                video_num = 1000+line_idx
                clips = create_clips(video, discard_extra=False)
                
                for i, clip in enumerate(clips):
                    example = create_example(video_label, clip, video_no=video_num, clip_num=i, flag_synth=0)
                    tfrecords[i%CLIPS_PER_VIDEO]['examples'].append(example)
                    
                    while len(tfrecords[i%CLIPS_PER_VIDEO]['examples']) >= EXAMPLES_PER_FILE:
                        examples = tfrecords[i%CLIPS_PER_VIDEO]['examples'][:EXAMPLES_PER_FILE]
                        
                        tfrecords[i%CLIPS_PER_VIDEO]['examples'] = tfrecords[i%CLIPS_PER_VIDEO]['examples'][EXAMPLES_PER_FILE:]
                        
                        write_tfrecord(clip_num, count, examples)
                
                save_video(video, video_label, video_num, transform)


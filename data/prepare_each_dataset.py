import os 
import json
import csv
import yaml
from collections import defaultdict
import pickle
import glob
import math
from functools import partial
import sys
import io
import warnings
import random

import numpy as np
import torch
import laion_clap

import librosa
from pydub import AudioSegment
import soundfile as sf

import faiss

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

try:
    from tqdm import tqdm 
except:
    tqdm = lambda x: x


def suppress_all_output(func):
    def wrapper(*args, **kwargs):
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        old_fd_out = os.dup(1)
        old_fd_err = os.dup(2)
        null_fd = os.open(os.devnull, os.O_RDWR)
        
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                result = func(*args, **kwargs)
            finally:
                os.dup2(old_fd_out, 1)
                os.dup2(old_fd_err, 2)
                os.close(null_fd)
                os.close(old_fd_out)
                os.close(old_fd_err)
        
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        return result
    return wrapper


def filter_file(file_path, file_list, filename):
    if file_list is not None:
        if filename not in file_list:
            print(filename, 'not exist')
            return True 
    else:
        if not os.path.exists(os.path.join(file_path, filename)):
            print(filename, 'not exist')
            return True 

    if os.path.getsize(os.path.join(file_path, filename)) < 16000:
        print(filename, 'less than 0.5 to 1 second')
        return True
    
    return False


# ==================== Prepare dataset files from each data folder ====================

EMOTION_MAP_DICT = {
    'amused':       'amused'      , 
    'anger':        'angry'       , 'angry':        'angry'       , 
    'anxious':      'anxious'     , 
    'apologetic':   'apologetic'  , 
    'assertive':    'assertive'   ,
    'calm':         'calm'        , 
    'concerned':    'concerned'   , 
    'contempt':     'contempt'    , 
    'disgust':      'disgusted'   , 'disgusted':    'disgusted'   , 
    'encouraging':  'encouraging' , 
    'excited':      'excited'     , 
    'fear':         'fearful'     , 'fearful':      'fearful'     , 
    'frustated':    'frustated'   ,
    'happy':        'happy'       , 'joy':          'happy'       , 
    'neutral':      'neutral'     , 
    'sad':          'sad'         , 'sadness':      'sad'         , 
    'sleepy':       'sleepy'      , 
    'surprise':     'surprised'   , 'surprised':    'surprised'   ,
    'pleasantly surprised': 'pleasantly surprised' ,
}

def load_dataset_file(dataset_file):
    with open(dataset_file) as f:
        contents = f.read()
    contents = json.loads(contents)

    audio_files = [
        os.path.join(
            contents["dataset_path"],
            contents["split_path"],
            contents["data"][str(i)]["name"]
        ) for i in range(contents["total_num"])
    ]

    return contents, audio_files


def compute_label_graph(dataset_name, dataset_path, top_n, output_file):
    if os.path.exists(output_file):
        print('loading precomputed graph:', output_file)
        with open(output_file, 'r') as json_file:
            graph = json.load(json_file)
            
    else:
        import torch
        from sentence_transformers import SentenceTransformer, util
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        print('precomputing graph and save to:', output_file)

        if dataset_name == 'AudioSetSL_singlelabel':
            names = []
            with open(os.path.join(dataset_path, 'class_labels_indices.csv'), newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='"')
                next(reader)
                for row in reader:
                    _, label, name = row  # 123, /m/02zsn, "Female speech, woman speaking"
                    names += name.split(', ')
            names = [x.lower().strip() for x in names]

        elif dataset_name == "Clotho-AQA_singlelabel":
            names = set([])
            with open(os.path.join(dataset_path, 'clotho_aqa_metadata.csv'), newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='"')
                next(reader)
                for row in tqdm(reader):
                    _, file_name, keywords, _, _, _, _ = row
                    names |= set(keywords.split(';'))
            names = [x.lower().strip() for x in names]

        names_embeddings = embedding_model.encode(names, convert_to_tensor=True)
        similarity_matrix = util.pytorch_cos_sim(names_embeddings, names_embeddings)

        similarity_threshold = 0.75
        n_items = len(names)
        
        graph = {}
        for i in range(n_items):
            adjusted_top_n = min(top_n, n_items - 1)
            values, indices = torch.topk(similarity_matrix[i], adjusted_top_n + 1, largest=True)

            most_similar_items = []
            for value, idx in zip(values, indices):
                if idx != i and value <= similarity_threshold:
                    most_similar_items.append(idx.item())
                if len(most_similar_items) == adjusted_top_n:
                    break
            graph[names[i]] = [names[j] for j in most_similar_items]

        with open(output_file, 'w') as json_file:
            json.dump(graph, json_file)
    
    # graph is a dict: key = each label, value = List[20 similar labels]
    return graph


def prepare_files(dataset_name, dataset_path, split, flamingo_task, output_file):
    
    assert not os.path.exists(output_file)
    dataset_dic = {
        "dataset_path": dataset_path,
        "split": split,
        "split_path": None,
        "flamingo_task": "{}-{}".format(dataset_name, flamingo_task),
        "total_num": 0,
        "data": {}  # {id: {'name': name, 'prompt': prompt, 'output': output}}
    }

    if dataset_name == "AudioSet":
        assert flamingo_task == "EventClassification"

        assert split == 'train'
        map_split = lambda split: 'train_wav' if split == 'train' else ''
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = list(filter(lambda x: x.endswith('.wav'), os.listdir(file_path)))

        dic = defaultdict(str)
        with open(os.path.join(dataset_path, 'class_labels_indices.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader)
            for row in tqdm(reader):
                _, label, name = row  # /m/02zsn,"Female speech, woman speaking"
                dic[label] = name
        
        with open(os.path.join(dataset_path, 'train.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader)
            for row in tqdm(reader):
                filename, _, _, labels = row  # --aE2O5G5WE /m/03fwl,/m/04rlf,/m/09x0r 
                filename = filename + '.wav'
                if filter_file(file_path, file_list, filename):
                    continue
                    
                label_list = labels.split(",")
                assert all(label in dic for label in label_list)

                text_output = ", ".join([dic[label] for label in label_list])
                if len(text_output) <= 1:
                    continue
                text_prompt = 'this is a sound of'

                dataset_dic["data"][dataset_dic["total_num"]] = {
                    "name": filename,
                    "prompt": text_prompt,
                    "output": text_output.replace('\n', ' ')
                }
                dataset_dic["total_num"] += 1
    
    elif dataset_name == "AudioSetFull":
        assert flamingo_task == "EventClassification"

        assert split == 'train'
        map_split = lambda split: '/mnt/fsx-main/rafaelvalle/datasets/audioset/unbalanced_train_segments/22khz'
        file_path = map_split(split)
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = None

        dic_code2label = defaultdict(str)
        with open(os.path.join(dataset_path, 'audioset-processing/data/class_labels_indices.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader)
            for row in tqdm(reader):
                _, code, name = row  # /m/02zsn,"Female speech, woman speaking"
                dic_code2label[code] = name
        
        dic_filename2code = {}
        with open(os.path.join(dataset_path, 'audioset-processing/data/unbalanced_train_segments.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader)
            next(reader)
            for row in tqdm(reader):
                filename, _, _, codes = row  # --aE2O5G5WE /m/03fwl,/m/04rlf,/m/09x0r 
                filename = 'Y' + filename + '.wav'
                dic_filename2code[filename] = codes.split(",")

        for part in tqdm(range(41)):
            part_str = str(part)
            if len(part_str) == 1:
                part_str = '0' + part_str
            part_folder = 'unbalanced_train_segments_part{}'.format(part_str)

            for filename in os.listdir(os.path.join(file_path, part_folder)):
                if not filename.endswith('.wav'):
                    continue 

                if filter_file(file_path, file_list, os.path.join(part_folder, filename)):
                    continue
                
                if filename not in dic_filename2code:
                    continue 

                text_output = ", ".join([dic_code2label[code] for code in dic_filename2code[filename] if code in dic_code2label])
                if len(text_output) <= 1:
                    continue
                text_prompt = 'this is a sound of'

                dataset_dic["data"][dataset_dic["total_num"]] = {
                    "name": os.path.join(part_folder, filename),
                    "prompt": text_prompt,
                    "output": text_output.replace('\n', ' ')
                }
                dataset_dic["total_num"] += 1
    
    elif dataset_name == "AudioSetFullwoAudioMusicCaps":
        assert flamingo_task == "EventClassification"

        assert split == 'train'
        map_split = lambda split: '/mnt/fsx-main/rafaelvalle/datasets/audioset/unbalanced_train_segments/22khz'
        file_path = map_split(split)
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = None

        print('extracting AudioCaps and MusicCaps ytid to avoid these samples')
        audiocaps_ytid = []
        for f in ['audiocaps_dataset/train.csv', 'audiocaps_dataset/test.csv', 'audiocaps_dataset/val.csv']:
            with open(os.path.join(dataset_path, f), newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='"')
                next(reader)
                for row in reader:
                    _, ytid, _, _ = row 
                    audiocaps_ytid.append('Y' + ytid + '.wav')
        audiocaps_ytid = set(audiocaps_ytid)
        
        musiccaps_ytid = []
        with open(os.path.join(dataset_path, 'musiccaps_dataset/musiccaps_manifest.json')) as f:
            data = f.read()
        musiccaps_list = json.loads(data)
        for row in musiccaps_list:
            musiccaps_ytid.append('Y' + row["ytid"] + '.wav')
        musiccaps_ytid = set(musiccaps_ytid)

        print('Will exclude {} samples from MusicCaps and {} from AudioCaps'.format(len(audiocaps_ytid), len(musiccaps_ytid)))

        dic_code2label = defaultdict(str)
        with open(os.path.join(dataset_path, '../AudioSetFull/audioset-processing/data/class_labels_indices.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader)
            for row in tqdm(reader):
                _, code, name = row  # /m/02zsn,"Female speech, woman speaking"
                dic_code2label[code] = name
        
        dic_filename2code = {}
        with open(os.path.join(dataset_path, '../AudioSetFull/audioset-processing/data/unbalanced_train_segments.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader)
            next(reader)
            for row in tqdm(reader):
                filename, _, _, codes = row  # --aE2O5G5WE /m/03fwl,/m/04rlf,/m/09x0r 
                filename = 'Y' + filename + '.wav'
                dic_filename2code[filename] = codes.split(",")

        music_audio_caps_excluded = 0
        for part in tqdm(range(41)):
            part_str = str(part)
            if len(part_str) == 1:
                part_str = '0' + part_str
            part_folder = 'unbalanced_train_segments_part{}'.format(part_str)

            for filename in os.listdir(os.path.join(file_path, part_folder)):
                if not filename.endswith('.wav'):
                    continue 

                if filename in audiocaps_ytid or filename in musiccaps_ytid:
                    music_audio_caps_excluded += 1
                    continue

                if filter_file(file_path, file_list, os.path.join(part_folder, filename)):
                    continue
                
                if filename not in dic_filename2code:
                    continue 

                text_output = ", ".join([dic_code2label[code] for code in dic_filename2code[filename] if code in dic_code2label])
                if len(text_output) <= 1:
                    continue
                text_prompt = 'this is a sound of'

                dataset_dic["data"][dataset_dic["total_num"]] = {
                    "name": os.path.join(part_folder, filename),
                    "prompt": text_prompt,
                    "output": text_output.replace('\n', ' ')
                }
                dataset_dic["total_num"] += 1

    elif dataset_name == "AudioSetSL_singlelabel":
        import numpy as np 

        assert flamingo_task == "EventClassification"

        assert split == 'train'
        map_split = lambda split: '../AudioSet/train_wav'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = list(filter(lambda x: x.endswith('.wav'), os.listdir(file_path)))

        dic = defaultdict(str)
        with open(os.path.join(dataset_path, 'class_labels_indices.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader)
            for row in tqdm(reader):
                _, label, name = row  # /m/02zsn,"Female speech, woman speaking"
                dic[label] = name
        
        graph = compute_label_graph(
            dataset_name, 
            dataset_path, 
            top_n=200,
            output_file=os.path.join(dataset_path, 'label_graph.json')
        )
        
        with open(os.path.join(dataset_path, 'train.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader)
            for row in tqdm(reader):
                filename, _, _, labels = row  # --aE2O5G5WE /m/03fwl,/m/04rlf,/m/09x0r 
                filename = filename + '.wav'
                if filter_file(file_path, file_list, filename):
                    continue
                    
                label_list = labels.split(",")
                assert all(label in dic for label in label_list)

                text_labels = ", ".join([dic[label] for label in label_list]).lower()
                text_labels = text_labels.split(', ')
                text_output = np.random.choice(text_labels)
                if len(text_output) <= 1:
                    continue

                num_options = np.random.choice(
                    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                    p=[ 0.05, 0.1, 0.1, 0.1, 0.1, 
                        0.05, 0.05, 0.05, 0.1, 0.05, 
                        0.05, 0.1, 0.05, 0.05]
                )

                negative_samples = [x for x in graph[text_output] if x not in set(text_labels)]
                candidate_negative_labels = list(np.random.choice(
                    negative_samples[:num_options*10],
                    size=num_options-1, 
                    replace=False
                ))
                if type(candidate_negative_labels) is str:
                    candidate_negative_labels = [candidate_negative_labels]

                all_options = [text_output] + candidate_negative_labels
                np.random.shuffle(all_options)

                text_prompt = 'Classify this sound.\nOPTIONS:\n - {}.'.format(
                    '.\n - '.join(all_options)
                )

                dataset_dic["data"][dataset_dic["total_num"]] = {
                    "name": filename,
                    "prompt": text_prompt,
                    "output": text_output.replace('\n', ' ')
                }
                dataset_dic["total_num"] += 1
    
    elif dataset_name == "AUDIOCAPS13k":
        assert flamingo_task == 'AudioCaptioning'

        map_split = lambda split: 'audio_32000Hz/{}'.format(split)
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = list(filter(lambda x: x.endswith('.flac'), os.listdir(file_path)))

        with open(os.path.join(
            dataset_path,
            '{}_manifest.json'.format(split + ('_v2' if split == 'train' else ''))
        ), 'r') as f:
            data = f.readlines()
        data = [json.loads(row) for row in data]

        for row in tqdm(data):
            filename = row['audio_filepath'].split('/')[-1]
            if filter_file(file_path, file_list, filename):
                continue
            
            text_output = row['text']
            if len(text_output) <= 1:
                continue
            text_prompt = 'generate audio caption'

            dataset_dic["data"][dataset_dic["total_num"]] = {
                "name": filename,
                "prompt": text_prompt,
                "output": text_output.replace('\n', ' ')
            }
            dataset_dic["total_num"] += 1

    elif dataset_name == "audiocaps":
        assert flamingo_task == 'AudioCaptioning'

        map_split = lambda split: 'audio/{}'.format(split if split in ['train', 'test'] else 'valid')
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = list(filter(lambda x: x.endswith('.flac'), os.listdir(file_path)))

        for filename in tqdm(file_list):
            if filter_file(file_path, file_list, filename):
                continue
            
            with open(os.path.join(file_path, filename.replace('.flac', '.json')), 'r') as f:
                data = json.load(f)
            
            captions = data['text']
            for text_output in captions:
                if len(text_output) <= 1:
                    continue
                text_prompt = 'generate audio caption'

                dataset_dic["data"][dataset_dic["total_num"]] = {
                    "name": filename,
                    "prompt": text_prompt,
                    "output": text_output.replace('\n', ' ')
                }
                dataset_dic["total_num"] += 1
    
    elif dataset_name == 'BG-Gun-Sound-Dataset':
        assert flamingo_task == "SoundClassification"
        assert split in ["train", "test"]
        
        map_split = lambda split: 'data/gun_sound_v2'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = os.listdir(file_path)

        all_cates = set([])
        with open(os.path.join(dataset_path, 'data/v3_exp3_{}.csv'.format(split)), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader)
            for row in tqdm(reader):
                filename, cate, dist, dire = row
                if filter_file(file_path, file_list, filename):
                    continue

                text_output = cate
                if len(text_output) <= 1:
                    continue
                text_prompt = 'What is the gun of this sound?'

                all_cates.add(cate)
                
                dataset_dic["data"][dataset_dic["total_num"]] = {
                    "name": filename,
                    "prompt": text_prompt,
                    "output": text_output.replace('\n', ' ')
                }
                dataset_dic["total_num"] += 1

        print(all_cates)

    elif dataset_name == "BirdsDataset":
        assert flamingo_task == "SoundClassification"
        assert split == 'train'

        map_split = lambda split: 'Voice_of_Birds'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = None

        for bird_type in tqdm(os.listdir(file_path)):
            bird_name = ' '.join(bird_type.split('_')[:-1])
            for filename in os.listdir(os.path.join(file_path, bird_type)):
                if filter_file(file_path, file_list, os.path.join(bird_type, filename)):
                    continue

                text_output = bird_name
                if len(text_output) <= 1:
                    continue
                text_prompt = 'What is the name of bird in this sound?'
                
                dataset_dic["data"][dataset_dic["total_num"]] = {
                    "name": os.path.join(bird_type, filename),
                    "prompt": text_prompt,
                    "output": text_output.replace('\n', ' ')
                }
                dataset_dic["total_num"] += 1

    elif dataset_name == "BBCSoundEffects":
        assert split in ['train']
        assert flamingo_task == 'AudioDescription'

        map_split = lambda split: '../WavCaps/BBC_Sound_Effects_flac'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = list(filter(lambda x: x.endswith('.flac'), os.listdir(file_path)))

        with open(os.path.join(dataset_path, 'BBCSoundDownloader/BBCSoundEffects.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader)
            for row in tqdm(reader):
                if len(row) != 7:
                    continue
                filename, description, _, _, _, _, _ = row
                filename = filename.replace('.wav', '.flac')

                if filter_file(file_path, file_list, filename):
                    continue
                
                text_output = description
                if len(text_output) <= 1:
                    continue
                text_prompt = 'generate audio description'

                dataset_dic["data"][dataset_dic["total_num"]] = {
                    "name": filename,
                    "prompt": text_prompt,
                    "output": text_output.replace('\n', ' ')
                }
                dataset_dic["total_num"] += 1
    
    elif dataset_name == "chime-home":
        assert flamingo_task == "EventClassification"
        assert split == 'train'

        map_split = lambda split: 'chime_home/chunks'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file48k_list = list(filter(lambda x: x.endswith('48kHz.wav'), os.listdir(file_path)))
        file16k_list = list(filter(lambda x: x.endswith('16kHz.wav'), os.listdir(file_path)))
        csv_file_list = list(filter(lambda x: x.endswith('.csv'), os.listdir(file_path)))

        label_mapping = {
            'c': 'child speaking',
            'm': 'male speaking',
            'f': 'female speaking',
            'p': 'human activity',
            't': 'television',
            'b': 'household appliances',
            's': 'silence'
        }

        for csv_file in tqdm(csv_file_list):
            with open(os.path.join(file_path, csv_file), newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='"')

                labels = None
                for row in reader:
                    if row[0] == 'majorityvote':
                        labels = row[1]
                        break 
                
            if labels is None or len(labels) == 0:
                continue 
            
            filename = csv_file.replace('.csv', '.48kHz.wav')
            if filter_file(file_path, file48k_list, filename):
                filename = csv_file.replace('.csv', '.16kHz.wav')
                if filter_file(file_path, file16k_list, filename):
                    continue

            text_output = ", ".join([label_mapping[l] for l in labels if l in label_mapping])
            if len(text_output) <= 1:
                continue
            text_prompt = 'this is a sound of'
            
            dataset_dic["data"][dataset_dic["total_num"]] = {
                "name": filename,
                "prompt": text_prompt,
                "output": text_output.replace('\n', ' ')
            }
            dataset_dic["total_num"] += 1

    elif dataset_name == "CLAP_freesound":
        assert flamingo_task == "AudioCaptioning"
        assert split in ["train", "test"]

        map_split = lambda split: os.path.join('freesound_no_overlap/split', split)
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = list(filter(lambda x: x.endswith('.flac'), os.listdir(file_path)))
        
        with open(os.path.join(
            dataset_path, 
            'freesound_no_overlap_meta.csv'
        ), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader)
            for row in tqdm(reader):
                if len(row[0].split('/')) != 2:
                    continue 
                if len(row) <= 1:
                    continue

                file_split, filename = row[0].split('/')

                if file_split != split:
                    continue 
                if filter_file(file_path, file_list, filename):
                    continue

                caption_1 = row[1]  # caption_2 = row[2] but not very good
                text_output = caption_1
                if len(text_output) <= 2:
                    continue

                text_prompt = 'generate audio caption'
                dataset_dic["data"][dataset_dic["total_num"]] = {
                    "name": filename,
                    "prompt": text_prompt,
                    "output": text_output.replace('\n', ' ')
                }
                dataset_dic["total_num"] += 1

    elif dataset_name == "Clotho-AQA":

        map_split = lambda split: 'audio_files'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = list(filter(lambda x: x.endswith('.wav'), os.listdir(file_path)))

        if flamingo_task == "EventClassification":
            dic = defaultdict(str)
            with open(os.path.join(dataset_path, 'clotho_aqa_metadata.csv'), newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='"')
                next(reader)
                for row in tqdm(reader):
                    _, file_name, keywords, _, _, _, _ = row
                    dic[file_name] = keywords.replace(';', ', ')

            with open(os.path.join(dataset_path, 'clotho_aqa_{}.csv'.format(split)), newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='"')
                next(reader)
                for row in tqdm(reader):
                    filename = row[0]
                    if filename not in dic or filter_file(file_path, file_list, filename):
                        continue

                    text_output = dic[filename]
                    if len(text_output) <= 1:
                        continue
                    text_prompt = 'this is a sound of'
                    del dic[filename]

                    dataset_dic["data"][dataset_dic["total_num"]] = {
                        "name": filename,
                        "prompt": text_prompt,
                        "output": text_output.replace('\n', ' ')
                    }
                    dataset_dic["total_num"] += 1
        
        elif flamingo_task == "AQA":
            dic_qa = defaultdict(list)
            with open(os.path.join(dataset_path, 'clotho_aqa_{}.csv'.format(split)), newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='"')
                next(reader)
                for row in tqdm(reader):
                    filename, question, answer, confidence = row
                    dic_qa[(filename, question)].append((answer.lower(), confidence.lower()))

            # get binary -> trinary
            def preprocess(list_ans_conf):
                assert set([x[1] for x in list_ans_conf]) <= set(['yes', 'no', 'maybe'])

                answers = set([x[0].lower() for x in list_ans_conf])
                if answers <= set(['yes', 'no']):
                    if len(answers) > 1:
                        return ['unsure']
                    else:
                        return list(answers)
                else:
                    return list(answers)
            
            # get majority vote
            def majority_vote(list_ans_conf):
                assert set([x[1] for x in list_ans_conf]) <= set(['yes', 'no', 'maybe'])
                weight = {'yes': 1.0, 'no': 0.1, 'maybe': 0.6}

                if set([x[0] for x in list_ans_conf]) <= set(['yes', 'no']):
                    score = {'yes': 1.0, 'no': -1.0}
                    pred = sum([score[x[0]] * weight[x[1]] for x in list_ans_conf])
                    if pred > 0:
                        return ['yes']
                    else:
                        return ['no']
                else:
                    return list(set([x[0] for x in list_ans_conf]))

            for key in dic_qa:
                filename, question = key
                if filter_file(file_path, file_list, filename):
                    continue

                if split == 'train':
                    answers = majority_vote(dic_qa[key])  # majority vote
                else:
                    answers = [x[0].strip().lower() for x in dic_qa[key]]
                    answers = [', '.join(answers)]

                for answer in answers:
                    text_output = answer
                    if len(text_output) <= 1:
                        continue
                    text_prompt = "Question: " + question

                    dataset_dic["data"][dataset_dic["total_num"]] = {
                        "name": filename,
                        "prompt": text_prompt,
                        "output": text_output.replace('\n', ' ')
                    }
                    dataset_dic["total_num"] += 1
    
    elif dataset_name == "Clotho-AQA_singlelabel":
        import numpy as np 
        
        assert flamingo_task == "EventClassification"

        map_split = lambda split: '../Clotho-AQA/audio_files'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = list(filter(lambda x: x.endswith('.wav'), os.listdir(file_path)))

        dic = defaultdict(str)
        with open(os.path.join(dataset_path, 'clotho_aqa_metadata.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader)
            for row in tqdm(reader):
                _, file_name, keywords, _, _, _, _ = row
                dic[file_name] = keywords.split(';')
        
        graph = compute_label_graph(
            dataset_name, 
            dataset_path, 
            top_n=300,
            output_file=os.path.join(dataset_path, 'label_graph.json')
        )

        with open(os.path.join(dataset_path, 'clotho_aqa_{}.csv'.format(split)), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader)
            for row in tqdm(reader):
                filename = row[0]
                if filename not in dic or filter_file(file_path, file_list, filename):
                    continue

                text_labels = [x.lower().strip() for x in dic[filename]]
                del dic[filename]

                for _ in range(6):
                    text_output = np.random.choice(text_labels)
                    if len(text_output) <= 1:
                        continue

                    num_options = np.random.choice(
                        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                        p=[ 0.05, 0.1, 0.1, 0.1, 0.1, 
                            0.05, 0.05, 0.05, 0.1, 0.05, 
                            0.05, 0.1, 0.05, 0.05]
                    )

                    negative_samples = [x for x in graph[text_output] if x not in set(text_labels)]
                    candidate_negative_labels = list(np.random.choice(
                        negative_samples[:num_options*20],
                        size=num_options-1, 
                        replace=False
                    ))
                    if type(candidate_negative_labels) is str:
                        candidate_negative_labels = [candidate_negative_labels]

                    all_options = [text_output] + candidate_negative_labels
                    np.random.shuffle(all_options)

                    text_prompt = 'Classify this sound.\nOPTIONS:\n - {}.'.format(
                        '.\n - '.join(all_options)
                    )
                    
                    dataset_dic["data"][dataset_dic["total_num"]] = {
                        "name": filename,
                        "prompt": text_prompt,
                        "output": text_output.replace('\n', ' ')
                    }
                    dataset_dic["total_num"] += 1

    elif dataset_name == "Clotho-v2":
        assert flamingo_task == "AudioCaptioning"
        assert split in ["train", "val", "test"]

        map_split = lambda split: 'development' if split == 'train' else ('validation' if split == "val" else "evaluation")
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = list(filter(lambda x: x.endswith('.wav'), os.listdir(file_path)))

        with open(os.path.join(
            dataset_path, 
            'clotho_captions_{}.csv'.format(map_split(split))
        ), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader)
            for row in tqdm(reader):
                filename = row[0]
                if filter_file(file_path, file_list, filename):
                    continue

                for text_output in row[1:]:
                    if len(text_output) <= 1:
                        continue
                    text_prompt = 'generate audio caption'
                    dataset_dic["data"][dataset_dic["total_num"]] = {
                        "name": filename,
                        "prompt": text_prompt,
                        "output": text_output.replace('\n', ' ')
                    }
                    dataset_dic["total_num"] += 1
    
    elif dataset_name == "CochlScene":
        import ndjson
        assert flamingo_task == "SceneClassification"

        map_split = lambda split: split.capitalize()
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = None

        with open(os.path.join(dataset_path, 'cochlscene_{}.ndjson'.format(split))) as ndjsonfile:
            reader = ndjson.load(ndjsonfile)
            for row in tqdm(reader):
                filename = "/".join(row["audiopath"].split("/")[1:])
                if filter_file(file_path, file_list, filename):
                    continue

                text_output = row["labels"].lower()
                if len(text_output) <= 1:
                    continue
                text_prompt = 'this acoustic scene is'

                dataset_dic["data"][dataset_dic["total_num"]] = {
                    "name": filename,
                    "prompt": text_prompt,
                    "output": text_output.replace('\n', ' ')
                }
                dataset_dic["total_num"] += 1
    
    elif dataset_name == "common-accent":
        import ndjson 
        import re

        assert flamingo_task == "AccentClassification"
        assert split in ["train", "test"]

        map_split = lambda split: '22khz'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = os.listdir(file_path)

        all_accent = []
        split_file = [f for f in os.listdir(dataset_path) if f.startswith(split) and f.endswith('.ndjson')][0]
        with open(os.path.join(dataset_path, split_file)) as ndjsonfile:
            reader = ndjson.load(ndjsonfile)
            for row in tqdm(reader):
                accent = row["accent"]
                accent = re.sub(r'\(.*?\)', '', accent)
                accent = accent.replace('English', '')
                accent = accent.split(',')
                accent = [x.strip() for x in accent if 'school' not in x]
                all_accent += accent

                filename = row["filename"]
                if filter_file(file_path, file_list, filename):
                    continue

                for accent_each in accent:
                    if accent_each == 'Javanese':
                        accent_each = 'Japanese'
                    if len(accent_each) > 25:
                        continue 

                    text_output = accent_each
                    if len(text_output) <= 1:
                        continue
                    text_prompt = 'Classify the accent of this speech.'
                    
                    dataset_dic["data"][dataset_dic["total_num"]] = {
                        "name": filename,
                        "prompt": text_prompt,
                        "output": text_output.replace('\n', ' ')
                    }
                    dataset_dic["total_num"] += 1

        print('all accents:', list(set(all_accent)))

    elif dataset_name == "CREMA-D":
        assert flamingo_task == "EmotionClassification"
        assert split in ["train"]

        map_split = lambda split: 'AudioWAV'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = list(filter(lambda x: x.endswith('.wav'), os.listdir(file_path)))

        split_file = os.path.join(
            dataset_path, 
            'crema-d_audiopath_text_sid_emotion_filelist.txt'
        )
        with open(split_file, 'r') as f:
            data = f.readlines()
        data = [x.replace('\n', '') for x in data]

        for row in tqdm(data):
            if row.count('|') != 3:
                continue
            filename, utterances, speaker, emotion = row.split('|')
            if filter_file(file_path, file_list, filename):
                continue
                
            text_output = emotion
            text_prompt = 'this emotion is'
            
            if len(text_output) <= 1:
                continue

            dataset_dic["data"][dataset_dic["total_num"]] = {
                "name": filename,
                "prompt": text_prompt,
                "output": text_output.replace('\n', ' ')
            }
            dataset_dic["total_num"] += 1
    
    elif dataset_name == "DCASE17Task4":
        assert flamingo_task == "SceneClassification"
        assert split in ["test"]

        map_split = lambda split: 'unbalanced_train_segments_testing_set_audio_formatted_and_segmented_downloads'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = list(filter(lambda x: x.endswith('.wav'), os.listdir(file_path)))

        split_file = os.path.join(
            dataset_path, 
            'Task-4-Large-scale-weakly-supervised-sound-event-detection-for-smart-cars',
            'groundtruth_release',
            'groundtruth_strong_label_testing_set.csv'
        )

        dic = defaultdict(list)
        all_labels = []
        with open(split_file, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t', quotechar='"')
            for row in tqdm(reader):
                filename = 'Y' + row[0]
                label = row[-1]

                if filter_file(file_path, file_list, filename):
                    continue

                dic[filename] += label.split(', ')
                all_labels += label.split(', ')
        
        print('all labels:\n', ', '.join(list(set(all_labels))))
        
        for filename in dic:
            text_output = ', '.join(list(set(dic[filename])))
            text_prompt = 'this acoustic scene is'
            
            if len(text_output) <= 1:
                continue

            dataset_dic["data"][dataset_dic["total_num"]] = {
                "name": filename,
                "prompt": text_prompt,
                "output": text_output.replace('\n', ' ')
            }
            dataset_dic["total_num"] += 1
    
    elif dataset_name == "emov-db":
        assert flamingo_task == "EmotionClassification"
        assert split in ["train", "val"]

        map_split = lambda split: '22khz_from_16khz'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = list(filter(lambda x: x.endswith('.wav'), os.listdir(file_path)))

        split_file = os.path.join(
            dataset_path, 
            'cleaned_emov_db_audiopath_text_sid_emotion_duration_filelist_merged_{}.txt'.format(split)
        )
        with open(split_file, 'r') as f:
            data = f.readlines()
        data = [x.replace('\n', '') for x in data]

        for row in tqdm(data):
            if row.count('|') != 4:
                continue
            filename, utterances, speaker, emotion, duration = row.split('|')
            if filter_file(file_path, file_list, filename):
                continue
                
            text_output = emotion
            text_output = EMOTION_MAP_DICT[text_output]
            text_prompt = 'this emotion is'
            
            if len(text_output) <= 1:
                continue

            dataset_dic["data"][dataset_dic["total_num"]] = {
                "name": filename,
                "prompt": text_prompt,
                "output": text_output.replace('\n', ' ')
            }
            dataset_dic["total_num"] += 1
    
    elif dataset_name == "Epidemic_sound":
        assert split == 'train'
        assert flamingo_task in ["AudioCaptioning", "Tagging"]

        map_split = lambda split: 'audio'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = list(filter(lambda x: x.endswith('.mp3'), os.listdir(file_path)))

        with open(os.path.join(dataset_path, 'Epidemic_all_debiased.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader)
            for row in tqdm(reader):
                if len(row) != 5:
                    continue 
                _, caption_1, caption_2, caption_t5, fileid = row
                filename = '{}.mp3'.format(fileid)
                if filter_file(file_path, file_list, filename):
                    continue

                if flamingo_task == "AudioCaptioning":
                    text_output = caption_t5
                    if len(text_output) <= 1:
                        continue
                    text_prompt = 'generate audio caption'

                    dataset_dic["data"][dataset_dic["total_num"]] = {
                        "name": filename,
                        "prompt": text_prompt,
                        "output": text_output.replace('\n', ' ')
                    }
                    dataset_dic["total_num"] += 1
                    
                elif flamingo_task == "Tagging":
                    if not caption_2.startswith('the sounds of'):
                        continue 
                    caption_2 = caption_2.replace('the sounds of ', '')
                    caption_2 = caption_2.replace(', and', ',')
                    if len(caption_2) < 2:
                        continue

                    tags = caption_2.split(', ')
                    tags = list(map(lambda x: x.replace("'", "").strip().lower(), tags))
                    text_output = '{}'.format(', '.join(tags))
                    if len(text_output) <= 1:
                        continue
                    text_prompt = 'generate tags'

                    dataset_dic["data"][dataset_dic["total_num"]] = {
                        "name": filename,
                        "prompt": text_prompt,
                        "output": text_output.replace('\n', ' ')
                    }
                    dataset_dic["total_num"] += 1
    
    elif dataset_name == "ESC50":
        assert flamingo_task in ["EventClassification"]
        assert split == 'train'

        map_split = lambda split: 'ESC-50-master/audio'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = list(filter(lambda x: x.endswith('.wav'), os.listdir(file_path)))

        with open(os.path.join(dataset_path, 'ESC-50-master/meta/esc50.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader)
            for row in tqdm(reader):
                if len(row) != 7:
                    continue

                filename, fold, target, category, esc10, src_file, take = row
                if filter_file(file_path, file_list, filename):
                    continue

                text_output = category.replace('_', ' ')
                text_prompt = 'classify this sound.'
                if len(text_output) <= 1:
                    continue 

                dataset_dic["data"][dataset_dic["total_num"]] = {
                    "name": filename,
                    "prompt": text_prompt,
                    "output": text_output.replace('\n', ' ')
                }
                dataset_dic["total_num"] += 1
    
    elif dataset_name == "FMA":
        import ast 

        assert flamingo_task in ["GenreClassification"]
        assert split == 'train'

        map_split = lambda split: 'fma_large'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = None

        with open(os.path.join(dataset_path, 'fma_metadata/raw_tracks.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader)
            for row in tqdm(reader):
                if len(row) != 39:
                    continue
                track_id,album_id,album_title,album_url, \
                    artist_id,artist_name,artist_url,artist_website, \
                    license_image_file,license_image_file_large, \
                    license_parent_id,license_title,license_url, \
                    tags,track_bit_rate,track_comments,track_composer, \
                    track_copyright_c,track_copyright_p,track_date_created,track_date_recorded, \
                    track_disc_number,track_duration,track_explicit,track_explicit_notes, \
                    track_favorites,track_file,track_genres,track_image_file,track_information, \
                    track_instrumental,track_interest,track_language_code, \
                    track_listens,track_lyricist,track_number,track_publisher,track_title,track_url = row
                
                l = len(str(track_id))
                if l <= 3:
                    filename = '{}/{}.mp3'.format(
                        '000',
                        '0'*(6-l)+str(track_id)
                    )
                else:
                    filename = '{}/{}.mp3'.format(
                        '0'*(6-l)+str(track_id)[:l-3],
                        '0'*(6-l)+str(track_id)
                    )
                if filter_file(file_path, file_list, filename):
                    continue

                if len(track_genres) == 0:
                    continue
                    
                track_genres = ast.literal_eval(track_genres)
                genres = ', '.join([dic['genre_title'].lower().strip() for dic in track_genres])
                text_output = genres + '.'

                text_prompt = "what is the genre of this music?"
                
                if len(text_output) <= 1:
                    continue 

                dataset_dic["data"][dataset_dic["total_num"]] = {
                    "name": filename,
                    "prompt": text_prompt,
                    "output": text_output.replace('\n', ' ')
                }
                dataset_dic["total_num"] += 1
    
    elif dataset_name == "FSD50k":
        import ndjson
        assert flamingo_task == "EventClassification"
        assert split in ["train", "test"]

        map_split = lambda split: '44khz/dev' if split == 'train' else '44khz/eval'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = list(filter(lambda x: x.endswith('.wav'), os.listdir(file_path)))

        with open(os.path.join(dataset_path, '{}.ndjson'.format(map_split(split).replace('44khz/', '')))) as ndjsonfile:
            reader = ndjson.load(ndjsonfile)
            for row in tqdm(reader):
                filename = row["filepath"].split("/")[1]
                if filter_file(file_path, file_list, filename):
                    continue

                labels = [x.replace("_", " ").lower() for x in row["labels"]]
                text_output = ", ".join(labels)
                if len(text_output) <= 1:
                    continue
                text_prompt = 'this is a sound of'

                dataset_dic["data"][dataset_dic["total_num"]] = {
                    "name": filename,
                    "prompt": text_prompt,
                    "output": text_output.replace('\n', ' ')
                }
                dataset_dic["total_num"] += 1
    
    elif dataset_name == "GTZAN":
        assert flamingo_task == "GenreClassification"
        assert split in ["train"]

        map_split = lambda split: 'gtzan/data/genres'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = None

        for genre in os.listdir(file_path):
            genre_wavs = [x for x in os.listdir(os.path.join(file_path, genre)) if x.endswith('.wav')]

            for genre_wav in genre_wavs:
                filename = os.path.join(genre, genre_wav)
                if filter_file(file_path, file_list, filename):
                    continue

                text_output = genre
                if len(text_output) <= 1:
                    continue
                text_prompt = 'What is the genre of this music?'

                dataset_dic["data"][dataset_dic["total_num"]] = {
                    "name": filename,
                    "prompt": text_prompt,
                    "output": text_output.replace('\n', ' ')
                }
                dataset_dic["total_num"] += 1
    
    elif dataset_name == "IEMOCAP":
        assert flamingo_task == "EmotionClassification"
        assert split in ["train", "test"]

        map_split = lambda split: 'IEMOCAP_full_release/16khz'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = None

        def read_this_ndjson(file_path):
            dic_list = []
            with open(file_path, 'r') as f:
                for line in f:
                    turn_name = line.split("'turn_name': ")[-1].split(',')[0].replace("'", "")
                    emotion = line.split("'emotion': ")[-1].split(',')[0].replace("'", "")
                    dic = {
                        'turn_name': turn_name,
                        'emotion': emotion
                    }
                    dic_list.append(dic)
            return dic_list

        all_emotions = []
        meta_files = [x for x in os.listdir(os.path.join(dataset_path, 'IEMOCAP_full_release/ndjson')) if x.endswith('.ndjson')]
        for meta_file in tqdm(meta_files):
            main_folder = meta_file.split('_')[0]
            sub_folder = (meta_file.split('.ndjson')[0])[len(main_folder)+1:]

            if split == "train" and main_folder == "Session5":
                continue
            elif split == "test" and main_folder != "Session5":
                continue

            metadata_list = read_this_ndjson(os.path.join(dataset_path, 'IEMOCAP_full_release/ndjson', meta_file))

            for dic in metadata_list:
                filename = os.path.join(main_folder, sub_folder, dic['turn_name']+'.wav')
                if filter_file(file_path, file_list, filename):
                    continue

                if dic['emotion'] in ['unknown', 'other']:
                    continue 

                text_output = dic['emotion']
                text_output = EMOTION_MAP_DICT[text_output]
                all_emotions.append(text_output)

                text_prompt = 'this emotion is'
        
                if len(text_output) <= 1:
                    continue

                dataset_dic["data"][dataset_dic["total_num"]] = {
                    "name": filename,
                    "prompt": text_prompt,
                    "output": text_output.replace('\n', ' ')
                }
                dataset_dic["total_num"] += 1
        
        print('all emotions:', list(set(all_emotions)))
    
    elif dataset_name == "jl-corpus":
        assert flamingo_task == "EmotionClassification"
        assert split in ["train", "val"]

        map_split = lambda split: '44khz'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = list(filter(lambda x: x.endswith('.wav'), os.listdir(file_path)))

        split_file = os.path.join(
            dataset_path, 
            'jl-corpus_audiopath_text_sid_emotion_duration_{}_filelist.txt'.format(split)
        )
        with open(split_file, 'r') as f:
            data = f.readlines()
        data = [x.replace('\n', '') for x in data]

        for row in tqdm(data):
            if row.count('|') != 4:
                continue
            filename, utterances, speaker, emotion, duration = row.split('|')
            if filter_file(file_path, file_list, filename):
                continue
                
            text_output = emotion
            text_output = EMOTION_MAP_DICT[text_output]
            text_prompt = 'this emotion is'
            
            if len(text_output) <= 1:
                continue

            dataset_dic["data"][dataset_dic["total_num"]] = {
                "name": filename,
                "prompt": text_prompt,
                "output": text_output.replace('\n', ' ')
            }
            dataset_dic["total_num"] += 1
    
    elif dataset_name == "LP-MusicCaps-MC":
        import pandas as pd
        assert flamingo_task in ["AudioCaptioning"]
        assert split in ["train", "test"]

        map_split = lambda split: '../MusicCaps/44khz'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = list(filter(lambda x: x.endswith('.wav'), os.listdir(file_path)))

        parquet_files = [f for f in os.listdir(os.path.join(dataset_path, 'data')) if f.endswith('.parquet') and f.startswith(split)]
        print('parquet_files', parquet_files)
        metadata_df = pd.concat([pd.read_parquet(os.path.join(dataset_path, 'data', f)) for f in parquet_files])

        for index, row in tqdm(metadata_df.iterrows()):
            filename = row['ytid'] + '.wav'
            if filter_file(file_path, file_list, filename):
                continue
            
            text_prompt = 'generate audio caption'
            for caption in [row['caption_writing'], row['caption_summary'], row['caption_paraphrase']]:
                text_output = caption

                if len(text_output) <= 1:
                    continue
                    
                dataset_dic["data"][dataset_dic["total_num"]] = {
                    "name": filename,
                    "prompt": text_prompt,
                    "output": text_output.replace('\n', ' ')
                }
                dataset_dic["total_num"] += 1

    elif dataset_name == "LP-MusicCaps-MSD":
        import pandas as pd
        assert flamingo_task in ["AudioCaptioning"]
        assert split in ["train", "test", "val"]

        map_split = lambda split: '../MSD/mp3s_22khz'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = None

        parquet_files = [f for f in os.listdir(dataset_path) if f.endswith('.parquet') and f.startswith(split)]
        print('parquet_files', parquet_files)
        metadata_df = pd.concat([pd.read_parquet(os.path.join(dataset_path, f)) for f in parquet_files])

        for index, row in tqdm(metadata_df.iterrows()):
            filename = row['path']
            if filter_file(file_path, file_list, filename):
                continue
            
            text_prompt = 'generate audio caption'
            for caption in [row['caption_writing'], row['caption_summary'], row['caption_paraphrase']]:
                text_output = caption

                if len(text_output) <= 1:
                    continue
                    
                dataset_dic["data"][dataset_dic["total_num"]] = {
                    "name": filename,
                    "prompt": text_prompt,
                    "output": text_output.replace('\n', ' ')
                }
                dataset_dic["total_num"] += 1
    
    elif dataset_name == "LP-MusicCaps-MTT":
        import pandas as pd
        assert flamingo_task in ["AudioCaptioning"]
        assert split in ["train", "test", "val"]

        map_split = lambda split: '../MagnaTagATune/16khz'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = None

        parquet_files = [f for f in os.listdir(dataset_path) if f.endswith('.parquet') and f.startswith(split)]
        print('parquet_files', parquet_files)
        metadata_df = pd.concat([pd.read_parquet(os.path.join(dataset_path, f)) for f in parquet_files])

        for index, row in tqdm(metadata_df.iterrows()):
            filename = row['path']
            if filter_file(file_path, file_list, filename):
                continue
            
            text_prompt = 'generate audio caption'
            for caption in [row['caption_writing'], row['caption_summary'], row['caption_paraphrase']]:
                text_output = caption

                if len(text_output) <= 1:
                    continue
                    
                dataset_dic["data"][dataset_dic["total_num"]] = {
                    "name": filename,
                    "prompt": text_prompt,
                    "output": text_output.replace('\n', ' ')
                }
                dataset_dic["total_num"] += 1

    elif dataset_name == "MACS":
        assert flamingo_task in ["AudioCaptioning", "Tagging"]
        assert split == 'train'

        map_split = lambda split: 'TAU_Urban_Acoustic_Scenes_2019/TAU-urban-acoustic-scenes-2019-development/audio'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = list(filter(lambda x: x.endswith('.wav'), os.listdir(file_path)))

        metadata_list = yaml.load(open(os.path.join(dataset_path, 'MACS.yaml')), Loader=yaml.FullLoader)['files']

        for file_metadata in tqdm(metadata_list):
            filename = file_metadata['filename']
            if filter_file(file_path, file_list, filename):
                continue

            for each_annotated in file_metadata['annotations']:
                caption = each_annotated['sentence']
                tags = ', '.join(each_annotated['tags']).replace('_', ' ')

                if flamingo_task == "AudioCaptioning":
                    text_output = caption
                    text_prompt = 'generate audio caption'

                elif flamingo_task == "Tagging":
                    raise NotImplementedError

                if len(text_output) <= 1:
                    continue
                    
                dataset_dic["data"][dataset_dic["total_num"]] = {
                    "name": filename,
                    "prompt": text_prompt,
                    "output": text_output.replace('\n', ' ')
                }
                dataset_dic["total_num"] += 1
    
    elif dataset_name == "Medley-solos-DB":
        import ndjson
        assert flamingo_task in ["InstrClassification"]

        map_split = lambda split: '44khz'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = list(filter(lambda x: x.endswith('.wav'), os.listdir(file_path)))

        with open(os.path.join(dataset_path, 'medleysolosdb_manifest.ndjson')) as ndjsonfile:
            metadata_list = ndjson.load(ndjsonfile)

        for file_metadata in tqdm(metadata_list):
            subset = file_metadata['subset']
            if not subset.startswith(split):
                continue

            filename = file_metadata['filepath']
            if filter_file(file_path, file_list, filename):
                continue
            
            instrument = file_metadata["instrument"]

            text_output = instrument
            text_prompt = 'this music note is produced by'

            if len(text_output) <= 1:
                continue
                
            dataset_dic["data"][dataset_dic["total_num"]] = {
                "name": filename,
                "prompt": text_prompt,
                "output": text_output.replace('\n', ' ')
            }
            dataset_dic["total_num"] += 1

    elif dataset_name == "MELD":
        import numpy as np
        assert flamingo_task in ["EmotionClassification", "SentimentClassification"]

        map_split = lambda split: '44khz'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = list(filter(lambda x: x.endswith('.wav'), os.listdir(file_path)))

        split_file = os.path.join(
            dataset_path, 
            '{}.txt'.format(split if split in ['train', 'test'] else 'dev')
        )
        with open(split_file, 'r') as f:
            data = f.readlines()
        data = [x.replace('\n', '') for x in data]

        emotion_count = {
            'neutral': 4703, 'happy': 1739, 'sad': 683, 'surprised': 1204,
            'disgusted': 271, 'angry': 1108, 'fearful': 268,
        }
        sentiment_count = {
            'neutral': 4703, 'positive': 2330, 'negative': 2943,
        }
        balancing_factor = 1

        for row in tqdm(data):
            if row.count('|') != 4:
                continue
            filename, utterances, speaker, emotion, sentiment = row.split('|')
            if filter_file(file_path, file_list, filename):
                continue

            if flamingo_task == "EmotionClassification":
                text_output = emotion
                text_output = EMOTION_MAP_DICT[text_output]
                text_prompt = 'this emotion is'

                if split == 'train':
                    balancing_factor = float(emotion_count['neutral']) / float(emotion_count[text_output])
            
            elif flamingo_task == "SentimentClassification":
                text_output = sentiment
                text_prompt = 'this sentiment is'

                if split == 'train':
                    balancing_factor = float(sentiment_count['neutral']) / float(sentiment_count[text_output])
            
            if len(text_output) <= 1:
                continue

            for _ in range(int(np.floor(balancing_factor))):
                dataset_dic["data"][dataset_dic["total_num"]] = {
                    "name": filename,
                    "prompt": text_prompt,
                    "output": text_output.replace('\n', ' ')
                }
                dataset_dic["total_num"] += 1
            
            if np.random.rand() < balancing_factor - np.floor(balancing_factor):
                dataset_dic["data"][dataset_dic["total_num"]] = {
                    "name": filename,
                    "prompt": text_prompt,
                    "output": text_output.replace('\n', ' ')
                }
                dataset_dic["total_num"] += 1
    
    elif dataset_name == "MSP-PODCAST-Publish-1.9":
        assert flamingo_task == "EmotionClassification"
        assert split in ["train", "val", "test"]

        map_split = lambda split: 'Audio'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        
        file_list = glob.glob('{}/*/*.wav'.format(file_path))
        file_list = [x[len(file_path)+1:] for x in file_list]

        subfolder_map = {}
        for f in tqdm(file_list):
            subfolder, filename = f.split('/')
            subfolder_map[filename] = subfolder
        file_list = None

        emotion_dic = {
            'A': 'Angry',
            'S': 'Sad',
            'H': 'Happy',
            'U': 'Surprise',
            'F': 'Fear',
            'D': 'Disgust',
            'C': 'Contempt',
            'N': 'Neutral',
            'O': 'Other',
            'X': 'Not clear'
        }

        with open(os.path.join(dataset_path, 'Labels/labels_concensus.json')) as f:
            data = f.read()
        metadata_dic = json.loads(data)

        for filename in tqdm(list(metadata_dic.keys())):
            values = metadata_dic[filename]
            if not values["Split_Set"].lower().startswith(split):
                continue
            if values["EmoClass"] in ["O", "X"] or values["EmoClass"] not in emotion_dic.keys():
                continue

            subfolder = subfolder_map[filename]
            filename = '{}/{}'.format(subfolder, filename)
            if filter_file(file_path, file_list, filename):
                continue
                
            text_output = emotion_dic[values["EmoClass"]].lower()
            text_output = EMOTION_MAP_DICT[text_output]
            text_prompt = 'this emotion is'
            
            if len(text_output) <= 1:
                continue

            dataset_dic["data"][dataset_dic["total_num"]] = {
                "name": filename,
                "prompt": text_prompt,
                "output": text_output.replace('\n', ' ')
            }
            dataset_dic["total_num"] += 1
    
    elif dataset_name == "mtg-jamendo":
        import ndjson
        assert flamingo_task == "MusicTagging"
        assert split in ["train", "val"]

        map_split = lambda split: '44khz'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = None

        with open(os.path.join(dataset_path, 'mtg_jamendo_{}_manifest.ndjson'.format(split))) as ndjsonfile:
            reader = ndjson.load(ndjsonfile)
            for row in tqdm(reader):
                filename = row["audiopath"]
                if filter_file(file_path, file_list, filename):
                    continue
                
                text_output = row["caption"]
                text_prompt = 'generate music tags (genre, instrument, mood/theme)'
                
                if len(text_output) <= 1:
                    continue

                dataset_dic["data"][dataset_dic["total_num"]] = {
                    "name": filename,
                    "prompt": text_prompt,
                    "output": text_output.replace('\n', ' ')
                }
                dataset_dic["total_num"] += 1
    
    elif dataset_name == "MU-LLAMA":
        
        assert flamingo_task in ['AQA']
        assert split in ['train', 'test']

        map_split = lambda split: 'MusicQA/audios'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = list(filter(lambda x: x.endswith('.wav'), os.listdir(file_path)))

        split_file = 'MusicQA/FinetuneMusicQA.json' if split == 'train' else 'MusicQA/EvalMusicQA.json'
        with open(os.path.join(dataset_path, split_file), 'r') as f:
            data = f.read()
        metadata_list = json.loads(data)

        for dic in tqdm(metadata_list):
            filename = dic["audio_name"]
            if filter_file(file_path, file_list, filename):
                continue

            text_prompt = 'Question: ' + dic["conversation"][0]["value"].strip()
            if not (text_prompt.endswith('.') or text_prompt.endswith('?')):
                text_prompt = text_prompt + '.'

            text_output = dic["conversation"][1]["value"].strip()
            if len(text_output) <= 1:
                continue

            dataset_dic["data"][dataset_dic["total_num"]] = {
                "name": filename,
                "prompt": text_prompt,
                "output": text_output.replace('\n', ' ')
            }
            dataset_dic["total_num"] += 1
    
    elif dataset_name == "musdbhq":
        assert flamingo_task in ["InstrClassification"]
        assert split in ["train", "test", "val"]

        map_split = lambda split: './'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = None

        with open(os.path.join(dataset_path, 'file_list_44k_{}.txt'.format(split))) as f:
            data = f.readlines()
        data = [x.replace('\n', '') for x in data]

        for row in tqdm(data):
            if row.count('|') != 1:
                continue 

            filename, duration = row.split('|')
            duration = float(duration)

            if filter_file(file_path, file_list, filename):
                continue
            
            text_output = filename.split('/')[-1].split('.wav')[0]
            if len(text_output) <= 1:
                continue
            text_prompt = 'this music is produced by'

            segment_length = 10
            for audio_start_idx in range(int(duration // segment_length)):
                dataset_dic["data"][dataset_dic["total_num"]] = {
                    "name": filename,
                    "prompt": text_prompt,
                    "output": text_output.replace('\n', ' '),
                    "audio_start": audio_start_idx * segment_length
                }
                dataset_dic["total_num"] += 1
    
    elif dataset_name == "Music-AVQA":
        import ast
        import re 

        assert flamingo_task in [
            "{}_{}".format(q, t) \
            for q in ['AQA', 'AVQA'] \
            for t in ['Comparative', 'Counting', 'Existential', 'Location', 'Temporal', 'All']
        ]

        def replace_bracketed_words(input_string, replacements):
            def replacer(match):
                word = next(replacements)
                return word

            replacements = iter(replacements)
            output_string = re.sub(r'<[^>]*>', replacer, input_string)
            return output_string

        map_split = lambda split: 'audio'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = list(filter(lambda x: x.endswith('.wav'), os.listdir(file_path)))

        with open(os.path.join(dataset_path, 'MUSIC-AVQA/data/json/avqa-{}.json'.format(split)), 'r') as f:
            data = f.read()
        metadata_list = json.loads(data)

        for dic in tqdm(metadata_list):
            filename = dic["video_id"] + '.wav'
            if filter_file(file_path, file_list, filename):
                continue

            types = ast.literal_eval(dic["type"])
            if 'Visual' in types:
                continue 
            
            if flamingo_task.startswith('AQA_') and 'Audio-Visual' in types:
                continue
            
            if flamingo_task.startswith('AVQA_') and 'Audio' in types:
                continue
            
            t = flamingo_task.split('_')[1]
            if (not t == 'All') and (not t in types):
                continue

            text_output = dic["anser"]
            if len(text_output) <= 1:
                continue
            
            question = dic["question_content"].replace("\uff1f", '?')
            templ_values = ast.literal_eval(dic["templ_values"])
            if len(templ_values) > 0:
                question = replace_bracketed_words(question, templ_values)
            text_prompt = "Question: " + question

            dataset_dic["data"][dataset_dic["total_num"]] = {
                "name": filename,
                "prompt": text_prompt,
                "output": text_output.replace('\n', ' ')
            }
            dataset_dic["total_num"] += 1
    
    elif dataset_name == "MusicCaps":
        assert flamingo_task in ["AudioCaptioning", "EventClassification"]
        assert split in ["train", "test"]

        map_split = lambda split: '44khz'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = list(filter(lambda x: x.endswith('.wav'), os.listdir(file_path)))

        with open(os.path.join(dataset_path, 'musiccaps_manifest.json')) as f:
            data = f.read()
        metadata_list = json.loads(data)

        for file_metadata in tqdm(metadata_list):
            filename = file_metadata['filepath']
            if filter_file(file_path, file_list, filename):
                continue
            
            start_s, end_s = file_metadata["start_s"], file_metadata["end_s"]
            caption = file_metadata["caption"]
            audioset_positive_labels = file_metadata["audioset_positive_labels"]  # audioset classes
            aspect_list = file_metadata["aspect_list"]  # annotated classes

            if (split == 'train') == file_metadata["is_audioset_eval"]:
                continue

            if flamingo_task == "AudioCaptioning":
                text_output = caption
                text_prompt = 'generate audio caption'

            elif flamingo_task == "EventClassification":
                raise NotImplementedError

            if len(text_output) <= 1:
                continue
                
            dataset_dic["data"][dataset_dic["total_num"]] = {
                "name": filename,
                "prompt": text_prompt,
                "output": text_output.replace('\n', ' ')
            }
            dataset_dic["total_num"] += 1
    
    elif dataset_name == "NonSpeech7k":
        assert flamingo_task in ["EventClassification"]
        assert split in ["train", "test"]

        map_split = lambda split: split
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = list(filter(lambda x: x.endswith('.wav'), os.listdir(file_path)))

        all_classes = []
        with open(os.path.join(dataset_path, 'metadata of {} set.csv').format(split), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader)
            for row in tqdm(reader):
                filename, _, _, _, classname, _, _, _ = row
                if filter_file(file_path, file_list, filename):
                    continue

                text_output = classname.lower()
                if len(text_output) <= 1:
                    continue
                text_prompt = 'this is a sound of'

                all_classes.append(classname)

                dataset_dic["data"][dataset_dic["total_num"]] = {
                    "name": filename,
                    "prompt": text_prompt,
                    "output": text_output.replace('\n', ' ')
                }
                dataset_dic["total_num"] += 1
        
        print('all classes:', list(set(all_classes)))

    elif dataset_name == "NSynth":
        import ndjson
        assert flamingo_task in [
            "InstrClassification", 
            "PitchClassification", 
            "VelocityClassification", 
            "SourceClassification",
            "QualityClassification",
            "MIR"
        ]
        assert split in ["train", "test", "val"]

        map_split = lambda split: 'nsynth-{}/audio'.format('valid' if split == 'val' else split)
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = list(filter(lambda x: x.endswith('.wav'), os.listdir(file_path)))

        with open(os.path.join(dataset_path, map_split(split), '../examples.json')) as f:
            data = f.read()
        reader = json.loads(data)

        for key in tqdm(reader):
            filename = key + '.wav'
            if filter_file(file_path, file_list, filename):
                continue

            if flamingo_task == "InstrClassification":
                text_output = reader[key]["instrument_family_str"]
                text_prompt = 'this music note is produced by'
                
            elif flamingo_task == "PitchClassification":
                text_output = str(reader[key]["pitch"])
                text_prompt = 'this music note has pitch'

            elif flamingo_task == "VelocityClassification":
                text_output = str(reader[key]["velocity"])
                text_prompt = 'this music note has velocity'

            elif flamingo_task == "SourceClassification":
                text_output = reader[key]["instrument_source_str"]
                text_prompt = 'this music note has sonic source'

            elif flamingo_task == "QualityClassification":
                qualities_str = reader[key]["qualities_str"]
                if len(qualities_str) >= 1:
                    text_output = ', '.join(qualities_str).replace('_', ' ')
                else:
                    text_output = 'none'
                text_prompt = 'this music note has sonic qualities' 
            
            elif flamingo_task == "MIR":
                instrument = reader[key]["instrument_family_str"]
                pitch = str(reader[key]["pitch"])
                velocity = str(reader[key]["velocity"])
                source = reader[key]["instrument_source_str"]
                qualities_str = ', '.join(reader[key]["qualities_str"]).replace('_', ' ')

                assert len(instrument) > 0
                text_output = 'produced by {}'.format(instrument)
                if len(pitch) > 0:
                    text_output = text_output + ', pitch {}'.format(pitch)
                if len(velocity) > 0:
                    text_output = text_output + ', velocity {}'.format(velocity)
                if len(source) > 0:
                    text_output = text_output + ', source {}'.format(source)
                if len(qualities_str) > 0:
                    text_output = text_output + ', and having qualities like {}'.format(qualities_str)

                text_prompt = 'this music note is' 

            if len(text_output) <= 1:
                continue 

            dataset_dic["data"][dataset_dic["total_num"]] = {
                "name": filename,
                "prompt": text_prompt,
                "output": text_output.replace('\n', ' ')
            }
            dataset_dic["total_num"] += 1
    
    elif dataset_name == "OMGEmotion":
        import numpy as np
        import webrtcvad
        import wave
        from pydub import AudioSegment

        assert flamingo_task == "EmotionClassification"
        assert split in ["train", "val"]

        def convert_to_wav(file_path):
            audio = AudioSegment.from_file(file_path).set_frame_rate(16000).set_channels(1)
            wav_path = file_path.rsplit('.', 1)[0] + "_converted.wav"
            audio.export(wav_path, format="wav")
            return wav_path

        def contains_speech(file_path, aggressiveness=0):
            # aggressiveness between 0 and 3, 0 for very clean speech, and 3 for noisy speech
            wav_path = convert_to_wav(file_path)
            vad = webrtcvad.Vad(aggressiveness)

            with wave.open(wav_path, 'rb') as audio:
                assert audio.getsampwidth() == 2, "Audio must be 16-bit"
                assert audio.getnchannels() == 1, "Audio must be mono"
                assert audio.getframerate() == 16000, "Audio must be sampled at 16kHz"

                frame_duration = 10  # ms
                frame_size = int(audio.getframerate() * frame_duration / 1000)
                num_frames = int(audio.getnframes() / frame_size)

                for _ in range(num_frames):
                    frame = audio.readframes(frame_size)
                    if vad.is_speech(frame, audio.getframerate()):
                        return True

            return False

        map_split = lambda split: 'processed-{}_utterance_data'.format(split)
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = None

        dic_code2emotion = {
            "0": "anger",
            "1": "disgust",
            "2": "fear",
            "3": "happy",
            "4": "neutral",
            "5": "sad",
            "6": "surprise",
        }

        all_emotions = []
        meta_file = os.path.join(
            dataset_path,
            'OMGEmotionChallenge',
            'omg_{}Videos.csv'.format('Train' if split == 'train' else 'Validation')
        )

        with open(meta_file, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader)
            for row in tqdm(reader):
                link, start, end, video, utterance, _, _, EmotionMaxVote = row
                emotion = dic_code2emotion[str(EmotionMaxVote)]

                filename = os.path.join(video, utterance.replace('.mp4', '.mp3'))
                if filter_file(file_path, file_list, filename):
                    continue 
                
                if not contains_speech(os.path.join(file_path, filename)):
                    print('{} does not contain speech'.format(filename))
                    continue

                text_prompt = 'this emotion is'
                text_output = emotion
                if len(text_output) <= 1:
                    continue

                all_emotions.append(EMOTION_MAP_DICT[emotion])

                dataset_dic["data"][dataset_dic["total_num"]] = {
                    "name": filename,
                    "prompt": text_prompt,
                    "output": text_output.replace('\n', ' ')
                }
                dataset_dic["total_num"] += 1
        
        print('all emotions:', list(set(all_emotions)))
    
    elif dataset_name == "OpenAQA":

        assert flamingo_task == 'AQA'
        assert split == 'train'

        map_split = lambda split: './'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = None

        no_word_list = [
            'cannot determine', 'not provided', 'cannot be determined', 'sorry', 'i cannot',
            'without more information', 'enough information',
            'not possible', 'more context', 'enough', 'impossible', 'cannot be determined',
            'without additional information',
            'unclear', 'cannot', 'not clear', 'do not provide sufficient', 'does not provide',
            'difficult to determine', 'no information provided',
            "can't infer", "difficult to infer", "not specified", "no specific", "no information",
            "without additional", 'it is difficult to', "no indication"
        ]

        print('computing dic_audiosetfull_parts')
        audiosetfull_root = '/mnt/fsx-main/rafaelvalle/datasets/audioset/unbalanced_train_segments/22khz/'
        part_strings = [('0'*(2-len(str(p))) + str(p)) for p in range(41)]
        dic_audiosetfull_parts = {
            part: set(os.listdir(os.path.join(audiosetfull_root, 'unbalanced_train_segments_part{}'.format(part)))) \
                for part in part_strings
        }

        audioset20k_filelist = set(os.listdir(os.path.join(file_path, '../AudioSet/train_wav')))

        print('computing dic_clotho_filename')
        clotho_files = os.listdir(os.path.join(dataset_path, '../Clotho-AQA/audio_files'))
        dic_clotho_filename = {
            '_'.join([s for s in f.split(' ') if len(s) > 0]): f \
                for f in clotho_files
        }

        print('reading open_ended/all_open_qa.json')
        with open(os.path.join(dataset_path, 'openaqa/data/open_ended/all_open_qa.json'), 'r') as f:
            data = f.read()
        metadata_list = json.loads(data)

        for dic in tqdm(metadata_list):
            #keys: instruction, input, dataset, audio_id, output, task

            text_output = dic["output"]
            if len(text_output) <= 1:
                continue
            if any(word in text_output.lower() for word in no_word_list):
                continue
            
            question = dic["instruction"]
            text_prompt = question 

            audio_id = dic["audio_id"]
            subset = dic["dataset"]
            if subset == 'clotho_development':
                filename = audio_id.split('/')[-1]
                processed_filename = '_'.join([s for s in filename.split('_') if len(s) > 0])
                if processed_filename in dic_clotho_filename:
                    filename = os.path.join(
                        '../Clotho-AQA/audio_files',
                        dic_clotho_filename[processed_filename]
                    )
                else:
                    continue

            elif subset in ['audiocaps_train', 'as_20k', 'as_strong_train']:
                found = False

                filename = audio_id.split('/')[-1].split('.flac')[0] + '.wav'
                if filename in audioset20k_filelist:
                    filename = os.path.join('../AudioSet/train_wav', filename)
                    found = True
                else:
                    filename = 'Y' + filename
                    for part in part_strings:
                        if filename in dic_audiosetfull_parts[part]:
                            filename = os.path.join(
                                audiosetfull_root, 
                                'unbalanced_train_segments_part{}'.format(part),
                                filename
                            )
                            found = True
                            break 
                    
                if not found:
                    print(filename, 'not found')
                    continue

            elif subset == 'freesound_10s':
                filename = os.path.join(
                    '../CLAP_freesound/freesound_no_overlap/split/train', 
                    audio_id.split('/')[-1]
                )

            elif subset == 'vggsound_train':
                continue
            
            if filter_file(file_path, file_list, filename):
                continue

            dataset_dic["data"][dataset_dic["total_num"]] = {
                "name": filename,
                "prompt": text_prompt,
                "output": text_output.replace('\n', ' ')
            }
            dataset_dic["total_num"] += 1
    
    elif dataset_name == "ravdess":
        assert flamingo_task == "EmotionClassification"
        assert split in ["train", "val"]

        map_split = lambda split: '44khz'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = None

        split_file = os.path.join(
            dataset_path, 
            'ravdess_audiopath_text_sid_emotion_duration_{}_filelist.txt'.format(split)
        )
        with open(split_file, 'r') as f:
            data = f.readlines()
        data = [x.replace('\n', '') for x in data]

        for row in tqdm(data):
            if row.count('|') != 4:
                continue
            filename, utterances, speaker, emotion, duration = row.split('|')
            if filter_file(file_path, file_list, filename):
                continue
                
            text_output = emotion
            text_prompt = 'this emotion is'
            
            if len(text_output) <= 1:
                continue

            dataset_dic["data"][dataset_dic["total_num"]] = {
                "name": filename,
                "prompt": text_prompt,
                "output": text_output.replace('\n', ' ')
            }
            dataset_dic["total_num"] += 1
    
    elif dataset_name == "SongDescriber":
        assert flamingo_task in ["AudioCaptioning"]
        assert split in ["train"]

        map_split = lambda split: './audio/audio'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = None

        with open(os.path.join(dataset_path, 'song_describer.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader)

            for row in tqdm(reader):
                caption_id,track_id,caption,is_valid_subset,familiarity,artist_id,album_id,path,duration = row 
                filename = '{}/{}.2min.mp3'.format(track_id[-2:], track_id)
                duration = float(duration)

                if filter_file(file_path, file_list, filename):
                    continue
                
                text_output = caption
                if len(text_output) <= 1:
                    continue
                text_prompt = 'generate audio caption'

                segment_length = 30
                for audio_start_idx in range(int(duration // segment_length)):
                    dataset_dic["data"][dataset_dic["total_num"]] = {
                        "name": filename,
                        "prompt": text_prompt,
                        "output": text_output.replace('\n', ' '),
                        "audio_start": audio_start_idx * segment_length
                    }
                    dataset_dic["total_num"] += 1
    
    elif dataset_name == "SONYC-UST":
        import numpy as np 

        assert flamingo_task == "EventClassification"
        assert split in ["train", "test", "val"]

        map_split = lambda split: 'audio'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = list(filter(lambda x: x.endswith('.wav'), os.listdir(file_path)))

        all_labels = []
        with open(os.path.join(dataset_path, 'annotations.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for idx, row in tqdm(enumerate(reader)):
                if idx == 0:
                    header = np.array(row)
                    continue 
                    
                if not row[0].startswith(split):
                    continue 
                
                filename = row[2]
                if filter_file(file_path, file_list, filename):
                    continue

                labels = [header[i] for i in range(12, len(header)-8) if str(row[i]) == "1"]
                labels = [x.split("_")[1].replace('-', ' ').lower() for x in labels if 'X_' not in x]
                all_labels += labels 

                text_output = ", ".join(labels)
                if len(text_output) <= 1:
                    continue
                text_prompt = 'this is a sound of'

                dataset_dic["data"][dataset_dic["total_num"]] = {
                    "name": filename,
                    "prompt": text_prompt,
                    "output": text_output.replace('\n', ' ')
                }
                dataset_dic["total_num"] += 1
        
        print('all labels:', list(set(all_labels)))

    elif dataset_name == "SoundDescs":
        import torch
        assert flamingo_task in ["AudioDescription"]
        assert split in ["train"]

        map_split = lambda split: 'raw/audios'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = list(filter(lambda x: x.endswith('.wav'), os.listdir(file_path)))

        split_file = os.path.join(dataset_path, 'audio-retrieval-benchmark/data/SoundDescs/{}_list.txt'.format(split))
        with open(split_file, 'r') as f:
            data = f.readlines()
        names = set([x.replace('\n', '') for x in data])

        with open(os.path.join(dataset_path, 'audio-retrieval-benchmark/sounddescs_data/descriptions.pkl'), 'rb') as f:
            obj = f.read()
            metadata_dic = pickle.loads(obj, encoding='latin1')

        for name in tqdm(names):
            if name not in metadata_dic.keys():
                continue 

            filename = '{}.wav'.format(name)
            if filter_file(file_path, file_list, filename):
                continue
            
            description = metadata_dic[name]
            text_output = description
            text_prompt = 'generate audio description'

            if len(text_output) <= 1:
                continue
                
            dataset_dic["data"][dataset_dic["total_num"]] = {
                "name": filename,
                "prompt": text_prompt,
                "output": text_output.replace('\n', ' ')
            }
            dataset_dic["total_num"] += 1  

    elif dataset_name == "tess":
        assert flamingo_task == "EmotionClassification"
        assert split in ["train", "val"]

        map_split = lambda split: '24414hz'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = None

        split_file = os.path.join(
            dataset_path, 
            'tess_audiopath_text_sid_emotion_duration_{}_filelist.txt'.format(split)
        )
        with open(split_file, 'r') as f:
            data = f.readlines()
        data = [x.replace('\n', '') for x in data]

        for row in tqdm(data):
            if row.count('|') != 4:
                continue
            filename, utterances, speaker, emotion, duration = row.split('|')
            if filter_file(file_path, file_list, filename):
                continue
                
            text_output = emotion.replace('_', ' ')
            text_output = EMOTION_MAP_DICT[text_output]
            text_prompt = 'this emotion is'
            
            if len(text_output) <= 1:
                continue

            dataset_dic["data"][dataset_dic["total_num"]] = {
                "name": filename,
                "prompt": text_prompt,
                "output": text_output.replace('\n', ' ')
            }
            dataset_dic["total_num"] += 1
    
    elif dataset_name == "UrbanSound8K":
        assert flamingo_task in ["EventClassification"]
        assert split in ["train"]

        map_split = lambda split: 'audio'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = None

        with open(os.path.join(dataset_path, 'metadata/UrbanSound8K.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader)
            for row in tqdm(reader):
                filename, fsID, start, end, salience, fold, classID, class_name = row
                filename = 'fold{}/{}'.format(fold, filename)
                if filter_file(file_path, file_list, filename):
                    continue

                text_output = class_name.replace("_", " ").lower()
                if len(text_output) <= 1:
                    continue
                text_prompt = 'this is a sound of'

                dataset_dic["data"][dataset_dic["total_num"]] = {
                    "name": filename,
                    "prompt": text_prompt,
                    "output": text_output.replace('\n', ' ')
                }
                dataset_dic["total_num"] += 1

    elif dataset_name == "VocalSound":
        assert flamingo_task == "VocalClassification"

        map_split = lambda split: 'data_44k'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = list(filter(lambda x: x.endswith('.wav'), os.listdir(file_path)))

        split_file = os.path.join(
            dataset_path, 
            'meta/{}_meta.csv'.format(split[:2] if split in ['train', 'test'] else split[:3])
        )

        prefix = set([])
        with open(split_file, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in reader:
                prefix.add(row[0])
        
        all_labels = set([])
        for filename in tqdm(file_list):
            if not filename.split('_')[0] in prefix:
                continue 
            
            if filter_file(file_path, file_list, filename):
                    continue
            
            label = filename.split('_')[2].split('.wav')[0]
            if label == 'throatclearing':
                label = 'throat clearing'
            
            text_output = label
            text_prompt = 'this vocal sound is'
            all_labels.add(label)
        
            if len(text_output) <= 1:
                continue

            dataset_dic["data"][dataset_dic["total_num"]] = {
                "name": filename,
                "prompt": text_prompt,
                "output": text_output.replace('\n', ' ')
            }
            dataset_dic["total_num"] += 1

        print('all labels:\n', "\'" + "\', \'".join(list(all_labels)) + "\'")

    elif dataset_name.startswith("WavCaps"):
        assert split in ["train"]

        dataset_name, subset_name = dataset_name.split('-')
        dataset_path = os.path.join(
            '/'.join(dataset_path.split('/')[:-1]),
            dataset_name
        )
        dataset_dic['dataset_path'] = dataset_path

        map_split = lambda split: subset_name + '_flac'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = list(filter(lambda x: x.endswith('.flac'), os.listdir(file_path)))

        metadata_file = os.listdir(os.path.join(dataset_path, "json_files", subset_name))
        metadata_file = [x for x in metadata_file if x.endswith('json')][0]
        with open(os.path.join(dataset_path, "json_files", subset_name, metadata_file)) as f:
            data = f.read()
        reader = json.loads(data)

        if subset_name == "AudioSet_SL":
            assert flamingo_task == 'AudioCaptioning'

            for sample in tqdm(reader['data']):
                filename = sample["id"].replace('.wav', '.flac')
                if filter_file(file_path, file_list, filename):
                    continue

                text_output = sample['caption']
                if len(text_output) <= 1:
                    continue
                text_prompt = 'generate audio caption'

                dataset_dic["data"][dataset_dic["total_num"]] = {
                    "name": filename,
                    "prompt": text_prompt,
                    "output": text_output.replace('\n', ' ')
                }
                dataset_dic["total_num"] += 1
        
        else:
            assert flamingo_task in ['AudioCaptioning', 'AudioDescription']

            for sample in tqdm(reader['data']):
                filename = sample["id"] + '.flac'
                if filter_file(file_path, file_list, filename):
                    continue

                if flamingo_task == 'AudioCaptioning':
                    text_output = sample['caption']
                    text_prompt = 'generate audio caption'
                
                elif flamingo_task == 'AudioDescription':
                    text_output = sample['description']
                    text_prompt = 'generate audio description'
                
                if len(text_output) <= 1:
                        continue

                dataset_dic["data"][dataset_dic["total_num"]] = {
                    "name": filename,
                    "prompt": text_prompt,
                    "output": text_output.replace('\n', ' ')
                }
                dataset_dic["total_num"] += 1

    elif dataset_name == "WavText5K":
        assert split == 'train'

        map_split = lambda split: 'Webcrawl/44100/audios'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = list(filter(lambda x: x.endswith('.wav'), os.listdir(file_path)))

        dic = defaultdict(str)
        with open(os.path.join(dataset_path, 'WavText5K.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader)
            for row in tqdm(reader):
                _, _, title, description, filename, tags = row
                dic[filename] = (title, description, tags)

        if flamingo_task == "AudioCaptioning":
            for filename in tqdm(dic.keys()):
                if filter_file(file_path, file_list, filename):
                    continue

                title, description, tags = dic[filename]
                text_output = description
                if len(text_output) <= 1:
                    continue
                text_prompt = 'generate audio caption'

                dataset_dic["data"][dataset_dic["total_num"]] = {
                    "name": filename,
                    "prompt": text_prompt,
                    "output": text_output.replace('\n', ' ')
                }
                dataset_dic["total_num"] += 1
            
        elif flamingo_task == "Tagging":
            for filename in tqdm(dic.keys()):
                if filter_file(file_path, file_list, filename):
                    continue

                title, description, tags = dic[filename]
                if len(tags) < 2 or not tags.startswith('[') or not tags.endswith(']'):
                    continue

                tags = tags[1:-1].split(', ')
                tags = list(map(lambda x: x.replace("'", ""), tags))
                text_output = '{}'.format(', '.join(tags))
                if len(text_output) <= 1:
                    continue
                text_prompt = 'generate tags'

                dataset_dic["data"][dataset_dic["total_num"]] = {
                    "name": filename,
                    "prompt": text_prompt,
                    "output": text_output.replace('\n', ' ')
                }
                dataset_dic["total_num"] += 1
    

    with open(output_file, 'w') as json_file:
        json.dump(dataset_dic, json_file)


# ==================== Precompute CLAP and build Hashing ====================

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)


def update_progress_bar(arg):
    pbar.update()


@suppress_all_output
def load_clap_model(checkpoint):
    if checkpoint in ['630k-audioset-best.pt', '630k-best.pt', '630k-audioset-fusion-best.pt', '630k-fusion-best.pt']:
        amodel = 'HTSAT-tiny'
    elif checkpoint in ['music_speech_audioset_epoch_15_esc_89.98.pt']:
        amodel = 'HTSAT-base'
    else:
        raise NotImplementedError
    
    model = laion_clap.CLAP_Module(
        enable_fusion=('fusion' in checkpoint.lower()), 
        amodel=amodel
    ).cuda()
    model.load_ckpt(ckpt=os.path.join(
        '/lustre/fsw/portfolios/adlr/users/zkong/audio-flamingo-data/laion-clap-pretrained/laion_clap',
        checkpoint
    ))
    return model


def load_audio(file_path, target_sr=44100, duration=30.0, start=0.0):
    if file_path.endswith('.mp3'):
        audio = AudioSegment.from_file(file_path)
        if len(audio) > (start + duration) * 1000:
            audio = audio[start * 1000:(start + duration) * 1000]

        if audio.frame_rate != target_sr:
            audio = audio.set_frame_rate(target_sr)

        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        data = np.array(audio.get_array_of_samples())
        if audio.sample_width == 2:
            data = data.astype(np.float32) / np.iinfo(np.int16).max
        elif audio.sample_width == 4:
            data = data.astype(np.float32) / np.iinfo(np.int32).max
        else:
            raise ValueError("Unsupported bit depth: {}".format(audio.sample_width))

    else:
        with sf.SoundFile(file_path) as audio:
            original_sr = audio.samplerate
            channels = audio.channels

            max_frames = int((start + duration) * original_sr)

            audio.seek(int(start * original_sr))
            frames_to_read = min(max_frames, len(audio))
            data = audio.read(frames_to_read)

            if data.max() > 1 or data.min() < -1:
                data = data / max(abs(data.max()), abs(data.min()))
        
        if original_sr != target_sr:
            if channels == 1:
                data = librosa.resample(data.flatten(), orig_sr=original_sr, target_sr=target_sr)
            else:
                data = librosa.resample(data.T, orig_sr=original_sr, target_sr=target_sr)[0]
        else:
            if channels != 1:
                data = data.T[0]
    
    if data.min() >= 0:
        data = 2 * data / abs(data.max()) - 1.0
    else:
        data = data / max(abs(data.max()), abs(data.min()))
    return data


@torch.no_grad()
def compute_clap_each(audio_file, model):
    try:
        data = load_audio(audio_file, target_sr=48000, duration=10)
        print(audio_file, 'loaded')
    
    except Exception as e:
        print(audio_file, 'unsuccessful due to', e)
        return None
    
    audio_data = data.reshape(1, -1)

    audio_data_tensor = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float().cuda()
    audio_embed = model.get_audio_embedding_from_data(x=audio_data_tensor, use_tensor=True)
    audio_embed = audio_embed.squeeze(0).cpu()
    return audio_embed


@torch.no_grad()
def compute_embeddings_batch(batch, audio_files, model):
    batch_results = []
    for i in batch:
        if i >= len(audio_files):
            break
        audio_file = audio_files[i]
        audio_embed = compute_clap_each(audio_file, model)
        batch_results.append((i, audio_file, audio_embed))
    return batch_results


@torch.no_grad()
def precompute_clap_for_dataset(
    dataset_file, 
    embedding_output_file, 
    checkpoint='630k-audioset-fusion-best.pt'
):
    contents, audio_files = load_dataset_file(dataset_file)

    model = load_clap_model(checkpoint)

    if os.path.exists(embedding_output_file):
        print('loading already computed embedding file from', embedding_output_file)
        with open(embedding_output_file, 'rb') as f:
            saved_data = pickle.load(f)
            curr_audio_indices = saved_data['audio_indices']
            curr_audio_files = saved_data['audio_files']
            curr_audio_embeds = saved_data['audio_embeds']

    else:
        curr_audio_indices = []
        curr_audio_files = []
        curr_audio_embeds = []

    print('computing embeddings for {}'.format(dataset_file))
    start_index = len(curr_audio_files)
    remaining_indices = list(range(start_index, len(audio_files)))

    batch_size = 128
    batches = [
        list(range(i, min(i + batch_size, len(audio_files)))) \
            for i in range(start_index, len(audio_files), batch_size)
    ]

    with multiprocessing.Pool(processes=4) as pool:
        for i, batch in enumerate(batches):
            batch_results = pool.map(
                partial(compute_embeddings_batch, model=model, audio_files=audio_files), 
                [batch]
            )

            for result in batch_results[0]:
                curr_audio_indices.append(result[0])
                curr_audio_files.append(result[1])
                curr_audio_embeds.append(result[2])

            with open(embedding_output_file, 'wb') as f:
                pickle.dump({
                    'audio_indices': curr_audio_indices,
                    'audio_files': curr_audio_files, 
                    'audio_embeds': curr_audio_embeds
                }, f)

            print(f"Saved progress for batch {i+1}/{len(batches)}: \
                audio_indices {len(curr_audio_indices)}, \
                audio_files {len(curr_audio_files)}, \
                audio_embeds {len(curr_audio_embeds)}*{curr_audio_embeds[0].shape}")
    
    return curr_audio_indices, curr_audio_files, curr_audio_embeds


def build_faiss_index(embeddings):
    d = embeddings[0].size(0)
    index = faiss.IndexFlatL2(d)
    np_embeddings = np.vstack([emb.numpy() for emb in embeddings])
    index.add(np_embeddings)
    return index


def build_faiss_index_dataset(
    dataset_file, 
    embedding_output_file, 
    faiss_output_file, 
    checkpoint='630k-audioset-fusion-best.pt',
    only_precompute_clap=False
):
    audio_indices, audio_files, audio_embeds = precompute_clap_for_dataset(dataset_file, embedding_output_file, checkpoint)
    
    if only_precompute_clap:
        return 

    valid_indices, valid_files, valid_embeds = [], [], []
    for audio_index, audio_file, audio_embed in zip(audio_indices, audio_files, audio_embeds):
        if audio_embed is not None:
            valid_indices.append(audio_index)
            valid_files.append(audio_file)
            valid_embeds.append(audio_embed)

    print('building faiss index')
    faiss_index = build_faiss_index(valid_embeds)

    print('saving faiss index')
    faiss.write_index(faiss_index, faiss_output_file)
    with open(faiss_output_file + '.filenames', 'wb') as f:
        pickle.dump({'audio_indices': valid_indices, 'audio_files': valid_files}, f)


# ==================== Generate interleaved dataset files ====================
# only save index so that one can recover

def build_interleaved_dataset(dataset_file, interleaved_output_file, embedding_output_file, faiss_output_file, mode='random', n_samples=3):
    contents, audio_files = load_dataset_file(dataset_file)

    dataset_dic = {
        "dataset_path": contents["dataset_path"],
        "split": contents["split"],
        "split_path": contents["split_path"],
        "flamingo_task": contents["flamingo_task"],
        "total_num": 0,
        "interleaved_data": {},   
    }

    # interleaved_data is 
    # {
    #     id: {
    #         "generation_index_in_split": index of sample in the train or val or test.json,
    #         "fewshot_indices_in_train": list(indices) of few shot samples in train.json
    #     }
    # }

    if mode == 'knn':
        model = load_clap_model(checkpoint='630k-audioset-fusion-best.pt')

        print('loading already computed embedding file from', embedding_output_file)
        with open(embedding_output_file, 'rb') as f:
            precomputed_data = pickle.load(f)
            precomputed_audio_indices = precomputed_data['audio_indices']
            precomputed_audio_files = precomputed_data['audio_files']
            precomputed_audio_embeds = precomputed_data['audio_embeds']

        faiss_index = faiss.read_index(faiss_output_file)
        with open(faiss_output_file+'.filenames', 'rb') as f:
            _data = pickle.load(f)
        faiss_index_audio_indices = _data['audio_indices']
        faiss_index_audio_files = _data['audio_files']

    print('looking for few shot samples and building interleaved_{} data'.format(mode))
    for i in tqdm(range(contents["total_num"])):
        if mode == 'random':
            few_shot_indices = list(np.random.choice(
                list(set(list(range(contents["total_num"]))) - set([i])),
                size=n_samples-1,
                replace=False
            ))
            few_shot_indices = list(map(int, few_shot_indices))

        elif mode == 'knn':
            if audio_files[i] in precomputed_audio_files:
                idx = precomputed_audio_files.index(audio_files[i])
                query_embedding_np = precomputed_audio_embeds[idx]
                if query_embedding_np is not None:
                    query_embedding_np = query_embedding_np.numpy().reshape(1, -1)
                else:
                    continue

            else:
                query_embedding_np = compute_clap_each(audio_files[i], model)
                if query_embedding_np is not None:
                    query_embedding_np = query_embedding_np.numpy().reshape(1, -1)      
                else:
                    continue       

            distances, knn_indices = faiss_index.search(query_embedding_np, n_samples+50)
            distances = distances[0]
            knn_indices = knn_indices[0]

            knn_filenames = [faiss_index_audio_files[idx] for idx in knn_indices]
            combined = list(zip(knn_indices, knn_filenames))
            unique_indices = defaultdict(list)
            for idx, filename in combined:
                unique_indices[filename].append(idx)

            cleared_knn_indices = [random.choice(unique_indices[filename]) for filename in unique_indices if filename != audio_files[i]]

            if dataset_file.endswith('train.json'):
                cleared_knn_indices = [knn_i for knn_i in cleared_knn_indices if faiss_index_audio_indices[knn_i] != i]
            cleared_knn_indices = cleared_knn_indices[:n_samples-1]
            np.random.shuffle(cleared_knn_indices)
            
            few_shot_indices = [faiss_index_audio_indices[knn_i] for knn_i in cleared_knn_indices]

        dataset_dic["interleaved_data"][dataset_dic["total_num"]] = {
            "generation_index_in_split": i,
            "fewshot_indices_in_train": few_shot_indices
        }
        dataset_dic["total_num"] += 1
    
    with open(interleaved_output_file, 'w') as json_file:
        json.dump(dataset_dic, json_file)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_name', type=str, help='dataset name')
    parser.add_argument('-f', '--flamingo_task', type=str, help='flamingo task')
    parser.add_argument('--interleave', action="store_true", help='prepare the interleave dataset')
    args = parser.parse_args()

    ROOT = "/lustre/fsw/portfolios/adlr/users/zkong"
    dataset_root = os.path.join(ROOT, "datasets")
    output_root = os.path.join(ROOT, "audio-flamingo-data/dataset_files")
    os.makedirs(output_root, exist_ok=True)

    dataset_name = args.dataset_name  # "Clotho-v2", "AudioSet", "Clotho-AQA", "WavText5K", "FSD50k", ...
    flamingo_task = args.flamingo_task  # AQA, AudioCaptioning, EventClassification, SceneClassification, Tagging, ...

    # must be train first otherwise there's no train.embedding for query
    for split in ["train", "val", "test"]:
        dataset_path = os.path.join(dataset_root, dataset_name)

        output_folder = '{}-{}'.format(dataset_name, flamingo_task)
        os.makedirs(os.path.join(output_root, output_folder), exist_ok=True)

        dataset_file = os.path.join(output_root, output_folder, '{}.json'.format(split))
        if not os.path.exists(dataset_file):
            try:
                prepare_files(dataset_name, dataset_path, split, flamingo_task, dataset_file)
            except AssertionError as e:
                print('split {} not exist for {}: {}'.format(split, dataset_name, e))
                continue
        else:
            print('{} exists; exiting'.format(dataset_file))
        
        if args.interleave:
            faiss_output_file = dataset_file.replace('{}.json'.format(split), "train_faiss_index.index")
            embedding_output_file = dataset_file.replace('.json', ".embedding")

            if split == 'train':
                if (not os.path.exists(faiss_output_file)) or (not os.path.exists(faiss_output_file + '.filenames')):
                    build_faiss_index_dataset(
                        dataset_file, embedding_output_file, faiss_output_file, 
                        only_precompute_clap=False
                    )
                else:
                    print('{} exists; exiting'.format(faiss_output_file))
            else:
                build_faiss_index_dataset(
                    dataset_file, embedding_output_file, 
                    faiss_output_file=None, 
                    only_precompute_clap=True
                )
                print('precomputing embedding for {} subset finished'.format(split))

            for mode in ['knn', 'random']:
                interleaved_output_file = '/'.join(
                    dataset_file.split('/')[:-1] + \
                    ['interleaved_{}-'.format(mode) + dataset_file.split('/')[-1]]
                )
                if not os.path.exists(interleaved_output_file):
                    build_interleaved_dataset(
                        dataset_file=dataset_file, 
                        interleaved_output_file=interleaved_output_file, 
                        embedding_output_file=embedding_output_file, 
                        faiss_output_file=faiss_output_file, 
                        mode=mode, 
                        n_samples=4
                    )
                else:
                    print('{} exists; exiting'.format(interleaved_output_file))
            



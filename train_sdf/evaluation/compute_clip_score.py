import os
import numpy as np
import json
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

import torch
from torchmetrics.multimodal.clip_score import CLIPScore

metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")

IMG_DIR = '/cluster/andram/abokhovkin/data/Front3D/visuals/atiss_filtered_chunks' # insert your path to images here
CAP_DIR = '/cluster/balar/abokhovkin/data/Front3D/chunked_data_lowres_inference' # insert your path to captions here

with open('/cluster/balar/abokhovkin/data/Front3D/val_scenes_250_main.json', 'r') as fin:
    val_scenes = json.load(fin)

all_img_files = []
for scene_id in os.listdir(IMG_DIR):
    if not scene_id.endswith('.json'):
        all_img_files += [os.path.join(IMG_DIR, scene_id, x) for x in os.listdir(os.path.join(IMG_DIR, scene_id)) if x.endswith('0000.png')]
    
all_scores = []
for img_file in tqdm(all_img_files):
    scene_id = img_file.split('/')[-2]
    img_filename = img_file.split('/')[-1]

    tokens = img_filename.split('_')
    chunk_id = '_'.join(tokens[1:4]) # ours chunks
    # chunk_id = '_'.join(tokens[1:3] + [tokens[3][0]]) # sdfusion chunks, pvd chunks, text2room chunks
    # chunk_id = '_'.join(tokens[0:2] + [tokens[2][0]]) # nfd chunks
    # chunk_id = f'{tokens[0]}_0_{tokens[1]}' # sdfusion scene chunks, ours scene chunks

    caption_key = val_scenes[scene_id]
    # caption_key = 'full_qwen_caption'

    try:
        with open(os.path.join(CAP_DIR, scene_id, f'{chunk_id}.json'), 'r') as fin:
            meta_data = json.load(fin)
            caption = meta_data[caption_key]

            if isinstance(caption, list):
                caption_list = ''
                sep = ', '
                for item in caption:
                    caption_list = caption_list + item + sep
                caption_list = caption_list[:-2]
                caption = caption_list

            caption_walls = meta_data['caption_layout']
            caption_wall = caption_walls[0]
            if caption_wall != '':
                caption = caption + ', ' + caption_wall
    except:
        continue

    all_view_filenames = []
    for i in range(5):
        all_view_filenames += [img_filename.split('.')[0][:-1] + f'{i}.png']

    try:
        view_scores = []
        for view_filename in all_view_filenames:
            view = imread(os.path.join(IMG_DIR, scene_id, view_filename))
            view = view[:, :, :3]
            view = view[300:1180, 300:1180, :3]
            view = resize(view, (224, 224))
            view = np.transpose(view, (2, 0, 1))
            view = torch.LongTensor((view * 256).astype('int32'))

            score = metric(view, 'a render of a 3D scene with ' + caption)
            score = score.detach()

            view_scores += [float(score)]
        max_score = np.max(view_scores)
        all_scores += [max_score]

        print(view_scores, max_score)
    except:
        continue

mean_score = np.mean(all_scores)

print('Final CLIP score:', mean_score)


with open(os.path.join(IMG_DIR, 'clip_score.json'), 'w') as fout:
    json.dump({'score': mean_score}, fout)
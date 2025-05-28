import os, sys
import json
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

# SCENES_DIR = '/cluster/balar/abokhovkin/data/Front3D/chunked_data_lowres_inference' # inference chunks
SCENES_DIR = '/cluster/andram/abokhovkin/data/Front3D/train_chunks_sem_lowres_2x2_sem_canonic' # training chunks

CHOSEN_SCENES = ''


model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-14B-Chat",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-14B-Chat")

cat_to_name = {'Bed': 'bed',
                'Pier/Stool': 'stool',
                'Cabinet/Shelf/Desk': 'cabinet',
                'Lighting': 'lighting',
                'Sofa': 'sofa',
                'Chair': 'chair',
                'Table': 'table',
                'Others': 'object'}


def compute_caption(num_proc=1, proc=0):

    all_scene_ids = sorted([x for x in os.listdir(SCENES_DIR) if '-' in x])[:]
    all_scene_ids = [x for i, x in enumerate(all_scene_ids) if i % num_proc == proc]

    if CHOSEN_SCENES != '':
        with open(os.path.join(CHOSEN_SCENES), 'r') as fin:
            meta_data = json.load(fin)
            all_scene_ids = sorted([scene_id for scene_id in meta_data])
        all_scene_ids = [x for i, x in enumerate(all_scene_ids) if i % num_proc == proc]


    k = 0


    for scene_id in tqdm(all_scene_ids):
        
        json_files = [x for x in os.listdir(os.path.join(SCENES_DIR, scene_id)) if x.endswith('.json')]

        for json_file in tqdm(json_files):
            tokens = json_file.split('.')[0].split('_')
            samp_id = int(tokens[-1])
            # if samp_id > 63:
            #     continue
            try:
                with open(os.path.join(SCENES_DIR, scene_id, json_file), 'r') as fin:
                    chunk_meta_data = json.load(fin)
            except:
                print(scene_id, json_file)
                assert 0

            for caption_type in ['caption_room', 'caption_room_cats', 'caption_rooms_subcats']:

                output_caption_name = f'{caption_type}_qwen'

                if output_caption_name in chunk_meta_data:
                    continue

                if caption_type not in ['caption_room_cats', 'caption_rooms_subcats']:
                    caption_list = chunk_meta_data[caption_type]
                    if not isinstance(caption_list, list):
                        if caption_type not in ['caption_spatial', 'caption_subcat_spatial']:
                            caption_list = caption_list.split(', ')
                            sep = ', '
                        else:
                            caption_list = caption_list.split('; ')
                            sep = '; '
                    else:
                        sep = ', '
                    caption = ''
                    for item in caption_list:
                        caption = caption + item + sep
                    caption = caption[:-2]

                else:
                    caption_input = chunk_meta_data[caption_type]
                    if len(caption_input) == 0:
                        caption = 'Empty scene'
                    else:
                        caption = 'The scene has '
                    for room_name in caption_input:
                        if len(caption_input[room_name]) == 0:
                            caption += f'{room_name}; '
                        else:
                            caption += f'{room_name} with '
                            for object_name in caption_input[room_name]:
                                if caption_type == 'caption_room_cats':
                                    object_name_uniform = cat_to_name[object_name]
                                else:
                                    object_name_uniform = object_name
                                caption += f'{object_name_uniform}, '
                            caption = caption[:-2]
                            caption += '; '
                    caption = caption[:-2]


                # print(scene_id, json_file)
                # print(caption)
                # print(k)

                try:
                    prompt = f'Reformulate the following synthetic description of a 3D scene into a human-readable but concise, extremely minimalistic and non-list format in only one sentence: "{caption}"'
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    model_inputs = tokenizer([text], return_tensors="pt").to(device)


                    generated_ids = model.generate(
                        model_inputs.input_ids,
                        max_new_tokens=75
                    )
                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                    ]

                    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                except FileNotFoundError:
                    print('!')
                    print()
                    continue

                # print(response)
                # print()

                k += 1

                chunk_meta_data[output_caption_name] = response

            with open(os.path.join(SCENES_DIR, scene_id, json_file), 'w') as fout:
                json.dump(chunk_meta_data, fout)

            # break


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_proc', default=1, type=int)
    parser.add_argument('-p', '--proc', default=0, type=int)
    args = parser.parse_args()

    compute_caption(args.num_proc, args.proc)


        


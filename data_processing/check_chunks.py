import os, sys
import numpy as np


CHUNKS_DIR = '/cluster/balar/abokhovkin/data/Front3D/train_chunks_sem'

def compute_chunks(num_proc=1, proc=0):

    all_obj_ids = sorted(os.listdir(CHUNKS_DIR))[:]
    all_obj_ids = [x for i, x in enumerate(all_obj_ids) if i % num_proc == proc]


    all_bad_chunks = []

    for obj_id in all_obj_ids:
        for filename in os.listdir(os.path.join(CHUNKS_DIR, obj_id)):
            if not filename.endswith('_semantic.npy'):
                try:
                    chunk = np.load(os.path.join(CHUNKS_DIR, obj_id, filename))
                except ValueError:
                    all_bad_chunks += [(obj_id, filename)]
                    print(obj_id, filename)
                    continue


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_proc', default=1, type=int)
    parser.add_argument('-p', '--proc', default=0, type=int)
    args = parser.parse_args()

    compute_chunks(args.num_proc, args.proc)
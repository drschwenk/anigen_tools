import os
import multiprocessing
from functools import partial
from anigen_tools.mturk import unpickle_this
from anigen_tools.interpolation import interpolate_all_video_entites
from anigen_tools.tracking import track_all_video_entites
from anigen_tools.frame_extraction import video_to_npy
from anigen_tools.interpolation import draw_video_interps
from anigen_tools.hybrid_segmentation import segment_video


procs = os.cpu_count()


def multimap(method, iterable, *args):
    multiprocessing.set_start_method('spawn', force=True)
    os.environ['THEANO_FLAGS'] = 'device=cpu'
    pool = multiprocessing.Pool(procs)
    results = pool.map(method, iterable)
    pool.close()
    pool.join()
    return results


def perform_interpolation(videos):
    complete_vids_all = unpickle_this('complete_vids_all.pkl')
    complete_prod_dataset = unpickle_this('complete_prod_dataset.pkl')
    # interp_func = partial(interpolate_all_video_entites, complete_prod_dataset)    
    interp_func = partial(draw_video_interps, complete_prod_dataset)
    _ = multimap(interp_func, videos)


def perform_tracking(videos):
    _ = multimap(track_all_video_entites, videos)


def perform_frame_extraction(videos):
    _ = multimap(video_to_npy, videos)


def perform_segmentation(videos):
    _ = multimap(segment_video, videos)


if __name__ == '__main__':
    complete_vids_all = unpickle_this('sample_ds_100.pkl')
    # perform_frame_extraction(complete_vids_all[:10])
    # perform_interpolation(complete_vids_all[:10])
    # perform_tracking(complete_vids_all[:10])
    perform_segmentation(complete_vids_all[:9])

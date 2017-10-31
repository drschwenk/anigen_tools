import numpy as np
import skvideo.io as skvio
import skimage.transform as sktransform
import multiprocessing
import os
import glob

trajectories_dir = 'trajectories'
tracking_dir = 'tracking'
interp_dir = 'interpolation'
viz_dir = 'viz'
frame_arr_dir = 'frame_arr_data'
video_dir = 'video_data'


def video_to_images(in_video_name):
    videogen = skvio.FFmpegReader(in_video_name).nextFrame()
    images = []
    for img in videogen:
        images.append(img)
    return images


def images_to_npy(images, out_npy_name):
    for j, img in enumerate(images):
        img = sktransform.resize(img, (128, 128), mode='reflect')
        img = 255 * img
        img = img.astype(np.uint8)
        images[j] = img

    images = np.stack(images, 0)
    np.save(out_npy_name, images)


def video_to_npy(in_video_name):
    out_npy_name = in_video_name.replace('mp4', '.npy')
    images = video_to_images(in_video_name=in_video_name)
    images_to_npy(images=images, out_npy_name=out_npy_name)


def video_to_npy_parallel(file_names):
    # file_names: A list of tuples
    # file_names[x] = (in_video_name, out_npy_name)

    procs = os.cpu_count()
    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(procs)
    results = pool.starmap(video_to_npy, file_names)
    pool.close()
    pool.join()


if __name__ == '__main__':
    # Example to convert 1 video
    in_video_name = '/Users/anik/data/animation_gan/scratch/s_01_e_01_shot_000099_000173.mp4'
    out_npy_name = '/Users/anik/data/animation_gan/scratch/s_01_e_01_shot_000099_000173.mp4.npy'

    video_to_npy(in_video_name=in_video_name, out_npy_name=out_npy_name)

    # Example to convert a large set of videos using multiprocessing

    in_dir_name = '/Users/anik/data/animation_gan/samples'
    out_dir_name = '/Users/anik/data/animation_gan/out_npy'

    in_video_names = glob.glob(os.path.join(in_dir_name, '*.mp4'))
    out_npy_names = [os.path.join(out_dir_name, os.path.split(x)[1] + '.npy') for x in in_video_names]
    file_names = [(in_video_names[count], out_npy_names[count]) for count in range(len(in_video_names))]

    video_to_npy_parallel(file_names=file_names)

    # Comparing two npy files
    original_npy_file = '/Users/anik/data/animation_gan/scratch/s_01_e_01_shot_000099_000173.npy'
    compare_npy_files(out_npy_name, original_npy_file)

    # Visualizing two npy files
    viz_two_npy_files(file1=out_npy_name, file2=original_npy_file)

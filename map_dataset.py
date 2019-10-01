from configparser import ConfigParser
import os
import argparse
from random import shuffle, seed
# from data_collection.surfing_data_inspection import parse_file_name
import glob


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--ini_file", action="store", default=r"init.ini",
                    help="ini file name")
args = parser.parse_args()
config = ConfigParser()
config.read(args.ini_file)
seed(a=1337)

imgs_types = ('.png', '.jpg','.npy' )


def get_images_graph(doc_file, src):
    print('creating_graph')
    # for file in glob.iglob(os.path.join(src, '*.png')):

    for dir, subdirs, clips in os.walk(src):
        for clip in clips:
            clip_file = os.path.join(dir, clip)
            if not clip.endswith(imgs_types) or not os.stat(clip_file).st_size:
                continue
            doc_file.write(clip_file + '\n')
            break

    doc_file.close()
    # folder_files_gen = glob.iglob(os.path.join(src, '*.png'))
    # size = 0
    # while not size:
    #     file = next(folder_files_gen)
    #     size = os.stat(file).st_size
    #     # file_name, frame_num, orig_size, xcenter = parse_file_name(file)
    #     try:
    #         # float(xcenter)
    #         doc_file.write(file + '\n')
    #     except ValueError:
    #         continue
    #
    # doc_file.close()


def random_files(file, dst):

    images = file.read().split('\n')
    if images[-1] == '':
        del images[-1]
    shuffle(images)
    for im in images:
        dst.write(im + '\n')

    print('amount of images {}'.format(len(images)))


def main():
    file_loc = config['graph']['txt_file']
    file = open(file_loc, 'w')
    src = config['graph']['folder_to_map']

    get_images_graph(file, src)

    file = open(file_loc, 'r')

    rand_file_loc = config['graph']['txt_file_rand']
    rand_file = open(rand_file_loc, 'w')

    random_files(file, rand_file)

    file.close()
    rand_file.close()

if __name__ == '__main__':
    main()

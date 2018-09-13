import pydicom
# from openpyxl import load_workbook
from collections import Counter
import xlrd
from random import shuffle
import scipy.misc as sp
import matplotlib.pyplot as plt
import os
import logging
import numpy as np


def parse_xsl(path):
    """
    parse_xsl from table to list of dicts
    :param path: str - path of xslx file
    :return samples: list of dicts by rows. Each value is sorted by the column's name as the key
    """
    wb = xlrd.open_workbook(path)
    ws = wb.sheet_by_index(0)
    num_of_rows = ws.nrows
    num_of_cols = ws.ncols

    samples = []
    for i in range(1, num_of_rows):
        sample = dict()
        for j in range(num_of_cols):
            sample[str(ws.cell(0, j).value)] = str(ws.cell(i, j).value)
        samples.append(sample)

    format_accession_field(samples)
    return samples


def format_accession_field(samples):

    for sample in samples:
        try:
            float(sample['Accession'])
            sample['Accession'] = sample['Accession'].split('.')[0]
        except ValueError:
            pass


def samples_to_lables(samples):
    """
    seperate between plural efusion positive and negative and return lists of Accession field
    :param samples: list of samples generated from xl file
    :return plural_efusion_positive: list of positive samples' Accession
    :return all_the_rest: list of negative samples' Accession
    """
    plural_efusion_positive = [sample['Accession'] for sample in samples if (sample['Taflitright'] == '1.0' or
                                                                             sample['Taflitleft'] == '1.0')]
    all_the_rest = [sample['Accession'] for sample in samples if (sample['Taflitright'] == '0.0' and
                                                                  sample['Taflitleft'] == '0.0')]

    return plural_efusion_positive, all_the_rest


def unite_labels(samples_dict):

    samples = []
    for k, val in samples_dict.items():
        [samples.append({'Accession': sample, 'label': int(k)}) for sample in val]
    return samples


def load_images_to_samples(samples):
    pass


def eliminate_multiple_occurrences(dataset_descript):

    access_list = [sample['Accession'] for sample in dataset_descript]

    counter = Counter(access_list)

    multiple_occurence = [key for key, val in counter.items() if val > 1]

    for key in multiple_occurence:
        samples = [{'sample': sample, 'index': i} for i, sample in enumerate(dataset_descript)
                   if sample['Accession'] == key]

        for i, sample in enumerate(samples):
            for comp_sample in samples[i+1:]:
                if sample['sample'] == comp_sample['sample']:
                    del dataset_descript[comp_sample['index']]
                else:
                    del dataset_descript[comp_sample['index']]
                    del dataset_descript[sample['index']]


def unpack_data(src, samples, dst):
    def check_LAT_PA(fold):
        angles = {'PA' : [],
                  'LAT': []}
        if not sample['label']:  ### debug
            print('{}\n{}\n**************'.format(fold, os.listdir(fold)))
        for key in angles.keys():
            for file in os.listdir(fold):
                if file.find(key) != -1:
                    angles[key].append(file)
        return angles

    for sample in samples:
        fold = os.path.join(src, sample['Accession'])
        angle_files = check_LAT_PA(fold)
        if not all(angle_files.values()):
            if not angle_files['PA']:
                logging.warning('{} doesn\'t hold a PA file'.format(fold))
            continue

        matches = []
        matched = []
        for file_pa in angle_files['PA']:
            ending = file_pa.split('PA')[-1]

            match = [file_lat for file_lat in angle_files['LAT'] if file_lat.split('LAT')[-1] == ending]
            if match:
                logging.info('successfully matched {} to {} in {}'.format(file_pa, match[0], fold))
                matches.append({'PA': file_pa, 'LAT': match[0]})
                matched.append(match[0])
            else:
                logging.warning('{} has no match for {}'.format(fold, file_pa))

        [logging.warning('{} has no match for {}'.format(fold, file_lat))
         for file_lat in angle_files['LAT'] if file_lat not in matched]

        for i, match in enumerate(matches):
            for angle, angle_file in match.items():
                # if len(angle_file) > 1:
                #     err = '{} has {} {} files: {}'.format(fold, len(angle_file), angle, angle_file)
                #     # raise ValueError(err)
                #     logging.warning(err)

                dic_path = os.path.join(fold, angle_file)
                ds = pydicom.dcmread(dic_path)

                dst_dir = os.path.join(os.path.join(dst, str(sample['label'])), sample['Accession'] + '_{}'.format(i))
                os.makedirs(dst_dir, exist_ok=True)
                dst_path = os.path.join(dst_dir, sample['Accession'] + '_{}'.format(i) + '_' + angle + '.png')

                logging.info('saving {} to {}'.format(dic_path, dst_path))
                # sp.imsave(dst_path, ds.pixel_array)
                save_xray_scan(dst_path, ds)

def save_xray_scan(dst, ds):

    img = ds.pixel_array.astype(np.float32)
    if ds.PhotometricInterpretation == 'MONOCHROME1':
        maxl = np.amax(img)
        img = (2 ** int(np.ceil(np.log2(maxl - 1)))) - 1 - img
    sp.imsave(dst, img)

def main():

    handlers = [logging.FileHandler('/Users/Ido/Desktop/Project Data/data_log.log'), logging.StreamHandler()]
    formatter = '%(levelname)s - %(message)s'
    logging.basicConfig(handlers=handlers, level=logging.INFO, format=formatter)

    trainset_descript = parse_xsl('/Users/Ido/Desktop/Project Data/Shiba_dataset/db_description_cxr_train.xlsx')
    testset_descript = parse_xsl('/Users/Ido/Desktop/Project Data/Shiba_dataset/db_description_cxr_test.xlsx')

    eliminate_multiple_occurrences(trainset_descript)
    eliminate_multiple_occurrences(testset_descript)

    positives_trainset, negatives_trainset = samples_to_lables(trainset_descript)
    train_data = unite_labels({'0': negatives_trainset, '1': positives_trainset})
    shuffle(train_data)

    positives_testset, negatives_testset = samples_to_lables(testset_descript)
    test_data = unite_labels({'0': negatives_testset, '1': positives_testset})
    shuffle(test_data)
    # unpack_data('/Users/Ido/Desktop/Project Data/Shiba_dataset/train_dcm', train_data, '/Users/Ido/Desktop/Project Data/train_images')
    unpack_data('/Users/Ido/Desktop/Project Data/Shiba_dataset/test_dcm', test_data, '/Users/Ido/Desktop/Project Data/test_images')


if __name__ == '__main__':
    main()
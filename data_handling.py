import pydicom
# from openpyxl import load_workbook
from collections import Counter
import xlrd
from random import shuffle
import scipy.misc as sp
import matplotlib.pyplot as plt
import os


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

    for sample in samples:
        fold = os.path.join(src, sample['Accession'])
        dic_path = os.path.join(fold, sample['Accession'] + '_PA.dcm')
        ds = pydicom.dcmread(dic_path)
        dst_path = os.path.join(os.path.join(dst, str(sample['label'])), sample['Accession'] + '.png')
        print('saving {} to {}'.format(dic_path, dst_path))
        sp.imsave(dst_path, ds.pixel_array)



def main():

    trainset_descript = parse_xsl('/Users/Ido/Desktop/Project Data/db_description_cxr_train.xlsx')
    testset_descript = parse_xsl('/Users/Ido/Desktop/Project Data/db_description_cxr_test.xlsx')

    eliminate_multiple_occurrences(trainset_descript)
    eliminate_multiple_occurrences(testset_descript)

    positives_trainset, negatives_trainset = samples_to_lables(trainset_descript)
    train_data = unite_labels({'0': negatives_trainset, '1': positives_trainset})
    shuffle(train_data)

    positives_testset, negatives_testset = samples_to_lables(testset_descript)
    test_data = unite_labels({'0': negatives_testset, '1': positives_testset})
    shuffle(test_data)
    unpack_data('/Users/Ido/Desktop/Project Data/train_dcm', train_data, '/Users/Ido/Desktop/Project Data/train_images')


if __name__ == '__main__':
    main()
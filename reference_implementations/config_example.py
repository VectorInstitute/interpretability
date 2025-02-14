"""
#TODO: To be deleted or created into .md

Examples of loading data paths
"""

import os
import pprint
from data import Config

config = Config()

print('\nPrint interp_config from yaml')
print(config)

print("\n~~~~~ MANUAL USAGE ~~~~~~")

print("\n 1. Create path for data within repo")

print("\nPath to bootcamp repo:",
      config.root.repo)

print("\nPath to datafile on the cluster:",
      config.data.repo)

print("\nCreate a complete filepath for 1 dataset on the cluster: ", \
      f'{config.root.vcluster}/{config.data.vcluster.credit_card_behaviour}')
print("\nOR\n")
print(os.path.join(config.root.vcluster,\
                   config.data.vcluster.credit_card_behaviour))

print("\n2. Create path for data on cluster ")

print("\Root path to bootcamp folder:",
      config.root.vcluster)

print("\nRelative path to data folder:",
      config.data.vcluster)

print("\nCreate an absolute folder for 1 dataset on the cluster:")
data_folder = f'{config.root.vcluster}/{config.data.vcluster.credit_card_behaviour}'
print(data_folder)

print("\n Use a csv file for loading into dataframe: ")
csv_file = f'{data_folder}/Dev_data_to_be_shared.csv'
print(csv_file)

print("\n ~~~~~~API USAGE~~~~~")

print('\nGet list of all datasets for bootcamp:')
datasets = config.datasets
print(datasets)

print('\nGet absolute path of a data folder:')
data_path = config.get_datapath('bank_marketing')
print(data_path)

print('\n Get absolute file paths of a dataset from repo/cluster:')
data_files = config.get_datafiles('bank_marketing')
pprint.pprint(data_files)

print('\n Use a single file path for dataframe loading: ', data_files['bank.csv'])

print('\n Get large dataset from cluster WITHOUT subdir files/images')
large_data_files = config.get_datafiles('nih')
pprint.pprint(large_data_files)

print('\n Use a folder path for train data loading: ', large_data_files['images_005'])

print('\n Get dataset from cluster WITH subdir files/images. \
      It will give lots of output for a large as it lists all data files/images')
#large_data_files = config.get_datafiles('nih', exclude_subdirs=False)
#pprint.pprint(large_data_files)
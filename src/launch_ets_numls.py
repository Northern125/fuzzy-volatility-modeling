import logging
from numpy import array
from pandas import Timestamp, read_pickle
import json
import os
from pathlib import Path
import pickle

from model import FuzzyVolatilityModel

# from membership_functions.trapezoidal import _calc_trapezoidal_md_scalar

with open('config.json') as file:
    config = json.load(file)

INPUT = Path(config['files_folders']['processed'])
LOGS_DIR = Path(config['files_folders']['logs'])
OUTPUT = Path(config['files_folders']['research_results'])
MD_DIR = Path(config['files_folders']['calculations_metaparams'])

MD_FILE_NAME_CURRENT = 'current_fvm_ets_metadata.pkl'
TESTED_MODEL_FILE_NAME = 'tested_ets'
BIG_FILE_NAME = 'tested_data_ets'

script_name = os.path.basename(__file__).split('.')[0]

logger = logging.getLogger(script_name)

if __name__ == '__main__':
    # input
    script_n = input('Script #: ')
    if script_n == '':
        script_n = 0
        print('OK, using default (#0)')
    else:
        script_n = int(script_n)

    _log_file = LOGS_DIR / f'{logger.name}_{script_n}.log'
    logging.basicConfig(level=logging.INFO,
                        filename=_log_file,
                        filemode='w',
                        format='%(asctime)s.%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    _metadata_file_name = input('metadata file name = (enter to use default) ')
    if _metadata_file_name == '':
        print(f'OK, using file w/ current metadata {MD_FILE_NAME_CURRENT}')
        _metadata_file_name = MD_FILE_NAME_CURRENT

    print('OK, starting the procedure')

    # loading data
    metadata = read_pickle(MD_DIR / _metadata_file_name)
    fvm_ini_params: dict = metadata['fvm_ini_params']
    fvm_test_params: dict = metadata['fvm_test_params']
    _desc: dict = metadata['_desc']

    print('Finished loading data')

    # creating an FVM instance & feeding test data
    fvm = FuzzyVolatilityModel(**fvm_ini_params)

    fvm.n_clusters = fvm_ini_params['clusterization_parameters']['n_clusters']
    fvm._n_clusters_hist.append(fvm.n_clusters)
    fvm.membership_degrees_current = array([1 / fvm.n_clusters for _ in range(1, fvm.n_clusters + 1)])

    fvm.fit()

    fvm.forecast()

    fvm.feed_daily_data(**fvm_test_params)

    # exporting
    _cur_time = str(Timestamp.today()).replace(':', '-').replace(' ', '_')

    _str_desc = str(_desc).replace(':', '=').replace("""'""", '')
    _file_name = f'{TESTED_MODEL_FILE_NAME}_{_str_desc}_{_cur_time}.pkl'.replace(' ', '_').replace(':', '-')
    with open(OUTPUT / _file_name, 'wb') as _file:
        pickle.dump(fvm, _file)

    data = {
        'fvm': fvm,
        '_desc': _desc
    }

    _big_file_name = f'{BIG_FILE_NAME}_{_str_desc}_{_cur_time}.pkl'
    with open(OUTPUT / _big_file_name, 'wb') as _file:
        pickle.dump(data, _file)

    print('Complete')

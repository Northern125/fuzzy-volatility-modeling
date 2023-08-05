from pathlib import Path
import os
import json
import logging
from multiprocessing import Pool
import traceback
import pickle

from numpy import array
from pandas import read_pickle, Timestamp

from model import FuzzyVolatilityModel

with open('config.json') as file:
    config = json.load(file)

INPUT = Path(config['files_folders']['processed'])
LOGS_DIR = Path(config['files_folders']['logs'])
LOGS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = Path(config['files_folders']['research_results'])
OUTPUT.mkdir(parents=True, exist_ok=True)
MD_DIR = Path(config['files_folders']['calculations_metaparams'])

MD_FILE_NAME_CURRENT = 'current_fvm_eTS_metadata_mul.pkl'
TESTED_MODEL_FILE_NAME = 'tested_ets_mul'

script_name = os.path.basename(__file__).split('.')[0]

logger = logging.getLogger(script_name)


def do_single_input(metadata: dict):
    _desc: dict = metadata['_desc']

    try:
        fvm_ini_params: dict = metadata['fvm_ini_params']
        fvm_test_params: dict = metadata['fvm_test_params']

        fvm = FuzzyVolatilityModel(**fvm_ini_params)

        fvm.n_clusters = fvm_ini_params['clusterization_parameters']['n_clusters']
        fvm._n_clusters_hist.append(fvm.n_clusters)
        fvm.membership_degrees_current = array([1 / fvm.n_clusters for _ in range(1, fvm.n_clusters + 1)])

        fvm.fit()

        fvm.forecast()

        fvm.feed_daily_data(**fvm_test_params)

        script_info = {
            'status': 0,
            'exception': None,
            'traceback': None
        }

        print('Single iteration completed')
    except Exception as e:
        logger.exception(f'Single iteration failed: {e}')
        print(f'Single iteration failed: {e}')

        fvm = None
        script_info = {
            'status': -1,
            'exception': e,
            'traceback': traceback.format_exc()
        }
    return fvm, _desc, script_info


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

    _pool_params_file_name = input('pool_params file name = (enter to use empty dict) ')
    if _pool_params_file_name == '':
        print('OK, using empty dict')
        pool_params = {}
    else:
        with open(INPUT / _pool_params_file_name, 'r') as _file:
            pool_params = json.load(_file)

    _starmap_params_file_name = input('starmap_params file name = (enter to use empty dict) ')
    if _starmap_params_file_name == '':
        print('OK, using empty dict')
        starmap_params = {}
    else:
        with open(INPUT / _starmap_params_file_name, 'r') as _file:
            starmap_params = json.load(_file)

    _use_mp = input('use multiprocessing? (y/n) ')
    if _use_mp == 'y':
        use_mp = True
    elif _use_mp == 'n':
        use_mp = False
    else:
        raise ValueError('the answer should be either y or n')

    print('OK, starting the procedure')

    # loading data
    metadata_multiple: list = read_pickle(MD_DIR / _metadata_file_name)
    md_mp = [(_md, ) for _md in metadata_multiple]

    print('Finished loading data')

    # launching algorithm
    if use_mp:
        with Pool(**pool_params) as p:
            algorithm_results = p.starmap(do_single_input, md_mp, **starmap_params)
    else:
        algorithm_results = []
        for metadata in metadata_multiple:
            algorithm_results.append(do_single_input(metadata))

    # exporting
    _cur_time = str(Timestamp.today())
    _file_name = f'{TESTED_MODEL_FILE_NAME}_{_cur_time}.pkl'.replace(' ', '_').replace(':', '-')
    with open(OUTPUT / _file_name, 'wb') as _file:
        pickle.dump(algorithm_results, _file)

import pandas as pd
import json
import logging
import pickle
import os
from pathlib import Path

with open('../config.json') as file:
    config = json.load(file)

RES_DIR = Path(config['files_folders']['research_results'])
LOGS_DIR = Path(config['files_folders']['logs'])

TESTED_MODEL_FILE_NAME = 'tested_model'
BIG_FILE_NAME = 'tested_data'

script_name = os.path.basename(__file__).split('.')[0]

logger = logging.getLogger(script_name)

_log_file = LOGS_DIR / f'{logger.name}.log'
logging.basicConfig(level=logging.INFO,
                    filename=_log_file,
                    filemode='w',
                    format='%(asctime)s.%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


if __name__ == '__main__':
    # input
    _apf_result_file_name = input('APF result file name = ')

    print('OK, starting the procedure')

    # loading data
    _files = [_apf_result_file_name]
    res_big = [pd.read_pickle(RES_DIR / _file) for _file in _files]
    _res = [_elem for _res_piece in res_big for _elem in _res_piece['result']].copy()
    res = pd.DataFrame.from_records(_res).copy()

    if (res['status'] != 0).any():
        raise Exception('Not all models were fitted')

    best_res = res[res['rmse'] == res['rmse'].min()].iloc[0].copy()
    fitted_ap = best_res['antecedent_params'].copy()
    fvm = best_res['fvm']

    test = res_big[0]['test'].copy()
    data_to_cluster_test = res_big[0]['data_to_cluster_test'].copy()

    print('Finished loading data')

    # feeding
    fvm.feed_daily_data(test, data_to_cluster=data_to_cluster_test)

    print('Finished feeding')

    # exporting
    _cur_time = str(pd.Timestamp.today()).replace(':', '-').replace(' ', '_')

    _desc = res_big[0]['_desc']
    _seas_or_wos = 'seas' if _desc['is_seas'] else 'wos' if not _desc['is_seas'] else None
    _sample_desc = f"""{_desc['series_name']}_{_seas_or_wos}_{_desc['train_start']}_{_desc['n_train']}_{_desc['n_test']}_{_desc['n_retrain']}_M={_desc['M']}_at_{_cur_time}.pkl"""
    _file_name = f'{TESTED_MODEL_FILE_NAME}_{_sample_desc}_{_cur_time}.pkl'
    with open(RES_DIR / _file_name, 'wb') as _file:
        pickle.dump(fvm, _file)

    data = {
        'fvm': fvm,
        '_desc': _desc
    }

    _big_file_name = f'{BIG_FILE_NAME}_{_sample_desc}_{_cur_time}.pkl'

    with open(RES_DIR / _big_file_name, 'wb') as _file:
        pickle.dump(data, _file)

    print('Complete')

import pandas as pd
import json
import logging
import pickle
import os
from pathlib import Path

from src.antecedent_fitting import fit_antecedent_params

with open('config.json') as file:
    config = json.load(file)

INPUT = Path(config['files_folders']['processed'])
RES_DIR = Path(config['files_folders']['research_results'])
AP_DIR = Path(config['files_folders']['antecedent_params_sets'])
LOGS_DIR = Path(config['files_folders']['logs'])

script_name = os.path.basename(__file__).split('.')[0]

logger = logging.getLogger(script_name)

_log_file = LOGS_DIR / f'{logger.name}.log'
logging.basicConfig(level=logging.INFO,
                    filename=_log_file,
                    filemode='w',
                    format='%(asctime)s.%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# some settings
_use_mp_default = config['defaults']['apf_settings']['use_multiprocessing']
if _use_mp_default.lower() == 'true':
    USE_MULTIPROCESSING_DEFAULT = True
elif _use_mp_default.lower() == 'false':
    USE_MULTIPROCESSING_DEFAULT = False
else:
    raise ValueError("""Default value for use_mp should be either 'True' or 'False'; """
                     f"""got '{_use_mp_default}'""")

_do_feeding_default = config['defaults']['apf_settings']['do_feeding']
if _do_feeding_default.lower() == 'true':
    DO_FEEDING_DEFAULT = True
elif _do_feeding_default.lower() == 'false':
    DO_FEEDING_DEFAULT = False
else:
    raise ValueError("""Default value for do_feeding should be either 'True' or 'False'; """
                     f"""got '{_do_feeding_default}'""")

if __name__ == '__main__':
    # input
    _use_multiprocessing = input('use_multiprocessing = ')
    if _use_multiprocessing == '':
        print(f'OK, using default value {USE_MULTIPROCESSING_DEFAULT}')
        use_multiprocessing = USE_MULTIPROCESSING_DEFAULT
    elif _use_multiprocessing.lower() == 'false':
        use_multiprocessing = False
    elif _use_multiprocessing.lower() == 'true':
        use_multiprocessing = True
    else:
        raise ValueError('The use_multiprocessing input value should be true or false')

    _do_feeding = input('do_feeding = ')
    if _do_feeding == '':
        print(f'OK, using default value {DO_FEEDING_DEFAULT}')
        do_feeding = DO_FEEDING_DEFAULT
    elif _do_feeding.lower() == 'false':
        do_feeding = False
    elif _do_feeding.lower() == 'true':
        do_feeding = True
    else:
        raise ValueError('The do_feeding input value should be true or false')

    seas_or_wos = input('seas / wos = ')
    if seas_or_wos == 'seas':
        is_seas = True
    elif seas_or_wos == 'wos':
        is_seas = False
    else:
        raise ValueError(f"""seas_or_wos should be either 'seas' or 'wos'; got '{seas_or_wos}'""")

    print('OK, starting the procedure')

    # files names
    metadata_file_name = f'current_antecedent_fitting_metadata_{seas_or_wos}.pkl'
    summary_table_name = f'summary_table_{seas_or_wos}'

    res_raw_file_name = f'apf_{seas_or_wos}_raw_result'
    res_big_file_name = f'apf_{seas_or_wos}_result'

    # loading APF metadata
    with open(AP_DIR / f'{metadata_file_name}', 'rb') as file:
        data = pickle.load(file)

    _desc = data['_desc']
    series_name = _desc['series_name']
    train = data['train']
    test = data['test']
    consequent_metaparams = data['consequent_metaparams']
    consequent_params_ini = data['consequent_metaparams']['parameters_ini']
    antecedent_params_set = data['antecedent_params_set']
    clusterization_method = data['clusterization_method']
    local_method = data['local_method']
    data_to_cluster_train = data['data_to_cluster_train']
    data_to_cluster_test = data['data_to_cluster_test']
    cluster_sets_conjunction = data['cluster_sets_conjunction']
    n_last_points_to_use_for_clustering = data['n_last_pts_clustering']
    other_fvm_parameters = data['other_fvm_parameters']

    n_cluster_sets = len(data['clusterization_method'])

    print('Finished loading data')

    # fitting
    result = fit_antecedent_params(train,
                                   test,
                                   consequent_metaparams=consequent_metaparams,
                                   consequent_params_ini=consequent_params_ini,
                                   antecedent_params_set=antecedent_params_set,
                                   clusterization_method=clusterization_method,
                                   local_method=local_method,
                                   data_to_cluster_train=data_to_cluster_train,
                                   data_to_cluster_test=data_to_cluster_test,
                                   cluster_sets_conjunction=cluster_sets_conjunction,
                                   n_last_points_to_use_for_clustering=n_last_points_to_use_for_clustering,
                                   n_cluster_sets=n_cluster_sets,
                                   other_fvm_parameters=other_fvm_parameters,
                                   use_multiprocessing=use_multiprocessing,
                                   do_feeding=do_feeding)

    print('Finished fitting')

    _cur_time = str(pd.Timestamp.today()).replace(':', '-').replace(' ', '_')

    with open(f'{RES_DIR}/{res_raw_file_name}_{_cur_time}', 'wb') as file:
        pickle.dump(result, file)

    res_df = pd.DataFrame.from_records(result).copy()

    if (res_df['status'] != 0).any():
        logger.warning('Not all models were fitted')

    # adding some info
    fitted_antecedent_params = res_df[res_df['rmse'] == res_df['rmse'].min()].iloc[0]['antecedent_params']

    _train_start = str(train.index[0]).split(' ')[0]
    _n_train = train.shape[0]
    _n_test = test.shape[0]
    _n_retrain = other_fvm_parameters['n_points_fitting']

    result_big = {
        'result': result,
        'fitted': fitted_antecedent_params,
        'train': train,
        'test': test,
        'antecedent_params_set': antecedent_params_set,
        'consequent_metaparams': consequent_metaparams,
        'consequent_params_ini': consequent_params_ini,
        'clusterization_method': clusterization_method,
        'local_method': local_method,
        'data_to_cluster_train': data_to_cluster_train,
        'data_to_cluster_test': data_to_cluster_test,
        '_desc': _desc
    }

    _sample_desc = f"""{series_name}_{seas_or_wos}_{_train_start}_{_n_train}_{_n_test}_{_n_retrain}_M={n_last_points_to_use_for_clustering}_at_{_cur_time}.pkl"""
    _file_name = f'{res_big_file_name}_{_sample_desc}_{_cur_time}.pkl'
    with open(RES_DIR / _file_name, 'wb') as file:
        pickle.dump(result_big, file)

    print('Complete')

import pandas as pd
import os
import ast
import seaborn as sns
import matplotlib.pyplot as plt
import os
from utils.util_funcs import *
import gc

class Results_dealer:
    """Collect running results"""
    def __init__(self, dataset, res_fpath):
        self.dataset = dataset
        self.result_fpath = res_fpath

    def _avg_list(self, list):
        return '{:.4f}'.format(sum(list) / len(list))

    def save_model_results(self, res_dict):
        # create important configs
        fname = self.result_fpath + self.dataset + '_results.txt'
        print("Saving results to{}, Final Results".format(fname))
        print_dict(res_dict)
        with open(fname, 'a')as out_file:
            out_file.write('\n' + str(res_dict['Env_name']) + '\n')
            out_file.write(str(res_dict['Train States:']) + '\n')
            out_file.write(str(res_dict['para_dict']) + '\n')
            out_file.write(str(res_dict['exp_results']) + '\n')

    def result_to_exl(self, fname):
        def _res_to_df_dblp(fname):
            with open(fname) as res_file:
                res_lines = res_file.readlines()
            # load result dataframe
            exp_name, exp_settings, a_nmi, a_mi_f1, a_ma_f1, p_nmi, p_mi_f1, p_ma_f1 = {}, {}, {}, {}, {}, {}, {}, {}
            eid = 0
            for line in res_lines:
                if line[0] == 'N':  # Expname
                    tokens = line.strip('\n')
                    eid += 1
                    exp_name[eid] = tokens[5:]
                elif line[0] == '{':  # Experiment Results
                    d = ast.literal_eval(line.strip('\n'))
                    if 'train_time' in d.keys():
                        # Skip train states
                        continue
                    elif 'opt_method' in d.keys():
                        exp_settings[eid] = d
                    elif 'a_nmi' in d.keys():
                        a_nmi[eid] = float(d['a_nmi'])
                        a_mi_f1[eid] = float(d['a_mi_f1'])
                        a_ma_f1[eid] = float(d['a_ma_f1'])
                        p_nmi[eid] = float(d['p_nmi'])
                        p_mi_f1[eid] = float(d['p_mi_f1'])
                        p_ma_f1[eid] = float(d['p_ma_f1'])
                else:
                    continue
            out_list = [exp_name, a_nmi, a_mi_f1, a_ma_f1, p_nmi, p_mi_f1, p_ma_f1, exp_settings]
            out_df = pd.DataFrame.from_records(out_list)
            out_df.rename(index={0: 'exp_name', 1: 'a_nmi', 2: 'a_mi_f1', 3: 'a_ma_f1', 4: 'p_nmi', 5: 'p_mi_f1', 6: 'p_ma_f1',
                                 7: 'exp_settings'}, inplace=True)
            return out_df

        def _res_to_df_default(fname):
            with open(fname) as res_file:
                res_lines = res_file.readlines()
            # load result dataframe
            exp_name, exp_settings, nmi, mi_f1, ma_f1 = {}, {}, {}, {}, {}
            eid = 0
            for line in res_lines:
                if line[0] == 'N':  # Expname
                    tokens = line.strip('\n')
                    eid += 1
                    exp_name[eid] = tokens
                elif line[0] == '{':  # Experiment Results
                    d = ast.literal_eval(line.strip('\n'))
                    if 'train_time' in d.keys():
                        # train states
                        continue
                    elif 'opt_method' in d.keys():
                        exp_settings[eid] = d
                    elif 'nmi' in d.keys():
                        nmi[eid] = float(d['nmi'])
                        mi_f1[eid] = float(d['mi_f1'])
                        ma_f1[eid] = float(d['ma_f1'])
                else:
                    continue
            out_list = [exp_name, nmi, mi_f1, ma_f1, exp_settings]
            out_df = pd.DataFrame.from_records(out_list)
            out_df.rename(index={0: 'exp_name', 1: 'nmi', 2: 'mi_f1', 3: 'ma_f1', 4: 'exp_settings'}, inplace=True)
            return out_df

        fname = self.result_fpath + fname

        if self.dataset[:4] == 'dblp':
            out_df = _res_to_df_dblp(fname)
        else:
            out_df = _res_to_df_default(fname)
        out_df.to_excel(fname[:-4] + '_xl.xlsx')  # Strip .txts
        print('{} saved into excel files'.format(fname))

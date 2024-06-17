import numpy as np
import pandas as pd
from os import path

pd.set_option('display.max_columns', None)


def describe_data_g_targ(dat_df, target_var, logbase=np.e):
    """Describe the data given a target variable

    Parameters
    ----------
    dat_df : DataFrame
        A dataframe that contains the target variable
    target_var : string
        The model target variable, corresponding to a binary variable in dat_df
    logbase : int, float, optional (default=np.e)
        The base for logarithm functions used to compute log-odds - default is natural log (ln = log_e)

    Returns
    -------
    Dict
        A dictionary containing calculated values
    """
    num = dat_df.shape[0]
    n_targ = (dat_df[target_var] == 1).sum()
    n_ctrl = (dat_df[target_var] == 0).sum()
    assert n_ctrl + n_targ == num
    base_rate = n_targ / num
    base_odds = n_targ / n_ctrl
    lbm = 1 / np.log(logbase)
    base_log_odds = np.log(base_odds) * lbm
    NLL_null = -(dat_df[target_var] * np.log(base_rate) * lbm
                 + (1 - dat_df[target_var]) * np.log(
                1 - base_rate) * lbm).sum()
    LogLoss_null = NLL_null / num

    print("Number of records (num):", num)
    print("Target count (n_targ):", n_targ)
    print("Target rate (base_rate):", base_rate)
    print("Target odds (base_odds):", base_odds)
    print("Target log odds (base_log_odds):", base_log_odds)
    print("Dummy model negative log-likelihood (NLL_null):", NLL_null)
    print("Dummy model LogLoss (LogLoss_null):", LogLoss_null)
    print("")
    return {'num': num, 'n_targ': n_targ, 'base_rate': base_rate,
            'base_odds': base_odds
        , 'base_log_odds': base_log_odds, 'NLL_null': NLL_null,
            'LogLoss_null': LogLoss_null}


def one_hot_encode_data(dataset, id_vars, targ_var):
    non_p_vars = id_vars + [targ_var]
    p_vars = dataset.columns.drop(non_p_vars).tolist()
    #     num_p_vars = sorted(dataset[p_vars]._get_numeric_data().columns)
    num_p_vars = ['AgeInMonths', 'DaysSinceDisbursement', 'LTV',
                  'PERFORM_CNS_SCORE']
    cat_p_vars = sorted(set(dataset[p_vars].columns) - set(num_p_vars))

    ## One-hot encode categorical variables
    x_dat = dataset.loc[:, non_p_vars + num_p_vars]
    x_dat_c = dataset.loc[:, cat_p_vars]
    for c in cat_p_vars:
        x_dat_c[c] = x_dat_c[c].astype("category")
    dxx = pd.get_dummies(x_dat_c, prefix_sep='__')
    x_dat.shape, x_dat_c.shape, dxx.shape

    ## Regenerate dataset from components
    dataset = pd.concat([x_dat, dxx], axis=1, sort=False)

    ## Regenerate p_vars to include one-hot encoded variables.
    p_vars = list(dataset.columns.drop(non_p_vars))

    return dataset, p_vars


def load_training_data(run, data_dir, artifact_name:str, train_file='train.csv', val_file='val.csv'):
    data_art = run.use_artifact(artifact_name)
    dataset_dir = data_art.download(data_dir)
    trndat = pd.read_csv(path.join(dataset_dir, train_file))
    valdat = pd.read_csv(path.join(dataset_dir, val_file))
    return trndat, valdat

import wandb
import xgboost as xgb
import pandas as pd
import numpy as np
from os import path
from sklearn import metrics
from wandb.integration.xgboost import WandbCallback
from configparser import ConfigParser
from utils.datautils import (
    describe_data_g_targ, load_training_data, one_hot_encode_data
)
from utils.wandbutils import wandb_run_builder

train_run = wandb_run_builder("train-model")

def train_model():
    config = ConfigParser()
    config.read('config.ini')
    split_dataset = config["wandb"]["split_dataset"]
    targ_var = config["data"]["target_var"]
    data_dir = config["data"]["data_dir"]
    id_vars = [config["data"]["id_vars"]]

    # Load our training data from Artifacts
    trndat, valdat = load_training_data(
        run=train_run,
        data_dir=data_dir,
        artifact_name=split_dataset + ':latest'
    )
    trndict = describe_data_g_targ(trndat, targ_var)
    base_rate = round(trndict['base_rate'], 6)
    early_stopping_rounds = 40
    bst_params = {
        'objective': 'binary:logistic'
        # , 'base_score': base_rate
        , 'base_score': 0.8
        , 'gamma': 0.01  # def: 0
        , 'learning_rate': 1  ## def: 0.1
        , 'max_depth': 4
        , 'min_child_weight': 1  ## def: 1
        , 'n_estimators': 2
        , 'nthread': 24
        , 'random_state': 42
        , 'reg_alpha': 0
        , 'reg_lambda': 0  ## def: 1
        , 'eval_metric': ['auc', 'logloss']
        , 'tree_method': 'hist'  # use `gpu_hist` to train on GPU
    }
    train_run.config.update(dict(bst_params))
    train_run.config.update({'early_stopping_rounds': early_stopping_rounds})

    # Extract target column as a series
    y_trn = trndat.loc[:, targ_var].astype(int)
    y_val = valdat.loc[:, targ_var].astype(int)

    # Initialize the XGBoostClassifier with the WandbCallback
    xgbmodel = xgb.XGBClassifier(
        **bst_params,
        callbacks=[WandbCallback(log_model=True)],
        early_stopping_rounds=train_run.config['early_stopping_rounds']
    )
    # TODO: change this p_var building since it has some leakages.
    dataset = pd.read_csv(path.join(data_dir, 'vehicle_loans_subset.csv'))
    _, p_vars = one_hot_encode_data(dataset, id_vars, targ_var)

    # Train the model
    xgbmodel.fit(
        trndat[p_vars], y_trn,
        eval_set=[(valdat[p_vars], y_val)]
    )
    bstr = xgbmodel.get_booster()

    # Get train and validation predictions
    trnYpreds = xgbmodel.predict_proba(trndat[p_vars])[:, 1]
    valYpreds = xgbmodel.predict_proba(valdat[p_vars])[:, 1]

    # Log additional Train metrics
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(
        y_trn, trnYpreds)
    train_run.summary['train_auc'] = metrics.auc(false_positive_rate,
                                           true_positive_rate)
    train_run.summary['train_log_loss'] = -(
                y_trn * np.log(trnYpreds) + (1 - y_trn) * np.log(
            1 - trnYpreds)).sum() / len(y_trn)

    # Log additional Validation metrics
    train_run.summary["val_auc"] = metrics.roc_auc_score(y_val, valYpreds)
    train_run.summary["val_acc_0.5"] = metrics.accuracy_score(y_val, np.where(
        valYpreds >= 0.5, 1, 0))
    train_run.summary["val_log_loss"] = -(y_val * np.log(valYpreds)
                                    + (1 - y_val) * np.log(
                1 - valYpreds)).sum() / len(y_val)

    # Log the ROC curve to W&B
    valYpreds_2d = np.array(
        [1 - valYpreds, valYpreds])  # W&B expects a 2d array
    y_val_arr = y_val.values
    d = 0
    while len(valYpreds_2d.T) > 10000:
        d += 1
        valYpreds_2d = valYpreds_2d[::1, ::d]
        y_val_arr = y_val_arr[::d]
    train_run.log({"ROC_Curve": wandb.plot.roc_curve(y_val_arr, valYpreds_2d.T,
                                               labels=['no_default',
                                                       'loan_default'],
                                               classes_to_plot=[1])})
    train_run.finish()


if __name__ == "__main__":
    train_model()

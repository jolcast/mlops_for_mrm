import wandb
import xgboost as xgb
import pandas as pd
import numpy as np
from os import path
from wandb.integration.xgboost import WandbCallback
from configparser import ConfigParser
from utils.datautils import (
    describe_data_g_targ, load_training_data, one_hot_encode_data
)
from sklearn import metrics


config = ConfigParser()
config.read('config.ini')
project = config["wandb"]["project"]
split_dataset = config["wandb"]["split_dataset"]
data_dir = config["data"]["data_dir"]
targ_var = config["data"]["target_var"]
id_vars = [config["data"]["id_vars"]]

sweep_config = {
  "method": "random",
  "metric": {"goal": "maximize", "name": "val_acc_0.5"},
  "parameters": {
    "learning_rate": {
      "min": 0.001,
      "max": 1.0
    },
    "gamma": {
      "min": 0.001,
      "max": 1.0
    },
    "min_child_weight": {
      "min": 1,
      "max": 150
    },
    "early_stopping_rounds": {
      "values": [10, 20, 30, 40, 50]
    },
    "n_estimators": {
        "min": 1,
        "max": 50
    },
    "max_depth": {
        "min": 1,
        "max": 20
    },
    "base_score": {
        "min": 0.1,
        "max": 0.9
    }
  }
}

sweep_id = wandb.sweep(sweep_config, project=project)

def search():
    with wandb.init(
            project=config["wandb"]["project"],
            entity=config["wandb"]["entity"],
            job_type="sweep",
    ) as run:
        trndat, valdat = load_training_data(
            run=run,
            data_dir=data_dir,
            artifact_name=split_dataset + ':latest'
        )
        trndict = describe_data_g_targ(trndat, targ_var)
        base_rate = round(trndict['base_rate'], 6)
        bst_params = {
            'objective': 'binary:logistic'
            , 'base_score': run.config['base_score']
            , 'gamma': run.config['gamma']
            , 'learning_rate': run.config['learning_rate']
            , 'max_depth': run.config['max_depth']
            , 'min_child_weight': run.config['min_child_weight']
            , 'n_estimators': run.config['n_estimators']
            , 'nthread': 24
            , 'random_state': 42
            , 'reg_alpha': 0
            , 'reg_lambda': 0          ## def: 1
            , 'eval_metric': ['auc', 'logloss']
            , 'tree_method': 'hist'
        }

        # Initialize the XGBoostClassifier with the WandbCallback
        xgbmodel = xgb.XGBClassifier(
            **bst_params,
            callbacks=[WandbCallback()],
            early_stopping_rounds=run.config['early_stopping_rounds']
        )

        # TODO: change this p_var building since it has some leakages.
        dataset = pd.read_csv(path.join(data_dir, 'vehicle_loans_subset.csv'))
        _, p_vars = one_hot_encode_data(dataset, id_vars, targ_var)

        # Extract target column as a series
        y_trn = trndat.loc[:, targ_var].astype(int)
        y_val = valdat.loc[:, targ_var].astype(int)

        # Train the model
        xgbmodel.fit(
            trndat[p_vars], y_trn, eval_set=[(valdat[p_vars], y_val)]
        )

        bstr = xgbmodel.get_booster()

        # Get train and validation predictions
        trnYpreds = xgbmodel.predict_proba(trndat[p_vars])[:, 1]
        valYpreds = xgbmodel.predict_proba(valdat[p_vars])[:, 1]

        # Log additional Train metrics
        false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_trn, trnYpreds)
        run.summary['train_auc'] = metrics.auc(false_positive_rate, true_positive_rate)
        run.summary['train_log_loss'] = -(y_trn * np.log(trnYpreds) + (1-y_trn) * np.log(1-trnYpreds)).sum() / len(y_trn)

        # Log additional Validation metrics
        run.summary["val_auc"] = metrics.roc_auc_score(y_val, valYpreds)
        run.summary["val_acc_0.5"] = metrics.accuracy_score(y_val, np.where(valYpreds >= 0.5, 1, 0))
        run.summary["val_log_loss"] = -(y_val * np.log(valYpreds)
                                             + (1-y_val) * np.log(1-valYpreds)).sum() / len(y_val)


def search_hyperparameters(count):
    wandb.agent(sweep_id, function=search, count=count)
    wandb.finish()


if __name__ == '__main__':
    search_hyperparameters(50)

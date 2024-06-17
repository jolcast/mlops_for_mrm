from utils.wandbutils import wandb_run_builder, download_registered_model
from utils.datautils import load_training_data, one_hot_encode_data
from sklearn import metrics
from configparser import ConfigParser
import xgboost as xgb
import pandas as pd
from os import path

test_run = wandb_run_builder("test-model")


def test_model():
    config = ConfigParser()
    config.read('config.ini')
    split_dataset = config["wandb"]["split_dataset"]
    targ_var = config["data"]["target_var"]
    data_dir = config["data"]["data_dir"]
    id_vars = [config["data"]["id_vars"]]

    # TODO: change this p_var building since it has some leakages.
    dataset = pd.read_csv(path.join(data_dir, 'vehicle_loans_subset.csv'))
    _, p_vars = one_hot_encode_data(dataset, id_vars, targ_var)

    # Load testing data from Artifacts
    trndat, valdat = load_training_data(
        run=test_run,
        data_dir=data_dir,
        artifact_name=split_dataset + ':latest'
    )

    # Initialize the XGBoostClassifier
    best_registered_model = download_registered_model(
        test_run,
        "vehicle-loan-default-prediction:champion",
        "."
    )
    xgbmodel = xgb.XGBClassifier()
    xgbmodel.load_model("u5aafg9q_model.json")

    y_trn = trndat.loc[:, targ_var].astype(int)
    y_val = valdat.loc[:, targ_var].astype(int)
    trnYpreds = xgbmodel.predict_proba(trndat[p_vars])[:, 1]
    valYpreds = xgbmodel.predict_proba(valdat[p_vars])[:, 1]
    test_run.summary["val_auc"] = metrics.roc_auc_score(y_val, valYpreds)
    test_run.summary["val_auc_traom"] = metrics.roc_auc_score(y_trn, trnYpreds)
    test_run.finish()


if __name__ == "__main__":
    test_model()

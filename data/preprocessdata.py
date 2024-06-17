from utils.datautils import one_hot_encode_data
from utils.commonutils import function_to_string
from configparser import ConfigParser
from os import path
import pandas as pd
import wandb


def preprocess():
    config = ConfigParser()
    config.read("config.ini")
    id_vars = [config["data"]["id_vars"]]
    targ_var = config["data"]["target_var"]
    data_dir = config["data"]["data_dir"]
    project = config["wandb"]["project"]
    initial_dataset = config["wandb"]["initial_dataset"]
    processed_dataset = config["wandb"]["processed_dataset"]

    wandb.login()
    run = wandb.init(project=project, job_type='preprocess-data')
    dataset_art = run.use_artifact(initial_dataset, type='dataset')
    dataset_dir = dataset_art.download(data_dir)
    # Load data into Dataframe
    dataset = pd.read_csv(path.join(data_dir, 'vehicle_loans_subset.csv'))
    # One Hot Encode Data
    dataset, p_vars = one_hot_encode_data(dataset, id_vars, targ_var)
    # Save Preprocessed data
    processed_data_path = path.join(data_dir, 'preprocessed_dataset.csv')
    dataset.to_csv(processed_data_path, index=False)
    # Create a new artifact for the processed data, including the function that created it, to Artifacts
    processed_ds_art = wandb.Artifact(name=processed_dataset,
                                      type='processed_dataset',
                                      description='One-hot encoded dataset',
                                      metadata={
                                          'preprocessing_fn': function_to_string(
                                              one_hot_encode_data)}
                                      )
    # Attach our processed data to the Artifact
    processed_ds_art.add_file(processed_data_path)
    # Log this Artifact to the current wandb run
    run.log_artifact(processed_ds_art)
    run.finish()


if __name__ == '__main__':
    preprocess()


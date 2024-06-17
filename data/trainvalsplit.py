import wandb
import pandas as pd
from configparser import ConfigParser
from os import path
from sklearn import model_selection


def split():
    config = ConfigParser()
    config.read('config.ini')
    targ_var = config["data"]["target_var"]
    data_dir = config["data"]["data_dir"]
    project = config["wandb"]["project"]
    processed_dataset = config["wandb"]["processed_dataset"]
    test_perc = config["data"]["test_size"]
    split_dataset = config["wandb"]["split_dataset"]
    wandb.login()
    with wandb.init(project=project, job_type='train-val-split-data') as run:
        # Download the subset of the vehicle loan default data from W&B
        dataset_art = run.use_artifact(
            processed_dataset + ':latest', type='processed_dataset'
        )
        dataset_dir = dataset_art.download(data_dir)
        dataset = pd.read_csv(path.join(data_dir, 'preprocessed_dataset.csv'))

        # Set Split Params
        test_size = float(test_perc)
        random_state = 42

        # Log the splilt params
        run.config.update(
            {'test_size': test_size, 'random_state': random_state})

        # Do the Train/Val Split
        trndat, valdat = model_selection.train_test_split(
            dataset,
            test_size=test_size,
            random_state=random_state,
            stratify=dataset[[targ_var]]
        )

        print(f'Train dataset size: {trndat[targ_var].value_counts()} \n')
        print(f'Validation dataset sizeL {valdat[targ_var].value_counts()}')

        # Save split datasets
        train_path = path.join(data_dir, 'train.csv')
        val_path = path.join(data_dir, 'val.csv')
        trndat.to_csv(train_path, index=False)
        valdat.to_csv(val_path, index=False)

        # Create a new artifact for the processed data, including the function that created it, to Artifacts
        split_ds_art = wandb.Artifact(
            name=split_dataset,
            type='train-val-dataset',
            description='Processed dataset split into train and validation',
            metadata={'test_size': test_size, 'random_state': random_state}
        )

        # Attach our processed data to the Artifact
        split_ds_art.add_file(train_path)
        split_ds_art.add_file(val_path)

        # Log the Artifact
        run.log_artifact(split_ds_art)


if __name__ == '__main__':
    split()

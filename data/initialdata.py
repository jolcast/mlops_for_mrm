import wandb
import os
from dotenv import load_dotenv

load_dotenv()
WNB_API_KEY = os.environ["WANDB_API_KEY"]

wandb.login(key=WNB_API_KEY)

wandb.init(
    project="mlops-for-mrm",
    job_type="add-dataset",
    entity="personal-jolcastan"
)


if __name__ == "__main__":
    dir_to_log = "data/initial_datasets"

    # Create an artifact
    artifact = wandb.Artifact("vehicle_defaults_raw", type="dataset")

    # Add the directory to the artifact
    artifact.add_dir(dir_to_log)

    # Log the artifact
    wandb.log_artifact(artifact)
    wandb.finish()

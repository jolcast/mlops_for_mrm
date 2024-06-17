import wandb
import os
from configparser import ConfigParser
from dotenv import load_dotenv

load_dotenv()

wandb.login(key=os.environ["WANDB_API_KEY"])
config = ConfigParser()
config.read('config.ini')


def wandb_run_builder(job_type):
    run_build = wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        save_code=False,
        job_type=job_type,
    )
    return run_build


def download_registered_model(run, registry_name_and_tag, local_path):
    entity = config["wandb"]["entity"]
    artifact = run.use_artifact(
        f"{entity}/model-registry/{registry_name_and_tag}", type="model"
    )
    artifact.download(local_path)

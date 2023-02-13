import csv
import os
from pathlib import Path


import torch


from Workflow import Workflow

import util
from parameters.constants import collect_global_constants
import parameters.defaults as defaults

# define what you want to do for the specified job(s)
DATASET          = "test1"    # dataset name in "./data/pre-training/"
JOB_TYPE         = "preprocess"             # "preprocess", "train", "generate", or "test"
JOBDIR_START_IDX = 0                   # where to start indexing job dirs
N_JOBS           = 1                   # number of jobs to run per model
RESTART          = False               # whether or not this is a restart job
FORCE_OVERWRITE  = True                # overwrite job directories which already exist
JOBNAME          = "hc-solves"  # used to create a sub directory


HOME             = str(Path.home())
# PYTHON_PATH      = f"D:/Anaconda/envs/python36/python.exe"
GRAPHINVENT_PATH = "./"
DATA_PATH        = "./datatest/"

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

params = {
    "atom_types"   : ['B', 'C', 'O', 'F', 'S', 'I'],
    "formal_charge": [0, 1],
    "max_n_nodes"  : 20,
    "job_type"     : JOB_TYPE,
    "dataset_dir"  : f"{DATA_PATH}{DATASET}/",
    "restart"      : RESTART,
    "model"        : "GGNN",
    "sample_every" : 2,
    "init_lr"      : 1e-4,
    "epochs"       : 100,
    "batch_size"   : 500,
    "block_size"   : 1000,
    "n_samples"    : 100,
    "device": DEVICE,
}


def submit() -> None:
    """
    Creates and submits submission script. Uses global variables defined at top
    of this file.
    """

    # create an output directory
    dataset_output_path = f"./datatest/output_{DATASET}"

    if JOBNAME != "":
        dataset_output_path = os.path.join(dataset_output_path, JOBNAME)

    os.makedirs(dataset_output_path, exist_ok=True)

    print(f"* Creating dataset directory {dataset_output_path}/", flush=True)

    # submit `N_JOBS` separate jobs
    jobdir_end_idx = JOBDIR_START_IDX + N_JOBS
    for job_idx in range(JOBDIR_START_IDX, jobdir_end_idx):


        # specify and create the job subdirectory if it does not exist
        params["job_dir"]         = f"{dataset_output_path}/job_{job_idx}/"
        # params["tensorboard_dir"] = f"{tensorboard_path}/job_{job_idx}/"

        # create the directory if it does not exist already, otherwise raises an
        # error, which is good because *might* not want to override data our
        # existing directories!
        # os.makedirs(params["tensorboard_dir"], exist_ok=True)
        try:
            job_dir_exists_already = bool(
                JOB_TYPE in ["generate", "test"] or FORCE_OVERWRITE
            )
            os.makedirs(params["job_dir"], exist_ok=job_dir_exists_already)
            print(
                f"* Creating model subdirectory {dataset_output_path}/job_{job_idx}/",
                flush=True,
            )
        except FileExistsError:
            print(
                f"-- Model subdirectory {dataset_output_path}/job_{job_idx}/ already exists.",
                flush=True,
            )
            if not RESTART:
                continue

        # write the `input.csv` file
        write_input_csv(params_dict=params, filename="input.csv")

        # write `submit.sh` and submit

        print("* Running job as a normal process.", flush=True)
        print('2')

        # subprocess.run(["ls", f"{PYTHON_PATH}"], check=True)
        # print('3')
        # subprocess.run([f"{PYTHON_PATH}",
        #                 f"{GRAPHINVENT_PATH}main.py",
        #                 "--job-dir",
        #                 params["job_dir"]],
        #                check=True)



        # constants=Constants(params["job_dir"]).constants
        constants = collect_global_constants(parameters=defaults.parameters,
                                             job_dir=params["job_dir"])


        runmain(constants)


        # sleep a few secs before submitting next job
        # print("-- Sleeping 2 seconds.")
        # time.sleep(2)
        # return constants


def write_input_csv(params_dict : dict, filename : str="params.csv") -> None:
    """
    Writes job parameters/hyperparameters in `params_dict` to CSV using the specified
    `filename`.
    """
    dict_path = params_dict["job_dir"] + filename

    with open(dict_path, "w") as csv_file:

        writer = csv.writer(csv_file, delimiter=";")
        for key, value in params_dict.items():
            writer.writerow([key, value])




def runmain(constants):
    """
    Defines the type of job (preprocessing, training, generation, testing, or
    fine-tuning), writes the job parameters (for future reference), and runs
    the job.
    """
    # _ = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # fix date/time

    workflow = Workflow(constants=constants)

    job_type = constants.job_type
    print(f"* Run mode: '{job_type}'", flush=True)

    if job_type == "preprocess":
        # write preprocessing parameters

        util.write_preprocessing_parameters(params=constants)

        # preprocess all datasets
        workflow.preprocess_phase()


if __name__ == "__main__":
    # collect the constants using the functions defined above

    submit()
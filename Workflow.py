# load general packages and functions
from collections import namedtuple
# import pickle
# from copy import deepcopy
import time
import os
# from typing import Union, Tuple
# import torch
# import torch.utils.tensorboard
# from tqdm import tqdm
# from Analyzer import Analyzer
from GraphBuild import DataProcesser
# import util





class Workflow:
    """
    Single `Workflow` class for carrying out the following processes:
        1) preprocessing various molecular datasets
        2) training generative models
        3) generating molecules using pre-trained models
        4) evaluating generative models
        5) fine-tuning generative models (via RL)

    The preprocessing step reads a set of molecules and generates training data
    for each molecule in HDF file format, consisting of decoding routes and APDs.
    During training, the decoding routes and APDs are used to train graph neural
    network models to generate new APDs, from which actions are stochastically
    sampled and used to build new molecular graphs. During generation, a
    pre-trained model is used to generate a fixed number of structures. During
    evaluation, metrics are calculated for the test set.
    """
    def __init__(self, constants : namedtuple) -> None:

        self.start_time = time.time()
        self.constants  = constants

        # define path variables for various datasets
        self.test_h5_path  = self.constants.test_set[:-3] + "h5"
        self.train_h5_path = self.constants.training_set[:-3] + "h5"
        self.valid_h5_path = self.constants.validation_set[:-3] + "h5"

        # general paramters (placeholders)
        self.optimizer     = None
        self.scheduler     = None
        self.analyzer      = None
        self.current_epoch = None
        self.restart_epoch = None

        # non-reinforcement learning parameters (placeholders)
        self.model                 = None
        self.ts_properties         = None
        self.test_dataloader       = None
        self.train_dataloader      = None
        self.valid_dataloader      = None
        self.likelihood_per_action = None

        # reinforcement learning parameters (placeholders)
        self.agent_model      = None
        self.prior_model      = None
        self.basf_model       = None  # basf stands for "best agent so far"
        self.best_avg_score   = 0.0
        self.rl_step          = 0.0
        self.scoring_function = None

    def preprocess_test_data(self) -> None:
        """
        Converts test dataset to HDF file format.
        """
        if os.path.exists(self.constants.test_set):
            print("* Preprocessing test data.", flush=True)
            test_set_preprocesser = DataProcesser(constants=self.constants,path=self.constants.test_set)
            test_set_preprocesser.preprocess()
        else:
            print('No test.smi')



    def preprocess_train_data(self) -> None:
        """
        Converts training dataset to HDF file format.
        """
        if os.path.exists(self.constants.training_set):
            print("* Preprocessing training data.", flush=True)
            train_set_preprocesser = DataProcesser(constants=self.constants,path=self.constants.training_set,
                                                   is_training_set=True)
            train_set_preprocesser.preprocess()
        else:
            print('No train.smi')



    def preprocess_valid_data(self) -> None:
        """
        Converts validation dataset to HDF file format.
        """
        if os.path.exists(self.constants.validation_set):
            print("* Preprocessing validation data.", flush=True)
            valid_set_preprocesser = DataProcesser(constants=self.constants, path=self.constants.validation_set)
            valid_set_preprocesser.preprocess()
        else:
            print('No value.smi')


    def preprocess_phase(self) -> None:
        """
        Preprocesses all the datasets (validation, training, and testing).
        """
        if not self.constants.restart:
            # start preprocessing job from scratch
            hdf_files_in_data_dir = bool(os.path.exists(self.valid_h5_path)
                                         or os.path.exists(self.test_h5_path)
                                         or os.path.exists(self.train_h5_path))
            if hdf_files_in_data_dir:
                raise OSError(
                    "There currently exist(s) pre-created *.h5 file(s) in the "
                    "dataset directory. If you would like to proceed with "
                    "creating new ones, please delete them and rerun the "
                    "program. Otherwise, check your input file."
                )
            self.preprocess_valid_data()
            self.preprocess_test_data()
            self.preprocess_train_data()

        else:  # restart existing preprocessing job

            # first determine where to restart based on which HDF files have been created
            if (os.path.exists(self.train_h5_path + ".chunked") or
                os.path.exists(self.test_h5_path)):
                print(
                    "-- Restarting preprocessing job from 'train.h5' (skipping "
                    "over 'test.h5' and 'valid.h5' as they seem to be finished).",
                    flush=True,
                )
                self.preprocess_train_data()
            elif (os.path.exists(self.test_h5_path + ".chunked") or
                  os.path.exists(self.valid_h5_path)):
                print(
                    "-- Restarting preprocessing job from 'test.h5' (skipping "
                    "over 'valid.h5' as it appears to be finished).",
                    flush=True,
                )
                self.preprocess_test_data()
                self.preprocess_train_data()
            elif os.path.exists(self.valid_h5_path + ".chunked"):
                print("-- Restarting preprocessing job from 'valid.h5'",
                      flush=True)
                self.preprocess_valid_data()
                self.preprocess_test_data()
                self.preprocess_train_data()
            else:
                raise ValueError(
                    "Warning: Nothing to restart! Check input file and/or "
                    "submission script."
                )
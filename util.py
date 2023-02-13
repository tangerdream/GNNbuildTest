from collections import namedtuple
from typing import Union, Tuple
import torch
import numpy as np

import csv

def write_ts_properties(constants: namedtuple,training_set_properties : dict) -> None:
    """
    Writes the training set properties to CSV.

    Args:
    ----
        training_set_properties (dict) : The properties of the training set.
    """
    training_set = constants.training_set  # path to "train.smi"
    dict_path    = f"{training_set[:-4]}.csv"

    with open(dict_path, "w") as csv_file:

        csv_writer = csv.writer(csv_file, delimiter=";")
        for key, value in training_set_properties.items():
            if "validity_tensor" in key:
                # skip writing the validity tensor here because it is really
                # long, instead it gets its own file elsewhere
                continue
            if isinstance(value, np.ndarray):
                csv_writer.writerow([key, list(value)])
            elif isinstance(value, torch.Tensor):
                try:
                    csv_writer.writerow([key, float(value)])
                except ValueError:
                    csv_writer.writerow([key, [float(i) for i in value]])
            else:
                csv_writer.writerow([key, value])

def write_last_molecule_idx(last_molecule_idx : int, dataset_size : int,
                            restart_file_path : str) -> None:
    """
    Writes the index of the last preprocessed molecule and the current dataset
    size to a file.

    Args:
    ----
        last_molecules_idx (int) : Index of last preprocessed molecule.
        dataset_size (int) : The dataset size.
        restart_file_path (str) : Path indicating where to save indices (should
          be same directory as the dataset).
    """
    with open(restart_file_path + "index.restart", "w") as txt_file:
        txt_file.write(str(last_molecule_idx) + ", " + str(dataset_size))

def write_preprocessing_parameters(params : namedtuple) -> None:
    """
    Writes job parameters/hyperparameters in `params` (`namedtuple`) to
    CSV, so that parameters used during preprocessing can be referenced later.

    Args:
    ----
        params (namedtuple) : Contains job parameters and hyperparameters.
    """
    dict_path = params.dataset_dir + "preprocessing_params.csv"
    keys_to_write = ["atom_types",
                     "formal_charge",
                     "imp_H",
                     "chirality",
                     "group_size",
                     "max_n_nodes",
                     "use_aromatic_bonds",
                     "use_chirality",
                     "use_explicit_H",
                     "ignore_H"]

    with open(dict_path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")
        for key, value in enumerate(params._fields):
            if value in keys_to_write:
                writer.writerow([value, params[key]])



def get_feature_vector_indices(constants: namedtuple) -> list:
    """
    Gets the indices of the different segments of the feature vector. The
    indices are analogous to the lengths of the various segments.

    Returns:
    -------
        idc (list) : Contains the indices of the different one-hot encoded
          segments used in the feature vector representations of nodes in
          `MolecularGraph`s. These segments are, in order, atom type, formal
          charge, number of implicit Hs, and chirality.
    """
    idc = [constants.n_atom_types, constants.n_formal_charge]

    # indices corresponding to implicit H's and chirality are optional (below)
    if not constants.use_explicit_H and not constants.ignore_H:
        idc.append(constants.n_imp_H)

    if constants.use_chirality:
        idc.append(constants.n_chirality)

    return np.cumsum(idc).tolist()#累加


def one_of_k_encoding(x : Union[str, int], allowable_set : list) -> 'generator':
    """
    Returns the one-of-k encoding of a value `x` having a range of possible
    values in `allowable_set`.

    Args:
    ----
        x (str, int) : Value to be one-hot encoded.#单个独热编码
        allowable_set (list) : Contains all possible values to one-hot encode.#包含用于一次性编码的所有可能值。

    Returns:
    -------
        one_hot_generator (generator) : One-hot encoding. A generator of `int`s.
    """
    if x not in set(allowable_set):  # use set for speedup over list
        raise Exception(
            f"Input {x} not in allowable set {allowable_set}. Add {x} to "
            f"allowable set in either a) `features.py` or b) your submission "
            f"script (`submit.py`) and run again."
        )
    one_hot_generator = (int(x == s) for s in allowable_set)
    return one_hot_generator

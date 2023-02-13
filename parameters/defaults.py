# load general packages and functions
import sys

# load GraphINVENT-specific functions
sys.path.insert(1, "./parameters/")  # search "parameters/" directory



# general job parameters
parameters = {
    "atom_types"          : ["C", "N", "O", "S", "Cl"],
    "formal_charge"       : [-1, 0, 1],
    "imp_H"               : [0, 1, 2, 3],
    "chirality"           : ["None", "R", "S"],
    "device"              : "cuda",
    "generation_epoch"    : 30,
    "n_samples"           : 2000,
    "n_workers"           : 2,
    "restart"             : False,
    "max_n_nodes"         : 13,
    "job_type"            : "train",
    "sample_every"        : 10,
    "dataset_dir"         : "data/gdb13_1K/",
    "use_aromatic_bonds"  : False,
    "use_canon"           : True,
    "use_chirality"       : False,
    "use_explicit_H"      : False,
    "ignore_H"            : True,
    "tensorboard_dir"     : "tensorboard/",
    "batch_size"          : 1000,
    "block_size"          : 100000,
    "epochs"              : 100,
    "init_lr"             : 1e-4,
    "max_rel_lr"          : 1,
    "min_rel_lr"          : 0.0001,
    "decoding_route"      : "bfs",
    "activity_model_dir"  : "data/fine-tuning/",
    "score_components"    : ["QED", "drd2_activity", "target_size=13"],
    "score_thresholds"    : [0.5, 0.5, 0.0],  # 0.0 essentially means no threshold
    "score_type"          : "binary",
    "qsar_models"         : {"drd2_activity": "data/fine-tuning/qsar_model.pickle"},
    "pretrained_model_dir": "output/",
    "sigma"               : 20,
    "alpha"               : 0.5,
}



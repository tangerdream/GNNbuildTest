"""
`MolecularGraph.py` defines the parent MolecularGraph class and three sub-classes.
"""
# load general packages and functions
from collections import namedtuple
import itertools
import random
from copy import deepcopy
from typing import Union, Tuple
import numpy as np
import torch
import rdkit
from rdkit.Chem.rdmolfiles import MolToSmiles
import h5py
import os
from tqdm import tqdm

# load GraphINVENT-specific functions
import util
from Analyzer import Analyzer
import parameters.load as load

class MolecularGraph:
    """
    Parent class for all molecular graphs.

    This class is then inherited by three subclasses:
      `PreprocessingGraph`, which is used when preprocessing training data, and
      `TrainingGraph`, which is used when training structures.
      `GenerationGraph`, which is used when generating structures.

    The reason for the two classes is that `np.ndarray`s are needed to save
    test/train/valid sets to HDF file format when preprocessing, but for
    training it is more efficient to use `torch.Tensor`s since these can be
    easily used on the GPU for training/generation.
    """
    def __init__(self, constants : namedtuple,
                 molecule : rdkit.Chem.Mol ,
                 node_features : Union[np.ndarray, torch.Tensor],
                 edge_features : Union[np.ndarray, torch.Tensor],
                 atom_feature_vector : torch.Tensor) -> None:
        """
        Args:
        ----
            constants (namedtuple)             : Contains job parameters as
                                                 well as global constants.
            molecule (rdkit.Chem.Mol)          : Input used for creating
                                                 `PreprocessingGraph`.
            atom_feature_vector (torch.Tensor) : Input used for creating
                                                 `TrainingGraph`.
        """
        self.constants = constants

        # placeholders (these are set in the respective sub-classes)
        self.molecule      = None
        self.node_features = None
        self.edge_features = None
        self.n_nodes       = None

    def get_smiles(self) -> str:
        """
        Gets the SMILES representation of the current `MolecularGraph`.
        """
        try:
            smiles = MolToSmiles(mol=self.molecule, kekuleSmiles=False)
        except:
            # if molecule is invalid, set SMILES to `None`
            smiles = None
        return smiles

    def get_n_edges(self) -> int:
        """
        Gets the number of edges in the `MolecularGraph`.
        """
        # divide by 2 to avoid double-counting edges
        n_edges = self.edge_features.sum() // 2
        return n_edges

    def mol_to_graph(self, molecule : rdkit.Chem.Mol) -> None:#分子->graph转化
        """
        Generates the graph representation (`self.node_features` and
        `self.edge_features`) when creating a new `PreprocessingGraph`.
        """
        n_atoms = self.n_nodes
        atoms   = map(molecule.GetAtomWithIdx, range(n_atoms))

        # build the node features matrix using a Numpy array
        node_features = np.array(list(map(self.atom_features, atoms)),
                                 dtype=np.int32)

        # build the edge features tensor using a Numpy array
        edge_features = np.zeros(
            [n_atoms, n_atoms, self.constants.n_edge_features],
            dtype=np.int32
        )
        for bond in molecule.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = self.constants.bondtype_to_int[bond.GetBondType()]
            edge_features[i, j, bond_type] = 1
            edge_features[j, i, bond_type] = 1

        # define the number of nodes
        self.n_nodes = n_atoms

        self.node_features = node_features  # not padded!
        self.edge_features = edge_features  # not padded!

# 预处理进程
class PreprocessingGraph(MolecularGraph):
    """
    Class for molecular graphs to be used during the data preprocessing phase.
    Uses `np.ndarray`s for the graph attributes, so they can be stored as HDF5
    datasets. These are never loaded onto the GPU.
    """
    def __init__(self, constants : namedtuple,
                 molecule : rdkit.Chem.Mol) -> None:
        super().__init__(constants, molecule=False, node_features=False,
                         edge_features=False, atom_feature_vector=False)

        # define values previously set to `None` or undefined
        self.node_ordering = None  # to be defined in `self.node_remap()`

        if self.constants.use_explicit_H and not self.constants.ignore_H:
            molecule = rdkit.Chem.AddHs(molecule)

        self.n_nodes = molecule.GetNumAtoms()

        # get the graph attributes from the `rdkit.Chem.Mol()` object
        self.mol_to_graph(molecule=molecule)

        # remap the nodes using either a canonical or random node ordering
        self.node_remap(molecule=molecule)

        # pad up to size of largest graph in dataset (`self.constants.max_n_nodes`)
        self.pad_graph_representation()

# 原子特征描述
    def atom_features(self, atom : rdkit.Chem.Atom) -> np.ndarray:
        """
        Generates the feature vector for a given node on a molecular graph. Uses
        the following descriptors to recreate the molecular graph: atom type,
        formal charge, and, if specified, number of implicit Hs and chirality.
        The first two descriptors are the bare minimum needed.

        Args:
        ----
            atom (rdkit.Chem.Atom) : Atom in molecule for which to get feature
                                     vector.

        Returns:
        -------
            feature_vector (numpy.ndarray) : Corresponding feature vector.
        """
        feature_vector_generator = itertools.chain(
            util.one_of_k_encoding(atom.GetSymbol(), self.constants.atom_types),
            util.one_of_k_encoding(atom.GetFormalCharge(),
                                   self.constants.formal_charge)
        )
        if not self.constants.use_explicit_H and not self.constants.ignore_H:
            feature_vector_generator = itertools.chain(
                feature_vector_generator,
                util.one_of_k_encoding(atom.GetTotalNumHs(),
                                       self.constants.imp_H)
            )
        if self.constants.use_chirality:
            try:
                chiral_state = atom.GetProp("_CIPCode")
            except KeyError:
                chiral_state = self.constants.chirality[0]  # "None"

            feature_vector_generator = itertools.chain(
                feature_vector_generator,
                util.one_of_k_encoding(chiral_state, self.constants.chirality)
            )

        feature_vector = np.fromiter(feature_vector_generator, int)

        return feature_vector


# 广度搜索
    def breadth_first_search(self, node_ranking : list,
                             node_init : int=0) -> list:
        """
        Starting from the specified `node_init` in the graph, uses a breadth-
        first search (BFS) algorithm to find all adjacent nodes, returning an
        ordered list of these nodes. Prioritizes the nodes based on the input
        `node_ranking`. The function uses the edge feature tensor to find
        adjacent nodes.

        Args:
        ----
            node_ranking (list) : Contains the ranking of all the nodes in the
                                  graph (e.g. the canonical RDKit node ranking,
                                  or a random ranking).
            node_init (int)     : Index of node to start the BFS from. Default 0.

        Returns:
        -------
            nodes_visited (list) : BFS ordering for nodes in the molecular graph.
        """
        nodes_visited      = [node_init]
        last_nodes_visited = [node_init]

        # loop until all nodes have been visited
        while len(nodes_visited) < self.n_nodes:
            neighboring_nodes = []

            for node in last_nodes_visited:
                neighbor_nodes = []
                for bond_type in range(self.constants.n_edge_features):
                    neighbor_nodes.extend(list(
                        np.nonzero(self.edge_features[node, :, bond_type])[0]
                    ))
                new_neighbor_nodes = list(
                    set(neighbor_nodes) - (set(neighbor_nodes) & set(nodes_visited))
                )
                node_importance    = [node_ranking[neighbor_node] for
                                      neighbor_node in new_neighbor_nodes]

                # check all neighboring nodes and sort in order of importance
                while sum(node_importance) != -len(node_importance):
                    next_node = node_importance.index(max(node_importance))
                    neighboring_nodes.append(new_neighbor_nodes[next_node])
                    node_importance[next_node] = -1

            # append the new, sorted neighboring nodes to list of visited nodes
            nodes_visited.extend(set(neighboring_nodes))

            # update the list of most recently visited nodes
            last_nodes_visited = set(neighboring_nodes)

        return nodes_visited

    def depth_first_search(self, node_ranking : list, node_init : int=0) -> list:
        """
        Starting from the specified `node_init` in the graph, uses a depth-first
        search (DFS) algorithm to find the longest branch nodes, returning an
        ordered list of the nodes traversed. Prioritizes the nodes based on the
        input `node_ranking`. The function uses the edge feature tensor to find
        adjacent nodes.

        Args:
        ----
            node_ranking (list) : Contains the ranking of all the nodes in the
                                  graph (e.g. the canonical RDKit node ranking,
                                  or a random ranking).
            node_init (int)     : Index of node to start the DFS from. Default 0.

        Returns:
        -------
            nodes_visited (list) : DFS ordering for nodes in the molecular graph.
        """
        nodes_visited     = [node_init]
        last_node_visited = node_init

        # loop until all nodes have been visited
        while len(nodes_visited) < self.n_nodes:

            neighbor_nodes = []
            for bond_type in range(self.constants.n_edge_features):
                neighbor_nodes.extend(list(
                    np.nonzero(self.edge_features[last_node_visited, :, bond_type])[0]
                ))
            new_neighbor_nodes = list(
                set(neighbor_nodes) - (set(neighbor_nodes) & set(nodes_visited))
            )

            if not new_neighbor_nodes:  # list is empty
                # backtrack if there are no "new" neighbor nodes i.e. reached end of branch
                current_node_idx  = nodes_visited.index(last_node_visited)
                last_node_visited = nodes_visited[current_node_idx - 1]
                continue

            node_importance = [node_ranking[neighbor_node] for
                               neighbor_node in new_neighbor_nodes]

            # get the most important of the neighboring nodes
            most_important_neighbor_node = node_importance.index(max(node_importance))

            # append the new, sorted neighboring nodes to list of visited nodes
            nodes_visited.extend([new_neighbor_nodes[most_important_neighbor_node]])

            # update the most recently visited node
            last_node_visited = new_neighbor_nodes[most_important_neighbor_node]

        return nodes_visited

    def node_remap(self, molecule : rdkit.Chem.Mol) -> None:
        """
        Remaps nodes in `rdkit.Chem.Mol` object (`molecule`) either randomly, or
        using RDKit's canonical node ordering. This depends on if `use_canon` is
        specified or not.
        """
        if not self.constants.use_canon:
            # get a *random* node ranking
            atom_ranking = list(range(self.n_nodes))
            random.shuffle(atom_ranking)
        else:
            # get RDKit canonical ranking
            atom_ranking = list(
                rdkit.Chem.CanonicalRankAtoms(molecule, breakTies=True)
            )

        # using a random node as a starting point, get a new node ranking that
        # does not leave isolated fragments in graph traversal
        if self.constants.decoding_route == "bfs":
            self.node_ordering = self.breadth_first_search(node_ranking=atom_ranking,
                                                           node_init=atom_ranking[0])
        elif self.constants.decoding_route == "dfs":
            self.node_ordering = self.depth_first_search(node_ranking=atom_ranking,
                                                         node_init=atom_ranking[0])

        # reorder all nodes according to new node ranking
        self.reorder_nodes()

    def get_decoding_APD(self) -> np.ndarray:
        """
        For a given subgraph along a decoding route for a `PreprocessingGraph`,
        computes the target decoding APD that would take you to the next
        subgraph (adding one edge/node). Used when generating the training data.

        Returns:
        -------
            The graph decoding APD, comprised of the following probability values:

            f_add (numpy.ndarray)  : Add APD. Size Mx|A|x|F|x|H|x|B| tensor whose
                                     elements are the probabilities of adding a new
                                     atom of type a with formal charge f and implicit
                                     Hs h to existing atom v with a new bond of type
                                     b. If `use_chirality`==True, it is instead a
                                     size Mx|A|x|F|x|H|x|C|x|B| tensor, whose elements
                                     are the probabilities of adding such an atom
                                     with chiral state c.
            f_conn (numpy.ndarray) : Connect APD. Size |V|x|B| matrix, whose
                                     elements are the probability of connecting
                                     the last appended atom with existing atom v
                                     using a new bond of type b.
            f_term (int)           : Terminate APD. Scalar indicating probability
                                     of terminating the graph.

            M is the maximum number of nodes in a graph in any set (train, test,
            val), A is the set of atom types, F is the set of formal charges,
            H is the set of implicit Hs, C is the set of chiral states, and B
            is the set of bond types.
        """
        # determine the indices of the atom descriptors (e.g. atom type)
        last_node_idx  = self.n_nodes - 1  # zero-indexing
        fv_nonzero_idc = self.get_nonzero_feature_indices(node_idx=last_node_idx)

        # initialize action probability distribution (APD)
        f_add  = np.zeros(self.constants.dim_f_add, dtype=np.int32)
        f_conn = np.zeros(self.constants.dim_f_conn, dtype=np.int32)

        # determine which nodes are bonded
        bonded_nodes = []
        for bond_type in range(self.constants.n_edge_features):
            bonded_nodes.extend(list(
                np.nonzero(self.edge_features[:, last_node_idx, bond_type])[0]
            ))

        if bonded_nodes:
            degree            = len(bonded_nodes)
            v_idx             = bonded_nodes[-1]  # idx of node to form bond with
            bond_type_forming = int(
                np.nonzero(self.edge_features[v_idx, last_node_idx, :])[0]
            )

            if degree > 1:
                # if multiple bonds to one node first add bonds one by one
                # (modify `f_conn`)
                f_conn[v_idx, bond_type_forming] = 1
            else:
                # if only bound to one node, bond and node addition is one move
                # (modify `f_add`)
                f_add[tuple([v_idx] + fv_nonzero_idc + [bond_type_forming])] = 1
        else:
            # if it is the last node in the graph, node addition occurs in one
            # move (modify `f_add`); uses a dummy edge to "connect" to node 0
            f_add[tuple([0] + fv_nonzero_idc + [0])] = 1

        # concatenate `f_add`, `f_conn`, and `f_term` (`f_term`==0)
        apd = np.concatenate((f_add.ravel(), f_conn.ravel(), np.array([0])))
        return apd

    def get_final_decoding_APD(self) -> np.ndarray:
        """
        For a given subgraph along a decoding route for a `PreprocessingGraph`,
        computes the target decoding APD that would indicate termination. Used
        when generating the training data.

        Returns:
        -------
            The graph decoding APD, comprised of the following probability
            values (see `get_decoding_APD()` docstring above):

            f_add (numpy.ndarray)  : Add APD.
            f_conn (numpy.ndarray) : Connect APD.
            f_term (int)           : Terminate APD. Scalar (1, since terminating)
                                     indicating the probability of terminating
                                     the graph generation.
        """
        # initialize action probability distribution (APD)
        f_add  = np.zeros(self.constants.dim_f_add, dtype=np.int32)
        f_conn = np.zeros(self.constants.dim_f_conn, dtype=np.int32)

        # concatenate `f_add`, `f_conn`, and `f_term` (`f_term`==0)
        apd = np.concatenate((f_add.ravel(), f_conn.ravel(), np.array([1])))
        return apd

    def get_graph_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the graph representation of the current `PreprocessingGraph`.
        """
        return self.node_features, self.edge_features

    def get_nonzero_feature_indices(self, node_idx : int) -> list:
        """
        Gets indices of the nonzero values in a one-hot encoded atomic feature
        vector (for converting a feature vector into an `rdkit.Chem.Atom`
        object).

        Args:
        ----
            node_idx (int) : Index for a specific node in the `PreprocessingGraph`.

        Returns:
        -------
            segment_idc (list) : Contains the nonzero indices of the atom type,
                                 formal charge, number of implicit Hs, and chirality
                                 that describe a specific node in a feature vector.
                                 The indices are "corrected" for each one-hot encoded
                                 segment.
        """
        fv_idc = util.get_feature_vector_indices(self.constants)
        idc    = np.nonzero(self.node_features[node_idx])[0]

        # correct for the concatenation of the different segments of each node
        # feature vector
        segment_idc = [idc[0]]
        for idx, value in enumerate(idc[1:]):
            segment_idc.append(value - fv_idc[idx])

        return segment_idc

    def reorder_nodes(self) -> None:
        """
        Remaps the numerical ordering of nodes in the graph as specified by the
        `self.node_ordering`. Modifies the `PreprocessingGraph` in place.
        """
        # first remap the node features matrix
        node_features_remapped = np.array(
            [self.node_features[node] for node in self.node_ordering],
            dtype=np.int32
        )

        # then remap the edge features tensor
        edge_features_rows_done = np.array(
            [self.edge_features[node, :, :] for node in self.node_ordering],
            dtype=np.int32
        )
        edge_features_remapped = np.array(
            [edge_features_rows_done[:, node, :] for node in self.node_ordering],
            dtype=np.int32
        )

        self.node_features = node_features_remapped
        self.edge_features = edge_features_remapped

    def pad_graph_representation(self) -> None:
        """
        Pads arrays to size corresponding to largest graph in training, testing,
        and validation datasets.
        """
        # initialize the padded graph representation arrays
        node_features_padded = np.zeros((self.constants.max_n_nodes,
                                         self.constants.n_node_features))
        edge_features_padded = np.zeros((self.constants.max_n_nodes,
                                         self.constants.max_n_nodes,
                                         self.constants.n_edge_features))

        # pad up to size of largest graph
        node_features_padded[:self.n_nodes, :]                = self.node_features
        edge_features_padded[:self.n_nodes, :self.n_nodes, :] = self.edge_features

        self.node_features = node_features_padded
        self.edge_features = edge_features_padded

    def truncate_graph(self) -> None:
        """
        Truncates a molecule by editing its molecular graph (`self.node_features`
        and `self.edge_features`) in place. By default deletes the last node.

        If the last atom is bound to multiple atoms on the graph (i.e. a ring
        atom), then only deletes the least "important" bond, as determined from
        the breadth-first ordering. This is so as to allow the APD to be broken
        up into multiple steps (add, connect, terminate).
        """
        last_atom_idx = self.n_nodes - 1

        if self.n_nodes == 1:
            # remove the last atom
            self.node_features[last_atom_idx, :] = 0
            self.n_nodes                        -= 1
        else:
            # determine how many bonds on the least important atom
            bond_idc = []
            for bond_type in range(self.constants.n_edge_features):
                bond_idc.extend(
                    list(
                        np.nonzero(self.edge_features[:, last_atom_idx, bond_type])[0]
                    )
                )

            degree = len(bond_idc)

            if degree == 1:
                # delete atom from node features
                self.node_features[last_atom_idx, :] = 0
                self.n_nodes -= 1
            else:  # if degree > 1
                # if the last atom is bound to multiple atoms, only delete the
                # least important bond, but leave the atom and remaining bonds
                bond_idc = bond_idc[-1]  # mark bond for deletion (below)

            # delete bond from row feature tensor (first row, then column)
            self.edge_features[bond_idc, last_atom_idx, :] = 0
            self.edge_features[last_atom_idx, bond_idc, :] = 0

    def get_decoding_route_length(self) -> int:
        """
        Returns the number of subgraphs in the graph's decoding route, which is
        how many subgraphs would be formed in the process of deleting the last
        atom/bond in the molecule stepwise until only a single atom is left.
        Note that this is simply the number of edges plus two, since each action
        adds an edge, plus we have the initial and final actions.

        Returns:
        -------
            n_decoding_graphs (int) : Number of subgraphs in the input graph's
                                      decoding route.
        """
        return int(self.get_n_edges() + 2)

    def get_decoding_route_state(self, subgraph_idx : int) -> \
                                 Tuple[list, np.ndarray]:
        """
        Starting from the specified graph, returns the state (subgraph and
        decoding APD) indicated by `subgraph_idx` along the decoding route.

        Args:
        ----
            subgraph_idx (int) : Index of subgraph along decoding route.

        Returns:
        -------
            decoding_graph (list)      : Graph representation, structured as [X, E].
            decoding_APDs (np.ndarray) : Contains the decoding APD, structured as
                                         a concatenation of flattened (f_add, f_conn,
                                         f_term).
        """
        molecular_graph = deepcopy(self)

        if subgraph_idx != 0:
            # find which subgraph is indicated by the index by progressively
            # truncating the input molecular graph
            for _ in range(1, subgraph_idx):
                molecular_graph.truncate_graph()

            # get the APD before the last truncation (since APD says how to get
            # to the next graph, need to truncate once more after obtaining APD)
            decoding_APD = molecular_graph.get_decoding_APD()
            molecular_graph.truncate_graph()
            X, E         = molecular_graph.get_graph_state()

        elif subgraph_idx == 0:
            # return the first subgraph
            decoding_APD = molecular_graph.get_final_decoding_APD()
            X, E         = molecular_graph.get_graph_state()

        else:
            raise ValueError("`subgraph_idx` not a valid value.")

        decoding_graph = [X, E]

        return decoding_graph, decoding_APD


class DataProcesser:
    """
    A class for preprocessing molecular sets and writing them to HDF files.
    """
    def __init__(self,constants : namedtuple, path : str, is_training_set : bool=False) -> None:
        """
        Args:
        ----
            path (string)          : Full path/filename to SMILES file containing
                                     molecules.
            is_training_set (bool) : Indicates if this is the training set, as we
                                     calculate a few additional things for the training
                                     set.
        """
        # define some variables for later use
        self.constants = constants
        self.path            = path
        self.is_training_set = is_training_set
        self.dataset_names   = ["nodes", "edges", "APDs"]
        self.get_dataset_dims()  # creates `self.dims`

        # load the molecules
        self.molecule_set = load.molecules(self.path)

        # placeholders
        self.molecule_subset    = None
        self.dataset            = None
        self.skip_collection    = None
        self.resume_idx         = None
        self.ts_properties      = None
        self.restart_index_file = None
        self.hdf_file           = None
        self.dataset_size       = None

        # get total number of molecules, and total number of subgraphs in their
        # decoding routes
        self.n_molecules       = len(self.molecule_set)
        self.total_n_subgraphs = self.get_n_subgraphs()
        print(f"-- {self.n_molecules} molecules in set.", flush=True)
        print(f"-- {self.total_n_subgraphs} total subgraphs in set.",
              flush=True)

    def preprocess(self) -> None:
        """
        Prepares an HDF file to save three different datasets to it (`nodes`,
        `edges`, `APDs`), and slowly fills it in by looping over all the
        molecules in the data in groups (or "mini-batches").
        """
        if os.path.exists(f"{self.path[:-3]}h5.chunked"):
            os.remove(f"{self.path[:-3]}h5.chunked")
        with h5py.File(f"{self.path[:-3]}h5.chunked", "a") as self.hdf_file:

            self.restart_index_file = self.constants.dataset_dir + "index.restart"

            if self.constants.restart and os.path.exists(self.restart_index_file):
                self.restart_preprocessing_job()
            else:
                self.start_new_preprocessing_job()

                # keep track of the dataset size (to resize later)
                self.dataset_size = 0

            self.ts_properties = None

            # this is where we fill the datasets with actual data by looping
            # over subgraphs in blocks of size `constants.batch_size`
            for idx in range(0, self.total_n_subgraphs, self.constants.batch_size):

                if not self.skip_collection:

                    self.get_molecule_subset()

                    # add `constants.batch_size` subgraphs from
                    # `self.molecule_subset` to the dataset (and if training
                    # set, calculate their properties and add these to
                    # `self.ts_properties`)
                    self.get_subgraphs(init_idx=idx)

                    util.write_last_molecule_idx(
                        last_molecule_idx=self.resume_idx,
                        dataset_size=self.dataset_size,
                        restart_file_path=self.constants.dataset_dir
                    )


                if self.resume_idx == self.n_molecules:
                    # all molecules have been processed

                    self.resize_datasets()  # remove padding from initialization
                    print("Datasets resized.", flush=True)

                    if self.is_training_set and not self.constants.restart:

                        print("Writing training set properties.", flush=True)
                        util.write_ts_properties(constants=self.constants,
                            training_set_properties=self.ts_properties
                        )

                    break

        print("* Resaving datasets in unchunked format.")
        self.resave_datasets_unchunked()

    def restart_preprocessing_job(self) -> None:
        """
        Restarts a preprocessing job. Uses an index specified in the dataset
        directory to know where to resume preprocessing.
        """
        try:
            self.resume_idx, self.dataset_size = util.read_last_molecule_idx(
                restart_file_path=self.constants.dataset_dir
            )
        except:
            self.resume_idx, self.dataset_size = 0, 0
        self.skip_collection = bool(
            self.resume_idx == self.n_molecules and self.is_training_set
        )

        # load dictionary of previously created datasets (`self.dataset`)
        self.load_datasets(hdf_file=self.hdf_file)

    def start_new_preprocessing_job(self) -> None:
        """
        Starts a fresh preprocessing job.
        """
        self.resume_idx      = 0
        self.skip_collection = False

        # create a dictionary of empty HDF datasets (`self.dataset`)
        self.create_datasets(hdf_file=self.hdf_file)

    def resave_datasets_unchunked(self) -> None:
        """
        Resaves the HDF datasets in an unchunked format to remove initial
        padding.
        """
        with h5py.File(f"{self.path[:-3]}h5.chunked", "r", swmr=True) as chunked_file:
            keys        = list(chunked_file.keys())
            data        = [chunked_file.get(key)[:] for key in keys]
            data_zipped = tuple(zip(data, keys))

            with h5py.File(f"{self.path[:-3]}h5", "w") as unchunked_file:
                for d, k in tqdm(data_zipped):
                    unchunked_file.create_dataset(
                        k, chunks=None, data=d, dtype=np.dtype("int8")
                    )

        # remove the restart file and chunked file (don't need them anymore)
        os.remove(self.restart_index_file)
        os.remove(f"{self.path[:-3]}h5.chunked")

    def get_subgraphs(self, init_idx : int) -> None:
        """
        Adds `constants.batch_size` subgraphs from `self.molecule_subset` to the
        HDF dataset (and if currently processing the training set, also
        calculates the full graphs' properties and adds these to
        `self.ts_properties`).

        Args:
        ----
            init_idx (int) : As analysis is done in blocks/slices, `init_idx` is
                             the start index for the next block/slice to be taken
                             from `self.molecule_subset`.
        """
        data_subgraphs, data_apds, molecular_graph_list = [], [], []  # initialize

        # convert all molecules in `self.molecules_subset` to `PreprocessingGraphs`
        molecular_graph_generator = map(self.get_graph, self.molecule_subset)

        molecules_processed       = 0  # keep track of the number of molecules processed

        # loop over all the `PreprocessingGraph`s
        for graph in molecular_graph_generator:
            molecules_processed += 1

            # store `PreprocessingGraph` object
            molecular_graph_list.append(graph)

            # get the number of decoding graphs
            n_subgraphs = graph.get_decoding_route_length()

            for new_subgraph_idx in range(n_subgraphs):

                # `get_decoding_route_state() returns a list of [`subgraph`, `apd`],
                subgraph, apd = graph.get_decoding_route_state(
                    subgraph_idx=new_subgraph_idx
                )

                # "collect" all APDs corresponding to pre-existing subgraphs,
                # otherwise append both new subgraph and new APD
                count = 0
                for idx, existing_subgraph in enumerate(data_subgraphs):

                    count += 1
                    # check if subgraph `subgraph` is "already" in
                    # `data_subgraphs` as `existing_subgraph`, and if so, add
                    # the "new" APD to the "old"
                    try:  # first compare the node feature matrices
                        nodes_equal = (subgraph[0] == existing_subgraph[0]).all()
                    except AttributeError:
                        nodes_equal = False
                    try:  # then compare the edge feature tensors
                        edges_equal = (subgraph[1] == existing_subgraph[1]).all()
                    except AttributeError:
                        edges_equal = False

                    # if both matrices have a match, then subgraphs are the same
                    if nodes_equal and edges_equal:
                        existing_apd = data_apds[idx]
                        existing_apd += apd
                        break

                # if subgraph is not already in `data_subgraphs`, append it
                if count == len(data_subgraphs) or count == 0:
                    data_subgraphs.append(subgraph)
                    data_apds.append(apd)

                # if `constants.batch_size` unique subgraphs have been
                # processed, save group to the HDF dataset
                len_data_subgraphs = len(data_subgraphs)
                if len_data_subgraphs == self.constants.batch_size:
                    self.save_group(data_subgraphs=data_subgraphs,
                                    data_apds=data_apds,
                                    group_size=len_data_subgraphs,
                                    init_idx=init_idx)

                    # get molecular properties for group iff it's the training set
                    self.get_ts_properties(molecular_graphs=molecular_graph_list,
                                           group_size=self.constants.batch_size)

                    # keep track of the last molecule to be processed in
                    # `self.resume_idx`
                    # number of molecules processed:
                    self.resume_idx   += molecules_processed
                    # subgraphs processed:
                    self.dataset_size += self.constants.batch_size

                    return None

        n_processed_subgraphs = len(data_subgraphs)

        # save group with < `constants.batch_size` subgraphs (e.g. last block)
        self.save_group(data_subgraphs=data_subgraphs,
                        data_apds=data_apds,
                        group_size=n_processed_subgraphs,
                        init_idx=init_idx)

        # get molecular properties for this group iff it's the training set
        self.get_ts_properties(molecular_graphs=molecular_graph_list,
                               group_size=self.constants.batch_size)

        # keep track of the last molecule to be processed in `self.resume_idx`
        self.resume_idx   += molecules_processed  # number of molecules processed
        self.dataset_size += molecules_processed  # subgraphs processed

        return None

    def create_datasets(self, hdf_file : h5py._hl.files.File) -> None:
        """
        Creates a dictionary of HDF5 datasets (`self.dataset`).

        Args:
        ----
            hdf_file (h5py._hl.files.File) : HDF5 file which will contain the datasets.
        """
        self.dataset = {}  # initialize

        for ds_name in self.dataset_names:
            self.dataset[ds_name] = hdf_file.create_dataset(
                ds_name,
                (self.total_n_subgraphs, *self.dims[ds_name]),
                chunks=True,  # must be True for resizing later
                dtype=np.dtype("int8")
            )

    def resize_datasets(self) -> None:
        """
        Resizes the HDF datasets, since much longer datasets are initialized
        when first creating the HDF datasets (it it is impossible to predict
        how many graphs will be equivalent beforehand).
        """
        for dataset_name in self.dataset_names:
            try:
                self.dataset[dataset_name].resize(
                    (self.dataset_size, *self.dims[dataset_name]))
            except KeyError:  # `f_term` has no extra dims
                self.dataset[dataset_name].resize((self.dataset_size,))

    def get_dataset_dims(self) -> None:
        """
        Calculates the dimensions of the node features, edge features, and APDs,
        and stores them as lists in a dict (`self.dims`), where keys are the
        dataset name.

        Shapes:
        ------
            dims["nodes"] : [max N nodes, N atom types + N formal charges]
            dims["edges"] : [max N nodes, max N nodes, N bond types]
            dims["APDs"]  : [APD length = f_add length + f_conn length + f_term length]
        """
        self.dims = {}
        self.dims["nodes"] = self.constants.dim_nodes
        self.dims["edges"] = self.constants.dim_edges
        self.dims["APDs"]  = self.constants.dim_apd

    def get_graph(self, mol : rdkit.Chem.Mol) -> PreprocessingGraph:
        """
        Converts an `rdkit.Chem.Mol` object to `PreprocessingGraph`.

        Args:
        ----
            mol (rdkit.Chem.Mol) : Molecule to convert.

        Returns:
        -------
            molecular_graph (PreprocessingGraph) : Molecule, now as a graph.
        """
        if mol is not None:
            if not self.constants.use_aromatic_bonds:
                rdkit.Chem.Kekulize(mol, clearAromaticFlags=True)
            molecular_graph = PreprocessingGraph(molecule=mol,
                                                 constants=self.constants)
        return molecular_graph

    def get_molecule_subset(self) -> None:
        """
        Slices `self.molecule_set` into a subset of molecules of size
        `constants.batch_size`, starting from `self.resume_idx`.
        `self.n_molecules` is the number of molecules in the full
        `self.molecule_set`.
        """
        init_idx             = self.resume_idx
        subset_size          = self.constants.batch_size
        self.molecule_subset = []
        max_idx              = min(init_idx + subset_size, self.n_molecules)

        count = -1
        for mol in self.molecule_set:
            if mol is not None:
                count += 1
                if count < init_idx:
                    continue
                elif count >= max_idx:
                    return self.molecule_subset
                else:
                    self.molecule_subset.append(mol)

    def get_n_subgraphs(self) -> int:
        """
        Calculates the total number of subgraphs in the decoding route of all
        molecules in `self.molecule_set`. Loads training, testing, or validation
        set. First, the `PreprocessingGraph` for each molecule is obtained, and
        then the length of the decoding route is trivially calculated for each.

        Returns:
        -------
            n_subgraphs (int) : Sum of number of subgraphs in decoding routes for
                                all molecules in `self.molecule_set`.
        """
        n_subgraphs = 0  # start the count

        # convert molecules in `self.molecule_set` to `PreprocessingGraph`s
        molecular_graph_generator = map(self.get_graph, self.molecule_set)

        # loop over all the `PreprocessingGraph`s
        for molecular_graph in molecular_graph_generator:

            # get the number of decoding graphs (i.e. the decoding route length)
            # and add them to the running count
            n_subgraphs += molecular_graph.get_decoding_route_length()

        return int(n_subgraphs)

    def get_ts_properties(self, molecular_graphs : list, group_size : int) -> \
        None:
        """
        Gets molecular properties for group of molecular graphs, only for the
        training set.

        Args:
        ----
            molecular_graphs (list) : Contains `PreprocessingGraph`s.
            group_size (int)        : Size of "group" (i.e. slice of graphs).
        """
        if self.is_training_set:

            analyzer      = Analyzer(self.constants)
            ts_properties = analyzer.evaluate_training_set(
                preprocessing_graphs=molecular_graphs
            )

            # merge properties of current group with the previous group analyzed
            if self.ts_properties:  # `self.ts_properties` is a dictionary
                self.ts_properties = analyzer.combine_ts_properties(
                    prev_properties=self.ts_properties,
                    next_properties=ts_properties,
                    weight_next=group_size
                )
            else:  # `self.ts_properties` is None (has not been calculated yet)
                self.ts_properties = ts_properties
        else:
            self.ts_properties = None

    def load_datasets(self, hdf_file : h5py._hl.files.File) -> None:
        """
        Creates a dictionary of HDF datasets (`self.dataset`) which have been
        previously created (for restart jobs only).

        Args:
        ----
            hdf_file (h5py._hl.files.File) : HDF file containing all the datasets.
        """
        self.dataset = {}  # initialize dictionary of datasets

        # use the names of the datasets as the keys in `self.dataset`
        for ds_name in self.dataset_names:
            self.dataset[ds_name] = hdf_file.get(ds_name)

    def save_group(self, data_subgraphs : list, data_apds : list,
                   group_size : int, init_idx : int) -> None:
        """
        Saves a group of padded subgraphs and their corresponding APDs to the HDF
        datasets as `numpy.ndarray`s.

        Args:
        ----
            data_subgraphs (list) : Contains molecular subgraphs.
            data_apds (list)      : Contains APDs.
            group_size (int)      : Size of HDF "slice".
            init_idx (int)        : Index to begin slicing.
        """
        # convert to `np.ndarray`s
        nodes = np.array([graph_tuple[0] for graph_tuple in data_subgraphs])
        edges = np.array([graph_tuple[1] for graph_tuple in data_subgraphs])
        apds  = np.array(data_apds)

        end_idx = init_idx + group_size  # idx to end slicing

        # once data is padded, save it to dataset slice
        self.dataset["nodes"][init_idx:end_idx] = nodes
        self.dataset["edges"][init_idx:end_idx] = edges
        self.dataset["APDs"][init_idx:end_idx]  = apds

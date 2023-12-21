"""Data and graphs."""
import os

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom as molDG
from rdkit.Chem import rdmolops
from torch.utils.data import Dataset


class Graph:
    def __init__(self, molecule_smiles: str, node_vec_len: int, max_atoms: int = None):
        """
        Construct a molecular graph of a given molecule with a SMILES string.

        Build a node matrix that has dimensions (max_atoms,node_vec_len) and an
        adjacency matrix with dimensions (max_atoms,max_atoms).

        Parameters
        ----------
        molecule_smiles : str
            SMILES string of the molecule
        node_vec_len : int
            DESCRIPTION.
        max_atoms : int, optional
            DESCRIPTION. The default is None.
        """

        self.smiles = molecule_smiles
        self.node_vec_len = node_vec_len
        self.max_atoms = max_atoms
        self.smiles_to_mol()
        if self.mol is not None:
            self.smiles_to_graph()

    def smiles_to_mol(self):
        """
        Converts smiles string to Mol object in RDKit
        """

        # Use MolFromSmiles from RDKit to get molecule object
        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None:
            self.mol = None
            return

        # Add hydrogens to molecule
        self.mol = Chem.AddHs(mol)

    def smiles_to_graph(self):
        """
        Converts smiles to a graph.
        """

        # Get list of atoms in molecule
        atoms = self.mol.GetAtoms()

        # Create empty node matrix
        if self.max_atoms is None:
            n_atoms = len(list(atoms))
        else:
            n_atoms = self.max_atoms
        node_mat = np.zeros((n_atoms, self.node_vec_len))

        # Iterate over atoms and add to node matrix
        for atom in atoms:
            # Get atom index and atomic number
            atom_index = atom.GetIdx()
            atom_no = atom.GetAtomicNum()

            # Assign to node matrix
            node_mat[atom_index, atom_no] = 1

        # Create empty adjacency matrix
        adj_mat = np.zeros((n_atoms, n_atoms))

        # Create adjacency matrix
        adj_mat = rdmolops.GetAdjacencyMatrix(self.mol)
        self.std_adj_mat = np.copy(adj_mat)

        # Create distance matrix
        dist_mat = molDG.GetMoleculeBoundsMatrix(self.mol)
        dist_mat[dist_mat == 0.0] = 1

        # Get modified adjacency matrix
        adj_mat = adj_mat * (1 / dist_mat)

        # Pad the adjacency matrix
        dim_add = n_atoms - adj_mat.shape[0]
        adj_mat = np.pad(
            adj_mat, pad_width=((0, dim_add), (0, dim_add)), mode="constant"
        )

        # Add an identity matrix to adjacency matrix
        # This will make an atom its own neighbor
        adj_mat = adj_mat + np.eye(n_atoms)

        # Save both matrices
        self.node_mat = node_mat
        self.adj_mat = adj_mat


class GraphData(Dataset):
    def __init__(self, dataset_path: str, node_vec_len: int, max_atoms: int):
        """
        GraphData class inheriting from the Dataset class in PyTorch.

        Parameters
        ----------
        dataset_path: str
            Path to the dataset file
        node_vec_len : int
            Node vector length of molecular graphs
        max_atoms : int
            Maximum number of atoms in molecular graphs
        """

        # Save attributes
        self.node_vec_len = node_vec_len
        self.max_atoms = max_atoms

        # Open dataset file
        df = pd.read_csv(dataset_path)

        # Create lists
        self.indices = df.index.to_list()
        self.smiles = df["smiles"].to_list()
        self.outputs = df["measured log solubility in mols per litre"].to_list()

    def __len__(self):
        """
        Get length of the dataset

        Returns
        -------
        Length of dataset
        """
        return len(self.indices)

    def __getitem__(self, i: int):
        """
        Returns node matrix, adjacency matrix, output, and SMILES string of
        molecule.

        Parameters
        ----------
        i : int
            Dataset index

        Returns
        -------
        node_mat : torch.Tensor with dimension (max_atoms,node_vec_len)
            Node matrix
        adj_mat: torch.Tensor with dimension (max_atoms,max_atoms)
            Adjacency matrix
        output : torch.Tensor with dimension n_outputs
            Output vector
        smile : str
            SMILES string of molecule
        """

        # Get smile
        smile = self.smiles[i]

        # Create MolGraph object
        mol = Graph(smile, self.node_vec_len, self.max_atoms)

        # Get matrices
        node_mat = torch.Tensor(mol.node_mat)
        adj_mat = torch.Tensor(mol.adj_mat)

        # Get output
        output = torch.Tensor([self.outputs[i]])

        return (node_mat, adj_mat), output, smile

    def get_atom_no_sum(self, i):
        """
        Get sum of the atomic numbers of all molecules in the dataset

        Parameters
        ----------
        i : int
            Dataset index.

        Returns
        -------
        atomic_no_sum: int
            Sum of all atomic numbers
        """
        # Get smile
        smile = self.smiles[i]

        # Create MolGraph object
        mol = Graph(smile, self.node_vec_len, self.max_atoms)

        # Get matrices
        node_mat = mol.node_mat

        # Get atomic number sum
        one_pos_mat = np.argwhere(node_mat == 1)
        atomic_no_sum = one_pos_mat[:, -1].sum()
        return atomic_no_sum


def collate_graph_dataset(dataset: Dataset):
    """
    Collate function for the GraphData dataset.

    Parameters
    ----------
    dataset : GraphData
        Object of the GraphData class.

    Returns
    -------
    node_mats_tensor, adj_mats_tensor : tuple of two torch.Tensor objects
        Node matrices with dimensions (batch_size, max_atoms, node_vec_len) and
        adjacency matrices with dimensions (batch_size, max_atoms, max_atoms)
    outputs_tensor : torch.Tensor with dimensions (batch_size, n_outputs)
        Tensor containing outputs.
    smiles : list
        List of size batch_size containing SMILES strings.
    """

    # Create empty lists of node and adjacency matrices, outputs, and smiles
    node_mats = []
    adj_mats = []
    outputs = []
    smiles = []

    # Iterate over list and assign each component to the correct list
    for i in range(len(dataset)):
        (node_mat, adj_mat), output, smile = dataset[i]
        node_mats.append(node_mat)
        adj_mats.append(adj_mat)
        outputs.append(output)
        smiles.append(smile)

    # Create tensors
    node_mats_tensor = torch.cat(node_mats, dim=0)
    adj_mats_tensor = torch.cat(adj_mats, dim=0)
    outputs_tensor = torch.stack(outputs, dim=0)

    # Return tensors
    return (node_mats_tensor, adj_mats_tensor), outputs_tensor, smiles


if __name__ == "__main__":
    filepath = os.path.abspath(__file__)
    main_dirpath = os.path.dirname(os.path.dirname(filepath))
    dataset_path = os.path.join(main_dirpath + "/data/solubility_data.csv")
    data = GraphData(dataset_path, 10, 75)
    print(torch.diag(data[0][0][1].sum(dim=-1)))

"""Utility functions."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error


class Standardizer:
    def __init__(self, X):
        """
        Class to standardize ChemGCN outputs

        Parameters
        ----------
        X : torch.Tensor
            Tensor of outputs
        """
        self.mean = torch.mean(X)
        self.std = torch.std(X)

    def standardize(self, X):
        """
        Convert a non-standardized output to a standardized output

        Parameters
        ----------
        X : torch.Tensor
            Tensor of non-standardized outputs

        Returns
        -------
        Z : torch.Tensor
            Tensor of standardized outputs

        """
        Z = (X - self.mean) / (self.std)
        return Z

    def restore(self, Z):
        """
        Restore a standardized output to the non-standardized output

        Parameters
        ----------
        Z : torch.Tensor
            Tensor of standardized outputs

        Returns
        -------
        X : torch.Tensor
            Tensor of non-standardized outputs

        """
        X = self.mean + Z * self.std
        return X

    def state(self):
        """
        Return dictionary of the state of the Standardizer

        Returns
        -------
        dict
            Dictionary with the mean and std of the outputs

        """
        return {"mean": self.mean, "std": self.std}

    def load(self, state):
        """
        Load a dictionary containing the state of the Standardizer and assign mean and std

        Parameters
        ----------
        state : dict
            Dictionary containing mean and std
        """
        self.mean = state["mean"]
        self.std = state["std"]


# Utility functions to train, test model
def train_model(
    epoch,
    model,
    training_dataloader,
    optimizer,
    loss_fn,
    standardizer,
    use_GPU,
    max_atoms,
    node_vec_len,
):
    """
    Execute training of one epoch for the ChemGCN model.

    Parameters
    ----------
    epoch : int
        Current epoch
    model : ChemGCN
        ChemGCN model object
    training_dataloader : data.DataLoader
        Training DataLoader
    optimizer : torch.optim.Optimizer
        Model optimizer
    loss_fn : like nn.MSELoss()
        Model loss function
    standardizer : Standardizer
        Standardizer object
    use_GPU: bool
        Whether to use GPU
    max_atoms: int
        Maximum number of atoms in graph
    node_vec_len: int
        Maximum node vector length in graph

    Returns
    -------
    avg_loss : float
        Training loss averaged over batches
    avg_mae : float
        Training MAE averaged over batches
    """

    # Create variables to store losses and error
    avg_loss = 0
    avg_mae = 0
    count = 0

    # Switch model to train mode
    model.train()

    # Go over each batch in the dataloader
    for i, dataset in enumerate(training_dataloader):
        # Unpack data
        node_mat = dataset[0][0]
        adj_mat = dataset[0][1]
        output = dataset[1]

        # Reshape inputs
        first_dim = int((torch.numel(node_mat)) / (max_atoms * node_vec_len))
        node_mat = node_mat.reshape(first_dim, max_atoms, node_vec_len)
        adj_mat = adj_mat.reshape(first_dim, max_atoms, max_atoms)

        # Standardize output
        output_std = standardizer.standardize(output)

        # Package inputs and outputs; check if GPU is enabled
        if use_GPU:
            nn_input = (node_mat.cuda(), adj_mat.cuda())
            nn_output = output_std.cuda()
        else:
            nn_input = (node_mat, adj_mat)
            nn_output = output_std

        # Compute output from network
        nn_prediction = model(*nn_input)

        # Calculate loss
        loss = loss_fn(nn_output, nn_prediction)
        avg_loss += loss

        # Calculate MAE
        prediction = standardizer.restore(nn_prediction.detach().cpu())
        mae = mean_absolute_error(output, prediction)
        avg_mae += mae

        # Set zero gradients for all tensors
        optimizer.zero_grad()

        # Do backward prop
        loss.backward()

        # Update optimizer parameters
        optimizer.step()

        # Increase count
        count += 1

    # Calculate avg loss and MAE
    avg_loss = avg_loss.detach().cpu().numpy() / count
    avg_mae = avg_mae / count

    # Print stats
    print(
        "Epoch: [{0}]\tTraining Loss: [{1:.2f}]\tTraining MAE: [{2:.2f}]".format(
            epoch, avg_loss, avg_mae
        )
    )

    # Return loss and MAE
    return avg_loss, avg_mae


def test_model(
    model,
    test_dataloader,
    loss_fn,
    standardizer,
    use_GPU,
    max_atoms,
    node_vec_len,
):
    """
    Test the ChemGCN model.

    Parameters
    ----------
    model : ChemGCN
        ChemGCN model object
    test_dataloader : data.DataLoader
        Test DataLoader
    loss_fn : like nn.MSELoss()
        Model loss function
    standardizer : Standardizer
        Standardizer object
    use_GPU: bool
        Whether to use GPU
    max_atoms: int
        Maximum number of atoms in graph
    node_vec_len: int
        Maximum node vector length in graph

    Returns
    -------
    test_loss : float
        Test loss
    test_mae : float
        Test MAE
    """

    # Create variables to store losses and error
    test_loss = 0
    test_mae = 0
    count = 0

    # Switch model to train mode
    model.eval()

    # Go over each batch in the dataloader
    for i, dataset in enumerate(test_dataloader):
        # Unpack data
        node_mat = dataset[0][0]
        adj_mat = dataset[0][1]
        output = dataset[1]

        # Reshape inputs
        first_dim = int((torch.numel(node_mat)) / (max_atoms * node_vec_len))
        node_mat = node_mat.reshape(first_dim, max_atoms, node_vec_len)
        adj_mat = adj_mat.reshape(first_dim, max_atoms, max_atoms)

        # Standardize output
        output_std = standardizer.standardize(output)

        # Package inputs and outputs; check if GPU is enabled
        if use_GPU:
            nn_input = (node_mat.cuda(), adj_mat.cuda())
            nn_output = output_std.cuda()
        else:
            nn_input = (node_mat, adj_mat)
            nn_output = output_std

        # Compute output from network
        nn_prediction = model(*nn_input)

        # Calculate loss
        loss = loss_fn(nn_output, nn_prediction)
        test_loss += loss

        # Calculate MAE
        prediction = standardizer.restore(nn_prediction.detach().cpu())
        mae = mean_absolute_error(output, prediction)
        test_mae += mae

        # Increase count
        count += 1

    # Calculate avg loss and MAE
    test_loss = test_loss.detach().cpu().numpy() / count
    test_mae = test_mae / count

    # Return loss and MAE
    return test_loss, test_mae


def parity_plot(
    save_dir,
    model,
    test_dataloader,
    standardizer,
    use_GPU,
    max_atoms,
    node_vec_len,
):
    """
    Create a parity plot for the ChemGCN model.

    Parameters
    ----------
    save_dir: str
        Name of directory to store the parity plot in
    model : ChemGCN
        ChemGCN model object
    test_dataloader : data.DataLoader
        Test DataLoader
    standardizer : Standardizer
        Standardizer object
    use_GPU: bool
        Whether to use GPU
    max_atoms: int
        Maximum number of atoms in graph
    node_vec_len: int
        Maximum node vector length in graph

    """

    # Create variables to store losses and error
    outputs = []
    predictions = []

    # Switch model to train mode
    model.eval()

    # Go over each batch in the dataloader
    for i, dataset in enumerate(test_dataloader):
        # Unpack data
        node_mat = dataset[0][0]
        adj_mat = dataset[0][1]
        output = dataset[1]

        # Reshape inputs
        first_dim = int((torch.numel(node_mat)) / (max_atoms * node_vec_len))
        node_mat = node_mat.reshape(first_dim, max_atoms, node_vec_len)
        adj_mat = adj_mat.reshape(first_dim, max_atoms, max_atoms)

        # Package inputs and outputs; check if GPU is enabled
        if use_GPU:
            nn_input = (node_mat.cuda(), adj_mat.cuda())
        else:
            nn_input = (node_mat, adj_mat)

        # Compute output from network
        nn_prediction = model(*nn_input)

        # Calculate MAE
        prediction = standardizer.restore(nn_prediction.detach().cpu())

        # Add to list
        outputs.append(output)
        predictions.append(prediction)

    # Flatten
    outputs_arr = np.concatenate(outputs)
    preds_arr = np.concatenate(predictions)

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=500)
    ax.scatter(
        outputs_arr, preds_arr, marker="o", color="mediumseagreen", edgecolor="black"
    )

    min_plot = min(ax.get_xlim()[0], ax.get_ylim()[0])
    max_plot = max(ax.get_xlim()[1], ax.get_ylim()[1])
    min_plot = (1 - np.sign(min_plot) * 0.2) * min_plot
    max_plot = (1 + np.sign(max_plot) * 0.2) * max_plot

    ax.plot([min_plot, max_plot], [min_plot, max_plot], linestyle="-", color="black")
    ax.margins(x=0, y=0)
    ax.set_xlim([min_plot, max_plot])
    ax.set_ylim([min_plot, max_plot])
    ax.set_xlabel("Measured values (log mols/l)")
    ax.set_ylabel("ChemGCN predictions (log mols/l)")
    ax.set_title("Parity plot")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "parity_plot.png"))


def loss_curve(save_dir, epochs, losses):
    """
    Make a loss curve.

    Parameters
    ----------
    save_dir: str
        Name of directory to store plot in
    epochs: list
        List of epochs
    losses: list
        List of losses

    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=500)
    ax.plot(epochs, losses, marker="o", linestyle="--", color="royalblue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean squared loss")
    ax.set_title("Loss curve")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "loss_curve.png"))

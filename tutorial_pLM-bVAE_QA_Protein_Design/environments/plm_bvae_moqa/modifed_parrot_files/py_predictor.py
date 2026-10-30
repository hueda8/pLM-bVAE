'''
Python module for integrating a trained network directly into a Python workflow.

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.

'''

from parrot import brnn_architecture
from parrot import encode_sequence

import torch
import numpy as np

def softmax(v):
    return (np.e ** v) / np.sum(np.e ** v)

class Predictor():
    """Class that for integrating a trained PARROT network into a Python workflow

    Usage:
    >>> from parrot import py_predictor
    >>> my_predictor = py_predictor.Predictor(
    ...     </path/to/saved_network.pt>,
    ...     dtype={"sequence" or "residues"},
    ...     encoding_file=None,  # optional
    ... )
    >>> value = my_predictor.predict(AA_sequence)

    ***
    Notes
    -----
    - Input size is inferred automatically from saved network weights.
    - If `encoding_file` is provided, that encoding is used preferentially.
    - If `encoding_file` is not provided:
        * input_size == 20: one-hot encoding is used.
        * input_size != 20: raises ValueError (encoding_file required).
    - Sequences are assumed to contain canonical amino acids.
    ***


    Attributes
    ----------
    dtype : str
            Data format that the network was trained for. Either "sequence" or
            "residues".
    num_layers : int
            Number of hidden layers in the trained network.
    hidden_vector_size : int
            Size of hidden vectoer in the trained network.
    n_classes : int
            Number of data classes that the network was trained for. If 1, then
            network is designed for regression task. If >1, then classification
            task with n_classes.
    task : str
            Designates if network is designed for "classification" or "regression".
    network : PyTorch object
            Initialized PARROT network with loaded weights.
    """

    def __init__(self, saved_weights, dtype, encoding_file=None):
        """
        Parameters
        ----------
        saved_weights : str or Path
            Path to saved network weights (.pt). Network architecture
            parameters are inferred from these weights.
        dtype : str
            Data format that the network was trained for.
            Must be either "sequence" or "residues".
        encoding_file : str or Path, optional
            Amino-acid descriptor table file.
            If provided, this encoding is used for prediction regardless of
            input size (dimension must match model input_size).
            If omitted, one-hot is used only when input_size == 20.
        """

        self.dtype = dtype
        self.encoding_file = encoding_file
        loaded_model = torch.load(saved_weights, map_location=torch.device('cpu'))
        self.input_size = int(np.shape(loaded_model['lstm.weight_ih_l0'])[1])

        # Dynamically infer network size from loaded weights
        self.num_layers = 0
        while True:
            s = f'lstm.weight_ih_l{self.num_layers}'
            try:
                _ = loaded_model[s]
                self.num_layers += 1
            except KeyError:
                break

        # Extract network hyperparameters from loaded weights
        self.hidden_vector_size = int(np.shape(loaded_model['lstm.weight_ih_l0'])[0] / 4)
        self.n_classes = np.shape(loaded_model['fc.bias'])[0]

        if self.n_classes > 1:
            self.task = "classification"
        else:
            self.task = "regression"

        # Instantiate network and load trained weights
        if self.dtype == "sequence":
            self.network = brnn_architecture.BRNN_MtO(self.input_size, self.hidden_vector_size,
                                            self.num_layers, self.n_classes, 'cpu')
        elif self.dtype == "residues":
            self.network = brnn_architecture.BRNN_MtM(self.input_size, self.hidden_vector_size,
                                            self.num_layers, self.n_classes, 'cpu')
        else:
            raise ValueError("dtype must equal 'residues' or 'sequence'")

        self.network.load_state_dict(loaded_model)

    def _load_user_encoding(self, file_path):
        table = {}
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                aa = parts[0].upper()
                vec = [float(x) for x in parts[1:]]
                table[aa] = vec
        if len(table) == 0:
            raise ValueError(f"Encoding file is empty: {file_path}")
        return table

    def _encode_with_table(self, seq, table):
        """Encode sequence using the provided amino-acid descriptor table."""
        arr = []
        for aa in seq:
            if aa not in table:
                raise ValueError(f"Invalid amino acid for user encoding: {aa}")
            arr.append(table[aa])
        return torch.tensor(arr, dtype=torch.float32)

    def predict(self, seq):
        """Use the network to predict values for a single sequence of valid amino acids

        Parameters
        ----------
        seq : str
            Valid amino acid sequence

        Encoding behavior:
        ----------
        - If encoding_file is provided: use descriptor-table encoding.
        - Else if model input_size == 20: use one-hot encoding.
        - Else: raise ValueError (descriptor table required).

        Returns
        -------
        np.ndarray
            Returns a 1D np.ndarray the length of the sequence where each position
            is the prediction at that position.
        """

        # convert sequence to uppercase
        seq = seq.upper()

        # 1) encoding_file
        if self.encoding_file is not None:
            table = self._load_user_encoding(self.encoding_file)
            try:
                first_dim = len(next(iter(table.values())))
            except StopIteration:
                raise ValueError(f"Encoding file is empty: {self.encoding_file}")

            if first_dim != self.input_size:
                raise ValueError(f"encoding_file dimension is {first_dim}, but model expects {self.input_size}")

            # Convert sequence using user-provided encoding table
            seq_vector = self._encode_with_table(seq, table)

        # 2) no encoding_file
        else:
            if self.input_size == 20:
                seq_vector = encode_sequence.one_hot(seq)
            else:
                raise ValueError(f"This model expects {self.input_size}-dim input. "
                                 "Please pass encoding_file=..."
                                )
        # formatting
        seq_vector = seq_vector.view(1, len(seq_vector), -1)

        # Forward pass
        prediction = self.network(seq_vector.float()).detach().numpy().flatten()

        # softmax to get class probabilities
        if self.task == "classification":
            if self.dtype == "residues":
                prediction = prediction.reshape(-1, self.n_classes)
                prediction = np.array(list(map(softmax, prediction)))

            else:
                prediction = softmax(prediction)

        return prediction
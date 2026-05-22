import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
try:
    from torchtrain.variable_data_loader import VariableDataLoader
except ImportError:  # pragma: no cover - fallback for older torchtrain layouts
    from variable_data_loader import VariableDataLoader
from torchtrain import Module

class DeepLog(Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        """DeepLog model used for training and predicting logs.

            Parameters
            ----------
            input_size : int
                Dimension of input layer.

            hidden_size : int
                Dimension of hidden layer.

            output_size : int
                Dimension of output layer.

            num_layers : int, default=2
                Number of hidden layers, i.e. stacked LSTM modules.
            """
        # Initialise nn.Module
        super(DeepLog, self).__init__()

        # Store input parameters
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers  = num_layers

        # Initialise model layers
        self.lstm    = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.out     = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    ########################################################################
    #                       Forward through network                        #
    ########################################################################

    def forward(self, X):
        """Forward sample through DeepLog.

            Parameters
            ----------
            X : tensor
                Input to forward through DeepLog network.

            Returns
            -------
            result : tensor

            """
        X = F.one_hot(self._to_device(X).to(torch.int64), self.input_size).to(
            dtype=torch.float32,
        )

        # Set initial hidden states
        hidden = self._get_initial_state(X)
        state  = self._get_initial_state(X)

        # Perform LSTM layer
        out, hidden = self.lstm(X, (hidden, state))
        # Perform output layer
        out = self.out(out[:, -1, :])
        # Create probability
        out = self.softmax(out)

        # Return result
        return out


    ########################################################################
    #                            Predict method                            #
    ########################################################################

    def predict(self, X, y=None, k=1, batch_size=32, variable=False, verbose=True):
        """Predict the k most likely output values

            Parameters
            ----------
            X : torch.Tensor of shape=(n_samples, seq_len)
                Input of sequences, these will be one-hot encoded to an array of
                shape=(n_samples, seq_len, input_size)

            y : Ignored
                Ignored

            k : int, default=1
                Number of output items to generate

            variable : boolean, default=False
                If True, predict inputs of different sequence lengths

            verbose : boolean, default=True
                If True, print output

            Returns
            -------
            result : torch.Tensor of shape=(n_samples, k)
                k most likely outputs

            confidence : torch.Tensor of shape=(n_samples, k)
                Confidence levels for each output
            """
        device = self._device()
        result = list()
        confidence = list()

        with torch.no_grad():
            if variable:
                data = VariableDataLoader(
                    X,
                    torch.zeros(len(X)),
                    index=False,
                    batch_size=batch_size,
                    shuffle=False,
                )
                iterator = tqdm.tqdm(data, desc="Predicting", disable=not verbose)
                for X_, _ in iterator:
                    prediction = self(X_).exp()
                    batch_confidence, batch_result = prediction.topk(k)
                    result.append(batch_result.cpu())
                    confidence.append(batch_confidence.cpu())
            else:
                iterator = tqdm.tqdm(
                    range(0, X.shape[0], batch_size),
                    desc="Predicting",
                    disable=not verbose,
                )
                for batch in iterator:
                    prediction = self(X[batch:batch + batch_size]).exp()
                    batch_confidence, batch_result = prediction.topk(k)
                    result.append(batch_result.cpu())
                    confidence.append(batch_confidence.cpu())

            if device.type == "mps" and hasattr(torch, "mps"):
                torch.mps.empty_cache()

        return torch.cat(result), torch.cat(confidence)

    def fit(
        self,
        X,
        y,
        epochs=10,
        batch_size=32,
        learning_rate=0.01,
        criterion=nn.NLLLoss(),
        optimizer=optim.SGD,
        variable=False,
        verbose=True,
        **kwargs,
    ):
        """Train DeepLog with batches moved to the model device."""
        optimizer = optimizer(params=self.parameters(), lr=learning_rate)
        device = self._device()

        for epoch in range(1, epochs + 1):
            try:
                if variable:
                    iterator = tqdm.tqdm(
                        VariableDataLoader(X, y, batch_size=batch_size, shuffle=True),
                        desc="[Epoch {:{width}}/{:{width}}]".format(
                            epoch,
                            epochs,
                            width=len(str(epochs)),
                        ),
                        disable=not verbose,
                    )
                    for X_, y_ in iterator:
                        optimizer.zero_grad()
                        y_ = y_.to(device)
                        y_pred = self(X_)
                        loss = criterion(y_pred, y_)
                        loss.backward()
                        optimizer.step()
                else:
                    indices = torch.randperm(X.shape[0])
                    iterator = tqdm.tqdm(
                        range(0, X.shape[0], batch_size),
                        desc="[Epoch {:{width}}/{:{width}}]".format(
                            epoch,
                            epochs,
                            width=len(str(epochs)),
                        ),
                        disable=not verbose,
                    )
                    for batch in iterator:
                        optimizer.zero_grad()
                        batch_indices = indices[batch:batch + batch_size]
                        X_ = X[batch_indices]
                        y_ = y[batch_indices].to(device)
                        y_pred = self(X_)
                        loss = criterion(y_pred, y_)
                        loss.backward()
                        optimizer.step()
            except KeyboardInterrupt:
                print("\nTraining interrupted, performing clean stop")
                break

        return self

    ########################################################################
    #                             I/O methods                              #
    ########################################################################

    def save(self, outfile):
        """Save model to output file.

            Parameters
            ----------
            outfile : string
                File to output model.
            """
        # Save to output file
        torch.save(self.state_dict(), outfile)

    @classmethod
    def load(cls, infile, device=None):
        """Load model from input file.

            Parameters
            ----------
            infile : string
                File from which to load model.
            """
        # Load state dictionary
        state_dict = torch.load(infile, map_location=device)

        print(state_dict.keys())

        # Get input variables from state_dict
        input_size  = state_dict.get('lstm.weight_ih_l0').shape[1]
        hidden_size = state_dict.get('lstm.weight_hh_l0').shape[1]
        output_size = input_size
        num_layers  = (len(state_dict) - 2) // 4

        # Create ContextBuilder
        result = cls(
            input_size  = input_size,
            hidden_size = hidden_size,
            output_size = output_size,
            num_layers  = num_layers,
        )

        # Cast to device if necessary
        if device is not None: result = result.to(device)

        # Set trained parameters
        result.load_state_dict(state_dict)

        # Return result
        return result

    ########################################################################
    #                         Auxiliary functions                          #
    ########################################################################

    def _get_initial_state(self, X):
        """Return a given hidden state for X."""
        # Return tensor of correct shape as device
        return torch.zeros(
            self.num_layers,
            X.size(0),
            self.hidden_size,
            device=self._device(),
            dtype=torch.float32,
        )

    def _to_device(self, X):
        """Move a tensor to the model device."""
        return X.to(self._device())

    def _device(self):
        """Return the current model device."""
        return next(self.parameters()).device

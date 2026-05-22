# Import pytorch
import torch

# Import classification report
from sklearn.metrics import classification_report

# Import DeepLog and Preprocessor
from deeplog              import DeepLog
from deeplog.preprocessor import Preprocessor

##############################################################################
#                                 Load data                                  #
##############################################################################

# Create preprocessor for loading data
preprocessor = Preprocessor(
    length  = 10,           # Extract sequences of 20 items
    timeout = float('inf'), # Do not include a maximum allowed time between events
)

# Load the three HDFS splits.
#
# The preprocessor remaps each file independently unless a mapping is supplied.
# Reusing the train mapping for the test splits keeps shared event IDs aligned
# with the training vocabulary.
X_train, y_train, label_train, mapping_train = preprocessor.text(
    "./data/hdfs_train",
    verbose=True,
)
X_test, y_test, label_test, mapping_test = preprocessor.text(
    "./data/hdfs_test_normal",
    verbose=True,
    mapping=mapping_train,
)
X_test_anomaly, y_test_anomaly, label_test_anomaly, mapping_test_anomaly = (
    preprocessor.text(
        "./data/hdfs_test_abnormal",
        verbose=True,
        mapping=mapping_train,
    )
)

input_size = max(
    len(mapping_train),
    len(mapping_test),
    len(mapping_test_anomaly),
)

no_event_id = next(
    index for index, event_id in mapping_train.items() if event_id == preprocessor.NO_EVENT
)


def _drop_warm_up(context: torch.Tensor, events: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop left-padded warm-up rows so scoring starts after real history."""
    mask = torch.all(context != no_event_id, dim=1)
    return context[mask], events[mask]


X_train, y_train = _drop_warm_up(X_train, y_train)
X_test, y_test = _drop_warm_up(X_test, y_test)
X_test_anomaly, y_test_anomaly = _drop_warm_up(X_test_anomaly, y_test_anomaly)

##############################################################################
#                                  DeepLog                                   #
##############################################################################

# Create DeepLog object
deeplog = DeepLog(
    input_size  = input_size,
    hidden_size = 64 , # Hidden dimension, we suggest 64
    output_size = input_size,
)

# Optionally cast data and DeepLog to cuda, if available
if torch.cuda.is_available():
    # Set deeplog to device
    deeplog = deeplog.to("cuda")

    # Set data to device
    X_train        = X_train       .to("cuda")
    y_train        = y_train       .to("cuda")
    X_test         = X_test        .to("cuda")
    y_test         = y_test        .to("cuda")
    X_test_anomaly = X_test_anomaly.to("cuda")
    y_test_anomaly = y_test_anomaly.to("cuda")

# Train deeplog
deeplog.fit(
    X          = X_train,
    y          = y_train,
    epochs     = 10,
    batch_size = 128,
    optimizer  = torch.optim.Adam,
)

# Predict normal data using deeplog
y_pred_normal, confidence = deeplog.predict(
    X = X_test,
    k = 9, # Change this value to get the top k predictions (called 'g' in DeepLog paper, see Figure 6)
)

# Predict anomalous data using deeplog
y_pred_anomaly, confidence = deeplog.predict(
    X = X_test_anomaly,
    k = 9, # Change this value to get the top k predictions (called 'g' in DeepLog paper, see Figure 6)
)

################################################################################
#                            Classification report                             #
################################################################################

print("Classification report - predictions")
print(classification_report(
    y_true = y_test.cpu().numpy(),
    y_pred = y_pred_normal[:, 0].cpu().numpy(),
    digits = 4,
    zero_division = 0,
))

################################################################################
#                             Check for anomalies                              #
################################################################################

# Check if the actual value matches any of the predictions
# If any prediction matches, it is not an anomaly, so to get the anomalies, we
# invert our answer using ~

# Check for anomalies in normal data (ideally, we should not find any)
anomalies_normal = ~torch.any(
    y_test == y_pred_normal.T,
    dim = 0,
)

# Check for anomalies in abnormal data (ideally, we should not find all)
anomalies_abnormal = ~torch.any(
    y_test_anomaly == y_pred_anomaly.T,
    dim = 0,
)

# Compute classification report for anomalies
y_pred = torch.cat((anomalies_normal, anomalies_abnormal))
y_true = torch.cat((
    torch.zeros(anomalies_normal  .shape[0], dtype=bool),
    torch.ones (anomalies_abnormal.shape[0], dtype=bool),
))

print("Classification report - anomalies")
print(classification_report(
    y_pred       = y_pred.cpu().numpy(),
    y_true       = y_true.cpu().numpy(),
    labels       = [False, True],
    target_names = ["Normal", "Anomaly"],
    digits       = 4,
))

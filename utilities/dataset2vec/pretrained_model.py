import os


def get_weights_path(split_n: int):
    """Get the specified split's pre-trained model's checkpoint. Must be in [0, 1, 2, 3, 4]"""

    if split_n not in [0, 1, 2, 3, 4]:
        raise ValueError(f"Split number {split_n} should be either 0, 1, 2, 3 or 4")

    rootdir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(rootdir, "checkpoints", f"split-{split_n}", "weights")


def get_configuration():
    """Get the pre-trained model's configuration, for instance to re-construct the dataset2vec model"""
    configuration = {
        "nonlinearity_d2v": "relu",
        "units_f": 32,
        "nhidden_f": 4,
        "architecture_f": "SQU",
        "resblocks_f": 8,
        "units_h": 32,
        "nhidden_h": 4,
        "architecture_h": "SQU",
        "resblocks_h": 8,
        "units_g": 32,
        "nhidden_g": 4,
        "architecture_g": "SQU",
        "ninstanc": 256,
        "nclasses": 3,
        "nfeature": 16,
        "number": 0,
        "split": 0,
        "LookAhead": False,
        "searchspace": "a",
        "learning_rate": 0.001,
        "delta": 2,
        "gamma": 1,
        "minmax": True,
        "batch_size": 16,
        "cardinality": 256,
        "D": 10,
        "trainable": 75072,
    }

    return configuration

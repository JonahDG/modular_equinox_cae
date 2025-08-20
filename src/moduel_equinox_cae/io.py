from pathlib import path

import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt

def load_data(npz_path):
    """loads data in the same shape and style as mnist.npz

    Args:
        npz_path (str): path string for data file

    Raises:
        FileNotFoundError: wrong path or file in incorrect spot
        ValueError: missing key in data
        ValueError: training wrong shape
        ValueError: testing wrong shape

    Returns:
        jnp.array: x and y training (labels are not returned because this is for unsupervised learning)
    """

    path=Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")
    
    data=jnp.load(path)
    required_keys=["x_train","y_train","x_test","y_test"]
    missing_keys=[]
    for k in required_keys:
        if k not in data:
            missing_keys.append(k)
    if len(missing_keys)!=0:
        raise ValueError(f"Missing keys {missing_keys} in {path}")
    
    x_train=data["x_train"]
    y_train=data["y_train"]

    if len(x_train.shape) != 3 or x_train.shape[1:] != (28, 28):
        raise ValueError(f"x_train must be (N,28,28), got {x_train.shape}")
    if len(x_test.shape) != 3 or x_test.shape[1:] != (28, 28):
        raise ValueError(f"x_test must be (N,28,28), got {x_test.shape}")
    
    return x_train, y_train

def preprocess(array):
    """Normalizes and reshape data to (N,1,28,28)

    Args:
        array (ndarray): unlabeled data

    Returns:
        ndarray: normalized array
    """

    array=jnp.astype("float32")/255.0
    array=jnp.resize(jnp.array(array),(len(array),1,28,28))
    return array

def add_noise(array,key,noise_factor=0.4):
    # soon TM
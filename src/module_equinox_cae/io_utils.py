import os
from typing import Tuple, Any
import json

# import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

# ============================================================================ #
# Save/load complete .eqx models
# ============================================================================ #
def save_model(model, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        eqx.tree_serialise_leaves(f, model)

def load_model(model_like, path: str):
    with open(path, "rb") as f:
        return eqx.tree_deserialise_leaves(model_like, f)

# ============================================================================ #
# Save/Load embeddings & Preditions
# ============================================================================ #
def save_embeddings(embeds: jnp.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    jnp.save(path, embeds)

def load_embeddings(path: str) -> jnp.ndarray:
    return jnp.load(path)

def save_predictions(preds: jnp.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    jnp.save(path, preds)

def load_predictions(path: str) -> jnp.ndarray:
    return jnp.load(path)

def run_sigmoid(preds: jnp.ndarray):
    return jax.nn.sigmoid(preds)

# ============================================================================ #
# Input data handling
# ============================================================================ #
def load_train_val(path: str) -> Tuple[jnp.ndarray, jnp.ndarray]:
    ext=os.path.splitext(path)[1].lower()
    if ext!=".npz":
        raise ValueError(f"Unsupported file extension: {ext}. Only .npz files are supported.")
    data=jnp.load(path)
    keys=list(data.keys())
    if "x_train" in keys and "x_test" in keys:
        x_train=data["x_train"]
        x_test=data["x_test"]
    else:
        if len(keys)<2:
            raise ValueError(f"Not enough arrays in .npz file. Found keys: {keys}")
        else:
            raise ValueError(f"Could not find 'x_train' and 'x_test' in .npz file. Found keys: {keys}")
    
    return x_train, x_test

def preprocess(array: jnp.ndarray) -> jnp.ndarray:
    array=array.astype("float32")/255.0
    array=jnp.resize(jnp.array(array),(len(array),1,28,28,))
    return array

def noise(array,noise_factor,key):
    normal_key,uniform_key=jax.random.split(key,2)
    if noise_factor==None:
        noise_factor=jax.random.uniform(uniform_key,shape=(),minval=0.0,maxval=1.0)
    noisy_array=array+noise_factor*jax.random.normal(normal_key,array.shape)
    return jnp.clip(noisy_array,0.,1.)

# ============================================================================ #
# Save training history
# ============================================================================ #
def save_train_losses(train_losses,val_losses,csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    jnp.savez(csv_path,train_losses=train_losses,val_losses=val_losses)

def load_train_losses(path: str) -> None:
    return jnp.load(path)

def save_checkpoint(path: str, model, opt_state,epoch: int, step: int, key: jax.random.PRNGKey) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path+".tmp"
    with open(tmp, "wb") as f:
        eqx.tree_serialise_leaves(f, (model, opt_state, epoch, step, key))
    os.replace(tmp, path)

def load_checkpoint(path: str, model_like, opt_state_like, key_like: jax.random.PRNGKey) -> Tuple[Any, Any, int, int, jax.random.PRNGKey]:
    with open(path, "rb") as f:
        model, opt_state, epoch, step, key = eqx.tree_deserialise_leaves((model_like, opt_state_like, 0, 0, key_like), f)
    return model, opt_state, epoch, step, key

# ============================================================================ #
# key handling
# ============================================================================ #
def key_handler(seed: int):
    print()
    primary_key=jax.random.PRNGKey(seed)
    model_key, primary_noise_key, primary_display_key=jax.random.split(primary_key,3)
    return primary_key, model_key, primary_noise_key, primary_display_key



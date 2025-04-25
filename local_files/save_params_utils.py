#<save_params_utils.py>
import jax.numpy as jnp
import numpy as np
import os


def flatten_dict(d, parent_key='', sep='//'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def save_model_params(d, file_path):
    flat_dict = flatten_dict(d)
    # Convert JAX arrays to NumPy for saving
    np_dict = {k: np.array(v) if isinstance(v, jnp.ndarray) else v for k, v in flat_dict.items()}
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.savez(file_path, **np_dict)



def unflatten_dict(d, sep='//'):
    result_dict = {}
    for flat_key, value in d.items():
        keys = flat_key.split(sep)
        d = result_dict
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = value
    return result_dict

def load_model_params(file_path):
    with np.load(file_path, allow_pickle=True) as npz_file:
        # Convert NumPy arrays back to JAX arrays
        jax_dict = {k: jnp.array(v) for k, v in npz_file.items()}
    return unflatten_dict(jax_dict)

#</save_params_utils.py>
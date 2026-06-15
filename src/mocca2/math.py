"""These routines can contract 2D chromatogram to create 1D data suitable for peak picking"""

from typing import Literal
from numpy.typing import NDArray, ArrayLike

import numpy as np
from scipy.stats import entropy
        
def cosine_similarity(x: NDArray, y: NDArray, axis: int = -1) -> float | NDArray:
    """Calculates cosine similarity over specified axis"""

    dot_prod = np.sum(x * y, axis=axis)
    norm_x = np.linalg.norm(x, axis=axis)
    norm_y = np.linalg.norm(y, axis=axis)

    norm_x = np.clip(norm_x, 1e-5, None)
    norm_y = np.clip(norm_y, 1e-5, None)

    return dot_prod/norm_x/norm_y

def entropy_similarity(x: NDArray, y: NDArray, noise_threshold: 0.01) -> float:
    """
    Entropy similarity for MS comparison

    Parameters
    ----------
    x : NDArray
        _description_
    y : NDArray
        _description_

    Returns
    -------
    float
        _description_
    """
    
    def _remove_noise(array: NDArray, threshold: float) -> NDArray:
        tmp_spectrum = array.copy()
        max_intensity = np.max(array)
        tmp_spectrum[tmp_spectrum < max_intensity * threshold] = 0.0
        return tmp_spectrum
    # Remove intensity values smaller than threshold of max intensity
    x_clean = _remove_noise(x, noise_threshold)
    y_clean = _remove_noise(y, noise_threshold)
    # Normalize
    x_norm = x_clean / np.sum(x_clean)
    y_norm = y_clean / np.sum(y_clean)
    merged = (x_norm + y_norm) / 2
    
    similarity = 1 - (2 * entropy(merged) - entropy(x_norm) - entropy(y_norm)) / np.log(4)
    
    return similarity
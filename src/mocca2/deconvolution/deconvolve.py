from typing import Tuple, Literal
from numpy.typing import NDArray

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from mocca2.deconvolution.nonnegative_lstsq import concentrations_from_spectra, spectra_from_concentrations
from mocca2.deconvolution.fit_peak_model import fit_peak_model
from mocca2.deconvolution.peak_models import PeakModel, BiGaussian, BiGaussianTailing, FraserSuzuki, Bemg, BiLaplacian


def deconvolve_adaptive(
        data: NDArray,
        model: PeakModel | Literal['BiGaussian', 'BiGaussianTailing', 'FraserSuzuki', 'Bemg', 'BiLaplacian'],
        max_mse: float,
        relaxe_concs: bool,
        min_comps: int,
        max_comps: int,
) -> Tuple[NDArray, NDArray, float]:
    """
    Deconvolves data with increasingly more components until MSE limit is reached

    Parameters
    ----------
    data: NDArray
        2D data [wavelength, data]

    model: PeakModel | Literal['BiGaussian', 'BiGaussianTailing', 'FraserSuzuki']
        mathematical model used for fitting shapes of components of peaks

    max_mse: float
        Maximum allowed MSE for termination

    relaxe_concs: bool
        If False, the fitted peak model functions are returned

        Otherwise, the concentrations are refined with restricted least squares

    min_comps: int
        Minimum number of components that can be fitted

    max_comps: int
        Maximum number of components that can be fitted

    Returns
    -------
    NDArray
        concentrations [compound, time]

    NDArray
        spectra [compound, wavelength], normalized such that mean = 1

    float
        MSE

    """

    if isinstance(model, str):
        model = {
            'BiGaussian': BiGaussian(),
            'BiGaussianTailing': BiGaussianTailing(),
            'FraserSuzuki': FraserSuzuki(),
            'Bemg': Bemg(),
            'BiLaplacian': BiLaplacian(),
        }[model]

    prev_mse = np.inf
    min_rel_improvement = 0.2
    for n_comps in range(min_comps, max_comps+1):
        # Deconvolve peak with some increasing number of components
        concs, spectra, mse = deconvolve_fixed(
            data, n_comps, model, relaxe_concs
        )
        # Check whether improvement is sufficiently large to justify extra component
        if prev_mse != np.inf:
            print(f"DEBUG: improvement with {n_comps} components", (prev_mse - mse) / prev_mse)            
            # If not, break and keep previous result
            if np.abs(prev_mse - mse) / prev_mse < min_rel_improvement: 
                concs, spectra, mse = prev_concs, prev_spectra, prev_mse
                break

        print("Number of spectra", len(spectra), spectra.shape, mse)
        # Check spectra of new components
        if len(spectra) > 1:
            sim_matrix = cosine_similarity(spectra)
            upper_tri = sim_matrix[np.triu_indices(sim_matrix.shape[0], k=1)]
            print(sim_matrix)
            if np.all(upper_tri > 0.99):
                concs, spectra, mse = prev_concs, prev_spectra, prev_mse
                break
        
        # Check whether MSE is sufficiently small with current number
        if mse < max_mse:
            break
        
        # Store current result for next iteration comparison
        prev_concs = concs
        prev_spectra = spectra
        prev_mse = mse

    return concs, spectra, mse


def deconvolve_fixed(data: NDArray, n_comps: int, model: PeakModel, relaxe_concs: bool) -> Tuple[NDArray, NDArray, float]:
    """
    Deconvolves data with given number of components. Returns concentration, spectra and MSE

    Parameters
    ----------
    data: NDArray
        2D data [wavelength, data]

    n_comps: int
        how many components should be used for deconvolution

    model: PeakModel
        mathematical model used for fitting shapes of components of peaks

    relaxe_concs: bool
        If False, the fitted bigaussian functions are returned

        Otherwise, the concentrations are refined with restricted least squares

    Returns
    -------
    NDArray
        concentrations [compound, time]

    NDArray
        spectra [compound, wavelength], normalized such that mean = 1

    float
        MSE

    """

    # Fit peak model
    concs, mse, _ = fit_peak_model(data, model, n_compounds=n_comps)

    # Get spectra
    spectra, mse = spectra_from_concentrations(data, concs)

    if relaxe_concs:
        # Relaxe the constrain on peaks being bigaussian
        concs, mse = concentrations_from_spectra(data, spectra)
        spectra, mse = spectra_from_concentrations(data, concs)

    # Normalize the spectra and scale concentrations accordingly
    norm_factors = np.mean(spectra, axis=1) + 1e-7
    spectra = (spectra.T / norm_factors).T
    concs = (concs.T * norm_factors).T

    return concs, spectra, mse

import typing as tp
from pathlib import Path
import numpy as np
from scipy.linalg import toeplitz
from autograd import numpy as npa

def trapezoid(a: np.ndarray, b: np.ndarray) -> float:
    try:
        return np.trapz(a, b)
    except:
        return np.trapz(a, b)

def bragg(X: np.ndarray) -> float:
    # ... (rest of the function remains the same)

def chirped(X: np.ndarray) -> float:
    # ... (rest of the function remains the same)

def marche(a: float, b: float, p: float, n: int, x: float) -> np.ndarray:
    # ... (rest of the function remains the same)

def creneau(k0: float, a0: float, pol: float, e1: float, e2: float, a: np.ndarray, n: int, x0: np.ndarray) -> tuple:
    # ... (rest of the function remains the same)

def homogene(k0: float, a0: float, pol: float, epsilon: float, n: int) -> tuple:
    # ... (rest of the function remains the same)

def interface(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    # ... (rest of the function remains the same)

def morpho(X: np.ndarray) -> float:
    # ... (rest of the function remains the same)

def cascade(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # ... (rest of the function remains the same)

def cascade2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # ... (rest of the function remains the same)

def absorption(lam: float, epsilon: np.ndarray, mu: np.ndarray, type_: np.ndarray, hauteur: np.ndarray, pol: int, theta: float) -> np.ndarray:
    # ... (rest of the function remains the same)

def cf_photosic_reference(X: np.ndarray) -> float:
    # ... (rest of the function remains the same)

def cf_photosic_realistic(eps_and_d: np.ndarray) -> float:
    # ... (rest of the function remains the same)

def ceviche(x: np.ndarray, benchmark_type: int = 0, discretize: bool = False, wantgrad: bool = False, wantfields: bool = False) -> tuple:
    """
    x = 2d or 3d array of scalars
    Inputs:
    1. benchmark_type = 0 1 2 or 3, depending on which benchmark you want
    2. discretize = True if we want to force x to be in {0,1} (just checks sign(x-0.5) )
    3. wantgrad = True if we want to know the gradient
    4. wantfields = True if we want the fields of the simulation
    Returns:
    1. the loss (to be minimized)
    2. the gradient or none (depending on wantgrad)
    3. the fields or none (depending on wantfields
    """
    global first_time_ceviche
    global model
    import autograd
    import autograd.numpy as npa
    import ceviche_challenges
    import autograd
    # ... (rest of the function remains the same)

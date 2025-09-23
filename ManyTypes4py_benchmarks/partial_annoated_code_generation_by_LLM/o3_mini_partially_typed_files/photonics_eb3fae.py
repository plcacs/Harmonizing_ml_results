#!/usr/bin/env python3
import typing as tp
from pathlib import Path
import numpy as np
from scipy.linalg import toeplitz

def trapezoid(a: np.ndarray, b: np.ndarray) -> float:
    try:
        return float(np.trapz(a, b))
    except Exception:
        return float(np.trapezoid(a, b))

def bragg(X: np.ndarray) -> float:
    """
    Cost function for the Bragg mirror problem: maximizing the reflection
    when the refractive index are given for all the layers.
    Input: a vector whose components represent each the thickness of each
    layer.
    https://hal.archives-ouvertes.fr/hal-02613161
    """
    lam: float = 600
    bar: int = int(np.size(X) / 2)
    n: np.ndarray = np.concatenate(([1], np.sqrt(X[0:bar]), [1.7320508075688772]))
    type_: np.ndarray = np.arange(0, bar + 2)
    hauteur: np.ndarray = np.concatenate(([0], X[bar:2 * bar], [0]))
    tmp: np.ndarray = np.tan(2 * np.pi * n[type_] * hauteur / lam)
    Z: complex = n[-1]
    for k in range(np.size(type_) - 1, 0, -1):
        Z = (Z - 1j * n[type_[k]] * tmp[k]) / (1 - 1j * tmp[k] * Z / n[type_[k]])
    r: complex = (1 - Z) / (1 + Z)
    c: float = np.real(1 - r * np.conj(r))
    return float(c)

def chirped(X: np.ndarray) -> float:
    lam: np.ndarray = np.linspace(500, 800, 50)
    n: np.ndarray = np.array([1, 1.4142135623730951, 1.7320508075688772])
    type_: np.ndarray = np.concatenate(([0], np.tile([2, 1], int(np.size(X) / 2)), [2]))
    hauteur: np.ndarray = np.concatenate(([0], X, [0]))
    r: np.ndarray = np.zeros(np.size(lam), dtype=complex)
    for m in range(0, np.size(lam)):
        tmp: np.ndarray = np.tan(2 * np.pi * n[type_] * hauteur / lam[m])
        Z: complex = 1.7320508075688772
        for k in range(np.size(type_) - 1, 0, -1):
            Z = (Z - 1j * n[type_[k]] * tmp[k]) / (1 - 1j * tmp[k] * Z / n[type_[k]])
        r[m] = (1 - Z) / (1 + Z)
    c: float = 1 - np.real(np.sum(r * np.conj(r)) / np.size(lam))
    return float(c)

def cascade(T: np.ndarray, U: np.ndarray) -> np.ndarray:
    n: int = int(T.shape[1] / 2)
    J: np.ndarray = np.linalg.inv(np.eye(n) - np.matmul(U[0:n, 0:n], T[n:2 * n, n:2 * n]))
    K: np.ndarray = np.linalg.inv(np.eye(n) - np.matmul(T[n:2 * n, n:2 * n], U[0:n, 0:n]))
    S: np.ndarray = np.block([
        [T[0:n, 0:n] + np.matmul(np.matmul(np.matmul(T[0:n, n:2 * n], J), U[0:n, 0:n]), T[n:2 * n, 0:n]),
         np.matmul(np.matmul(T[0:n, n:2 * n], J), U[0:n, n:2 * n])],
        [np.matmul(np.matmul(U[n:2 * n, 0:n], K), T[n:2 * n, 0:n]),
         U[n:2 * n, n:2 * n] + np.matmul(np.matmul(np.matmul(U[n:2 * n, 0:n], K), T[n:2 * n, n:2 * n]), U[0:n, n:2 * n])]
    ])
    return S

def c_bas(A: np.ndarray, V: np.ndarray, h: float) -> np.ndarray:
    n: int = int(A.shape[1] / 2)
    D: np.ndarray = np.diag(np.exp(1j * V * h))
    S: np.ndarray = np.block([
        [A[0:n, 0:n], np.matmul(A[0:n, n:2 * n], D)],
        [np.matmul(D, A[n:2 * n, 0:n]), np.matmul(np.matmul(D, A[n:2 * n, n:2 * n]), D)]
    ])
    return S

def marche(a: float, b: float, p: float, n: int, x: float) -> np.ndarray:
    l: np.ndarray = np.zeros(n, dtype=np.complex128)
    m: np.ndarray = np.zeros(n, dtype=np.complex128)
    tmp: np.ndarray = 1 / (2 * np.pi * np.arange(1, n)) * (np.exp(-2 * 1j * np.pi * p * np.arange(1, n)) - 1) * np.exp(-2 * 1j * np.pi * np.arange(1, n) * x)
    l[1:n] = 1j * (a - b) * tmp
    l[0] = p * a + (1 - p) * b
    m[0] = l[0]
    m[1:n] = 1j * (b - a) * np.conj(tmp)
    T: np.ndarray = toeplitz(l, m)
    return T

def creneau(k0: float, a0: float, pol: int, e1: float, e2: float, a: float, n: int, x0: float) -> tp.Tuple[np.ndarray, np.ndarray]:
    nmod: int = int(n / 2)
    alpha: np.ndarray = np.diag(a0 + 2 * np.pi * np.arange(-nmod, nmod + 1))
    if pol == 0:
        M: np.ndarray = alpha * alpha - k0 * k0 * marche(e1, e2, a, n, x0)
        (L, E) = np.linalg.eig(M)
        L = np.sqrt(-L + 0j)
        L = (1 - 2 * (np.imag(L) < -1e-15)) * L
        P: np.ndarray = np.block([E, np.matmul(E, np.diag(L))])
    else:
        U: np.ndarray = marche(1 / e1, 1 / e2, a, n, x0)
        T: np.ndarray = np.linalg.inv(U)
        M: np.ndarray = np.matmul(np.matmul(np.matmul(T, alpha), np.linalg.inv(marche(e1, e2, a, n, x0))), alpha) - k0 * k0 * T
        (L, E) = np.linalg.eig(M)
        L = np.sqrt(-L + 0j)
        L = (1 - 2 * (np.imag(L) < -1e-15)) * L
        P = np.block([E, np.matmul(np.matmul(U, E), np.diag(L))])
    return (P, L)

def homogene(k0: float, a0: float, pol: float, epsilon: float, n: int) -> tp.Tuple[np.ndarray, np.ndarray]:
    nmod: int = int(n / 2)
    valp: np.ndarray = np.sqrt(epsilon * k0 * k0 - (a0 + 2 * np.pi * np.arange(-nmod, nmod + 1)) ** 2 + 0j)
    valp = valp * (1 - 2 * (valp < 0))
    P: np.ndarray = np.block([np.eye(n), np.diag(valp * (pol / epsilon + (1 - pol)))])
    return (P, valp)

def interface(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    n: int = int(P.shape[1])
    S: np.ndarray = np.matmul(
        np.linalg.inv(np.block([[P[0:n, 0:n], -Q[0:n, 0:n]], [P[n:2 * n, 0:n], Q[n:2 * n, 0:n]]])),
        np.block([[-P[0:n, 0:n], Q[0:n, 0:n]], [P[n:2 * n, 0:n], Q[n:2 * n, 0:n]]])
    )
    return S

def morpho(X: np.ndarray) -> float:
    lam: float = 449.5897
    pol: float = 1.0
    d: float = 600.521475
    nmod: int = 25
    e2: float = 2.4336
    n: int = 2 * nmod + 1
    n_motifs: int = int(X.size / 4)
    X = X / d
    h: np.ndarray = X[0:n_motifs]
    x0: np.ndarray = X[n_motifs:2 * n_motifs]
    a: np.ndarray = X[2 * n_motifs:3 * n_motifs]
    spacers: np.ndarray = X[3 * n_motifs:4 * n_motifs]
    l: float = lam / d
    k0: float = 2 * np.pi / l
    (P, V) = homogene(k0, 0, pol, 1, n)
    S: np.ndarray = np.block([[np.zeros([n, n], dtype=np.complex128), np.eye(n, dtype=np.complex128)],
                              [np.eye(n, dtype=np.complex128), np.zeros([n, n], dtype=np.complex128)]])
    for j in range(0, n_motifs):
        (Pc, Vc) = creneau(k0, 0, int(pol), e2, 1, a[j], n, x0[j])
        S = cascade(S, interface(P, Pc))
        S = c_bas(S, Vc, h[j])
        S = cascade(S, interface(Pc, P))
        S = c_bas(S, V, spacers[j])
    (Pc, Vc) = homogene(k0, 0, pol, e2, n)
    S = cascade(S, interface(P, Pc))
    R: np.ndarray = np.zeros(3, dtype=np.float64)
    for j in range(-1, 2):
        R[j] = abs(S[j + nmod, nmod]) ** 2 * np.real(V[j + nmod]) / k0
    cost: float = 1 - (R[-1] + R[1]) / 2 + R[0] / 2
    lams: np.ndarray = (np.array([400, 500, 600, 700, 800]) + 0.24587) / d
    bar: float = 0
    for lo in lams:
        k0 = 2 * np.pi / lo
        (P, V) = homogene(k0, 0, pol, 1, n)
        S = np.block([[np.zeros([n, n], dtype=np.complex128), np.eye(n)],
                      [np.eye(n), np.zeros([n, n])]])
        for j in range(0, n_motifs):
            (Pc, Vc) = creneau(k0, 0, int(pol), e2, 1, a[j], n, x0[j])
            S = cascade(S, interface(P, Pc))
            S = c_bas(S, Vc, h[j])
            S = cascade(S, interface(Pc, P))
            S = c_bas(S, V, spacers[j])
        (Pc, Vc) = homogene(k0, 0, pol, e2, n)
        S = cascade(S, interface(P, Pc))
        bar += abs(S[nmod, nmod]) ** 2 * np.real(V[nmod]) / k0
    cost += bar / lams.size
    return cost

i: complex = complex(0, 1)

def epscSi(lam: np.ndarray) -> np.ndarray:
    a: np.ndarray = np.arange(250, 1500, 5)
    e: np.ndarray = np.load(Path(__file__).with_name('epsilon_epscSi.npy'))
    y: int = np.argmin(np.sign(lam - a)) - 1
    epsilon: np.ndarray = (e[y + 1] - e[y]) / (a[y + 1] - a[y]) * (lam - a[y]) + e[y]
    return epsilon

def cascade2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    This function takes two 2x2 matrices A and B, that are assumed to be scattering matrices
    and combines them assuming A is the "upper" one, and B the "lower" one, physically.
    The result is a 2x2 scattering matrix.
    """
    t: complex = 1 / (1 - B[0, 0] * A[1, 1])
    S: np.ndarray = np.zeros((2, 2), dtype=complex)
    S[0, 0] = A[0, 0] + A[0, 1] * B[0, 0] * A[1, 0] * t
    S[0, 1] = A[0, 1] * B[0, 1] * t
    S[1, 0] = B[1, 0] * A[1, 0] * t
    S[1, 1] = B[1, 1] + A[1, 1] * B[0, 1] * B[1, 0] * t
    return S

def solar(lam: np.ndarray) -> np.ndarray:
    a: np.ndarray = np.load(Path(__file__).with_name('wavelength_solar.npy'))
    e: np.ndarray = np.load(Path(__file__).with_name('epsilon_solar.npy'))
    jsc: np.ndarray = np.interp(lam, a, e)
    return jsc

def absorption(lam: float, epsilon: np.ndarray, mu: np.ndarray, type_: np.ndarray, hauteur: np.ndarray, pol: int, theta: float) -> np.ndarray:
    f: np.ndarray = mu if not pol else epsilon
    k0: float = 2 * np.pi / lam
    g: int = type_.size
    alpha: float = np.sqrt(epsilon[type_[0]] * mu[type_[0]]) * k0 * np.sin(theta)
    gamma: np.ndarray = np.sqrt(epsilon[type_] * mu[type_] * k0 ** 2 - np.ones(g) * alpha ** 2)
    if np.real(epsilon[type_[0]]) < 0 and np.real(mu[type_[0]]) < 0:
        gamma[0] = -gamma[0]
    if g > 2:
        gamma[1:g - 2] = gamma[1:g - 2] * (1 - 2 * (np.imag(gamma[1:g - 2]) < 0))
    if np.real(epsilon[type_[g - 1]]) < 0 and np.real(mu[type_[g - 1]]) < 0 and (np.real(np.sqrt(epsilon[type_[g - 1]] * mu[type_[g - 1]] * k0 ** 2 - alpha ** 2)) != 0):
        gamma[g - 1] = -np.sqrt(epsilon[type_[g - 1]] * mu[type_[g - 1]] * k0 ** 2 - alpha ** 2)
    else:
        gamma[g - 1] = np.sqrt(epsilon[type_[g - 1]] * mu[type_[g - 1]] * k0 ** 2 - alpha ** 2)
    T: np.ndarray = np.zeros((2 * g, 2, 2), dtype=complex)
    T[0] = [[0, 1], [1, 0]]
    for k2 in range(g - 1):
        t: complex = np.exp(i * gamma[k2] * hauteur[k2])
        T[2 * k2 + 1] = [[0, t], [t, 0]]
        b1: float = gamma[k2] / f[type_[k2]]
        b2: float = gamma[k2 + 1] / f[type_[k2 + 1]]
        T[2 * k2 + 2] = [[(b1 - b2) / (b1 + b2), 2 * b2 / (b1 + b2)], [2 * b1 / (b1 + b2), (b2 - b1) / (b1 + b2)]]
        t = np.exp(i * gamma[g - 1] * hauteur[g - 1])
        T[2 * g - 1] = [[0, t], [t, 0]]
    H: np.ndarray = np.zeros((2 * g - 1, 2, 2), dtype=complex)
    A: np.ndarray = np.zeros((2 * g - 1, 2, 2), dtype=complex)
    H[0] = T[2 * g - 1]
    A[0] = T[0]
    for j in range(2 * g - 2):
        A[j + 1] = cascade2(A[j], T[j + 1])
        H[j + 1] = cascade2(T[2 * g - 2 - j], H[j])
    t = A[len(A) - 1][1, 0]
    I: np.ndarray = np.zeros((2 * g, 2, 2), dtype=complex)
    for j in range(len(T) - 1):
        I[j][0, 0] = A[j][1, 0] / (1 - A[j][1, 1] * H[len(T) - 2 - j][0, 0])
        I[j][0, 1] = A[j][1, 1] * H[len(T) - 2 - j][0, 1] / (1 - A[j][1, 1] * H[len(T) - 2 - j][0, 0])
        I[j][1, 0] = A[j][1, 0] * H[len(T) - 2 - j][0, 0] / (1 - A[j][1, 1] * H[len(T) - 2 - j][0, 0])
        I[j][1, 1] = H[len(T) - 2 - j][0, 1] / (1 - A[j][1, 1] * H[len(T) - 2 - j][0, 0])
    I[2 * g - 1][0, 0] = I[2 * g - 2][0, 0] * np.exp(i * gamma[g - 1] * hauteur[g - 1])
    I[2 * g - 1][0, 1] = I[2 * g - 2][0, 1] * np.exp(i * gamma[g - 1] * hauteur[g - 1])
    I[2 * g - 1][1, 0] = 0
    I[2 * g - 1][1, 1] = 0
    w: int = 0
    poynting: np.ndarray = np.zeros(2 * g, dtype=complex)
    if pol == 0:
        for j in range(2 * g):
            poynting[j] = np.real((I[j][0, 0] + I[j][1, 0]) * np.conj((I[j][0, 0] - I[j][1, 0]) * gamma[w] / mu[type_[w]]) * mu[type_[0]] / gamma[0])
            w = w + 1 - np.mod(j + 1, 2)
    else:
        for j in range(2 * g):
            poynting[j] = np.real((I[j][0, 0] - I[j][1, 0]) * np.conj((I[j][0, 0] + I[j][1, 0]) * gamma[w] / epsilon[type_[w]]) * epsilon[type_[0]] / gamma[0])
            w = w + 1 - np.mod(j + 1, 2)
    tmp: np.ndarray = abs(-np.diff(poynting))
    absorb: np.ndarray = tmp[np.arange(0, 2 * g, 2)]
    return absorb

def cf_photosic_reference(X: np.ndarray) -> float:
    """vector X is only the thicknesses of each layers, because the materials (so the epsilon)
    are imposed by the function. This is similar in the chirped function.
    """
    lam_min: float = 375
    lam_max: float = 750
    n_lam: int = 100
    theta: float = 0 * np.pi / 180
    vlam: np.ndarray = np.linspace(lam_min, lam_max, n_lam)
    scc: np.ndarray = np.zeros(n_lam)
    Ab: np.ndarray = np.zeros(n_lam)
    for k in range(n_lam):
        lam: float = vlam[k]
        epsilon: np.ndarray = np.array([1, 2, 3, epscSi(np.array([lam]))[0]], dtype=complex)
        mu: np.ndarray = np.ones(epsilon.size)
        type_: np.ndarray = np.append(0, np.append(np.tile(np.array([1, 2]), int(X.size / 2)), 3))
        hauteur: np.ndarray = np.append(0, np.append(X, 30000))
        pol: int = 0
        absorb: np.ndarray = absorption(lam, epsilon, mu, type_, hauteur, pol, theta)
        scc[k] = solar(np.array([lam]))[0]
        Ab[k] = absorb[len(absorb) - 1]
    max_scc: float = trapezoid(scc, vlam)
    j_sc: float = trapezoid(scc * Ab, vlam)
    CE: float = j_sc / max_scc
    cost: float = 1 - CE
    return cost

def cf_photosic_realistic(eps_and_d: np.ndarray) -> float:
    """eps_and_d is a vector composed in a first part with the epsilon values
    (the material used in each one of the layers), and in a second part with the
    thicknesses of each one of the layers, like in Bragg.
    Any number of layers can work. Basically I used between 4 and 50 layers,
    and the best results are generally obtained when the structure has between 10 and 20 layers.
    The epsilon values are generally comprised between 1.00 and 9.00.
    """
    dimension: int = int(eps_and_d.size / 2)
    eps: np.ndarray = eps_and_d[0:dimension]
    d: np.ndarray = eps_and_d[dimension:dimension * 2]
    epsd: np.ndarray = np.array([eps, d])
    lam_min: float = 375
    lam_max: float = 750
    n_lam: int = 100
    theta: float = 0 * 180 / np.pi
    vlam: np.ndarray = np.linspace(lam_min, lam_max, n_lam)
    scc: np.ndarray = np.zeros(n_lam)
    Ab: np.ndarray = np.zeros(n_lam)
    for k in range(n_lam):
        lam: float = vlam[k]
        epsilon: np.ndarray = np.append(1, np.append(epsd[0], epscSi(np.array([lam]))[0]))
        mu: np.ndarray = np.ones(epsilon.size)
        type_: np.ndarray = np.arange(0, epsd[0].size + 2)
        hauteur: np.ndarray = np.append(0, np.append(epsd[1], 30000))
        pol: int = 0
        absorb: np.ndarray = absorption(lam, epsilon, mu, type_, hauteur, pol, theta)
        scc[k] = solar(np.array([lam]))[0]
        Ab[k] = absorb[len(absorb) - 1]
    max_scc: float = trapezoid(scc, vlam)
    j_sc: float = trapezoid(scc * Ab, vlam)
    CE: float = j_sc / max_scc
    cost: float = 1 - CE
    return cost

first_time_ceviche: bool = True
model: tp.Any = None
global no_neg
no_neg: bool = True

def ceviche(x: np.ndarray, benchmark_type: int = 0, discretize: bool = False, wantgrad: bool = False, wantfields: bool = False) -> tp.Any:
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
    3. the fields or none (depending on wantfields)
    """
    global first_time_ceviche
    global model
    import autograd
    import autograd.numpy as npa
    import ceviche_challenges
    if first_time_ceviche or x is None:
        if benchmark_type == 0:
            spec = ceviche_challenges.waveguide_bend.prefabs.waveguide_bend_2umx2um_spec()
            params = ceviche_challenges.waveguide_bend.prefabs.waveguide_bend_sim_params()
            model = ceviche_challenges.waveguide_bend.model.WaveguideBendModel(params, spec)
        elif benchmark_type == 1:
            spec = ceviche_challenges.beam_splitter.prefabs.pico_splitter_spec()
            params = ceviche_challenges.beam_splitter.prefabs.pico_splitter_sim_params()
            model = ceviche_challenges.beam_splitter.model.BeamSplitterModel(params, spec)
        elif benchmark_type == 2:
            spec = ceviche_challenges.mode_converter.prefabs.mode_converter_spec_23()
            params = ceviche_challenges.mode_converter.prefabs.mode_converter_sim_params()
            model = ceviche_challenges.mode_converter.model.ModeConverterModel(params, spec)
        elif benchmark_type == 3:
            spec = ceviche_challenges.wdm.prefabs.wdm_spec()
            params = ceviche_challenges.wdm.prefabs.wdm_sim_params()
            model = ceviche_challenges.wdm.model.WdmModel(params, spec)
    if discretize:
        x = np.round(x)
    if isinstance(x, str) and x == 'name':
        return {0: 'waveguide-bend', 1: 'beam-splitter', 2: 'mode-converter', 3: 'wdm'}[benchmark_type]
    elif x is None:
        return model.design_variable_shape
    assert x.shape == model.design_variable_shape, f'Expected shape: {model.design_variable_shape}'
    design: np.ndarray = x > 0.5 if discretize else x

    def loss_fn(x: npa.ndarray, fields_are_needed: bool = False) -> tp.Union[float, tp.Tuple[float, npa.ndarray]]:
        """A simple loss function taking mean s11 - mean s21."""
        (s_params, fields) = model.simulate(x)
        if benchmark_type == 0 or benchmark_type == 2:
            the_loss: float = 1 - npa.abs(s_params[0, 0, 1]) ** 2
        elif benchmark_type == 1:
            the_loss = npa.abs(npa.abs(s_params[0, 0, 1]) ** 2 - 0.5) + npa.abs(npa.abs(s_params[0, 0, 2]) ** 2 - 0.5)
        elif benchmark_type == 3:
            the_loss = 1 - (npa.abs(s_params[0, 0, 1]) ** 2 + npa.abs(s_params[1, 0, 2]) ** 2) / 2
        else:
            raise ValueError("Invalid benchmark type")
        global no_neg
        if the_loss < 0 and no_neg:
            no_neg = False
            print(f'NEG npa: pb{benchmark_type}, {the_loss} vs {loss_fn_nograd(x)}')
        if fields_are_needed:
            return (the_loss, fields)
        return the_loss

    def loss_fn_nograd(x: np.ndarray, fields_are_needed: bool = False) -> tp.Union[float, tp.Tuple[float, np.ndarray]]:
        """A simple loss function taking mean s11 - mean s21."""
        (s_params, fields) = model.simulate(x)
        if benchmark_type == 0 or benchmark_type == 2:
            the_loss: float = 1 - np.abs(s_params[0, 0, 1]) ** 2
        elif benchmark_type == 1:
            the_loss = np.abs(np.abs(s_params[0, 0, 1]) ** 2 - 0.5) + np.abs(np.abs(s_params[0, 0, 2]) ** 2 - 0.5)
        elif benchmark_type == 3:
            the_loss = 1 - (np.abs(s_params[0, 0, 1]) ** 2 + np.abs(s_params[1, 0, 2]) ** 2) / 2
        else:
            raise ValueError("Invalid benchmark type")
        global no_neg
        if the_loss < 0 and no_neg:
            no_neg = False
            print(f'NEG np: pb{benchmark_type}, np:{the_loss}, npa:{loss_fn(x)}')
        if fields_are_needed:
            return (the_loss, fields)
        return the_loss

    if wantgrad:
        (loss_value_npa, grad) = autograd.value_and_grad(loss_fn)(design)
    else:
        loss_value = loss_fn_nograd(design)
    first_time_ceviche = False
    assert not (wantgrad and wantfields)
    if wantgrad:
        return (loss_value_npa, grad)
    if wantfields:
        (loss_value, fields) = loss_fn_nograd(design, fields_are_needed=True)
        return (loss_value, fields)
    return loss_value

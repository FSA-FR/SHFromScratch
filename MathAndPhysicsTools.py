"""
MathAndPhysicsTools.py

FR: Module central pour les outils mathématiques et physiques.
    Ce module contient TOUTES les fonctions outils utilisées dans le package,
    à l'exception des fonctions de propagation qui doivent rester dans Beam.py.
    
    Fonctionnalités principales :
    - Création de grilles de coordonnées (1D, 2D, polaires)
    - Génération de bases de modes (Zernike, Legendre, Hermite-Gauss, Laguerre-Gauss)
    - Normalisations (PV, RMS, Noll, Wyant)
    - Conversions d'unités (nm ↔ λ, rad ↔ mrad, mm ↔ µm, etc.)
    - Calculs de statistiques (PV, RMS, moyenne, écart-type)
    - Gestion des NaN (sans propagation)
    - Opérations matricielles sûres (division, sqrt, log, etc.)
    - Interpolation et lissage
    - Génération de nombres aléatoires avec graines
    
    Règle d'or :
    - TOUTE fonction outil utilisée dans plusieurs fichiers DOIT être ici.
    - Les fonctions de propagation DOIVENT rester dans Beam.py.
    - Les fonctions de visualisation DOIVENT être dans Visualization.py.
    
    Unités par défaut :
    - Longueurs : mm (sauf indication contraire)
    - Longueur d'onde : nm
    - Phase : nm (principale), rad (pour les calculs)
    - Angles : rad (radians)

EN: Central module for mathematical and physical tools.
    This module contains ALL utility functions used in the package,
    except propagation functions which must remain in Beam.py.
    
    Main features:
    - Coordinate grid creation (1D, 2D, polar)
    - Mode basis generation (Zernike, Legendre, Hermite-Gauss, Laguerre-Gauss)
    - Normalizations (PV, RMS, Noll, Wyant)
    - Unit conversions (nm ↔ λ, rad ↔ mrad, mm ↔ µm, etc.)
    - Statistics calculations (PV, RMS, mean, std)
    - NaN handling (no propagation)
    - Safe array operations (division, sqrt, log, etc.)
    - Interpolation and smoothing
    - Random number generation with seeds
    
    Golden rule:
    - EVERY utility function used in multiple files MUST be here.
    - Propagation functions MUST remain in Beam.py.
    - Visualization functions MUST be in Visualization.py.
    
    Default units:
    - Lengths: mm (unless specified otherwise)
    - Wavelength: nm
    - Phase: nm (main), rad (for calculations)
    - Angles: rad (radians)

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch

Dependencies:
    - numpy
    - scipy (for special functions: Bessel, Hermite, Laguerre, etc.)

Sources:
    - "Numerical Recipes in C" by Press et al. (1992)
      -> Algorithmes numériques de base (FFT, interpolation, gestion des NaN)
      -> Chapitres 1 (gestion des erreurs), 3 (interpolation), 5 (FFT)
    - "Principles of Optics" by M. Born & E. Wolf (Cambridge, 1999)
      -> Fondements de l'optique (Ch. 1-3)
    - "Laser Beam Propagation" by J. W. Goodman (1996)
      -> Propagation des faisceaux (Ch. 3-4)
    - "Zernike polynomials and atmospheric turbulence" by J. W. Noll (1976)
      -> Polynômes de Zernike pour l'optique adaptative
    - "Handbook of Mathematical Functions" by Abramowitz & Stegun (1964)
      -> Fonctions spéciales (Bessel, Hermite, Laguerre)
"""

import numpy as np
import logging
from typing import Optional, Tuple, Dict, List, Union
from enum import Enum
import warnings


# =============================================================================
# CONFIGURATION GLOBALE
# =============================================================================

# Désactiver les warnings de division par zéro et NaN
warnings.filterwarnings("ignore", category=np.RankWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MathAndPhysicsTools")


# =============================================================================
# CONSTANTES PHYSIQUES
# =============================================================================

SPEED_OF_LIGHT_M_PER_S = 299792458.0  # m/s
PLANCK_CONSTANT_J_S = 6.62607015e-34  # J·s
PI = np.pi
TWO_PI = 2 * np.pi

DEFAULT_WAVELENGTH_NM = 633.0  # Longueur d'onde par défaut (He-Ne laser, nm)
NUMERICAL_TOLERANCE = 1e-12

# Facteurs de conversion
NM_TO_M = 1e-9
MM_TO_M = 1e-3
UM_TO_M = 1e-6
M_TO_NM = 1e9
M_TO_MM = 1e3
M_TO_UM = 1e6

RAD_TO_MRAD = 1000.0
MRAD_TO_RAD = 1.0 / RAD_TO_MRAD
RAD_TO_DEG = 180.0 / np.pi
DEG_TO_RAD = np.pi / 180.0


# =============================================================================
# ENUMS
# =============================================================================

class ZernikeOrdering(Enum):
    NOLL = "noll"
    WYANT = "wyant"
    STANDARD = "standard"


class NormalizationType(Enum):
    PV = "pv"
    RMS = "rms"
    NOLL = "noll"
    WYANT = "wyant"


# =============================================================================
# FONCTIONS DE CRÉATION DE GRILLES
# =============================================================================

def create_grid(num_points: int,
               diameter_mm: float = 1.0,
               center: Tuple[float, float] = (0.0, 0.0)) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    FR: Crée une grille de coordonnées 2D.
    EN: Creates a 2D coordinate grid.
    
    Sources: "Numerical Recipes in C" by Press et al. (1992), Ch. 1
    """
    radius_mm = diameter_mm / 2
    x = np.linspace(center[0] - radius_mm, center[0] + radius_mm, num_points)
    y = np.linspace(center[1] - radius_mm, center[1] + radius_mm, num_points)
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y


def create_polar_grid(num_points_r: int,
                      num_points_theta: int,
                      max_radius_mm: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    FR: Crée une grille de coordonnées polaires.
    EN: Creates a polar coordinate grid.
    
    Sources: "Numerical Recipes in C" by Press et al. (1992), Ch. 1
    """
    r = np.linspace(0, max_radius_mm, num_points_r)
    theta = np.linspace(0, TWO_PI, num_points_theta, endpoint=False)
    R, Theta = np.meshgrid(r, theta)
    return r, theta, R, Theta


def cartesian_to_polar(x: Union[float, np.ndarray],
                       y: Union[float, np.ndarray]) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """FR: Convertit des coordonnées cartésiennes en coordonnées polaires."""
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta


def polar_to_cartesian(r: Union[float, np.ndarray],
                       theta: Union[float, np.ndarray]) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """FR: Convertit des coordonnées polaires en coordonnées cartésiennes."""
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


# =============================================================================
# FONCTIONS DE GESTION DES NaN (TOUTES GÈRENT LES NaN SANS LES PROPAGER)
# =============================================================================

def handle_nan(array: np.ndarray,
               method: str = 'zero') -> np.ndarray:
    """
    FR: Gère les valeurs NaN dans un tableau.
    EN: Handles NaN values in an array.
    
    DOES NOT propagate NaN.
    
    Sources: "Numerical Recipes in C" by Press et al. (1992), Ch. 1
    """
    if not np.any(np.isnan(array)):
        return array
    if method == 'zero':
        return np.nan_to_num(array, nan=0.0)
    elif method == 'mean':
        return np.nan_to_num(array, nan=np.nanmean(array))
    elif method == 'median':
        return np.nan_to_num(array, nan=np.nanmedian(array))
    elif method == 'ignore':
        return array
    else:
        logger.warning(f"Unknown NaN handling method: {method}. Using 'zero'.")
        return np.nan_to_num(array, nan=0.0)


def safe_divide(numerator: Union[float, np.ndarray],
                denominator: Union[float, np.ndarray],
                default: float = 0.0) -> Union[float, np.ndarray]:
    """
    FR: Division sûre (évite les divisions par zéro et les NaN).
    EN: Safe division (avoids division by zero and NaN).
    
    DOES NOT propagate NaN.
    
    Sources: "Numerical Recipes in C" by Press et al. (1992), Ch. 1
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(numerator, denominator)
    return np.where(np.isfinite(result), result, default)


def safe_sqrt(x: Union[float, np.ndarray],
              default: float = 0.0) -> Union[float, np.ndarray]:
    """
    FR: Racine carrée sûre.
    EN: Safe square root.
    
    DOES NOT propagate NaN.
    
    Sources: "Numerical Recipes in C" by Press et al. (1992), Ch. 1
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.sqrt(np.maximum(np.real(x), 0.0))
    return np.where(np.isfinite(result), result, default)


def safe_log(x: Union[float, np.ndarray],
              default: float = 0.0) -> Union[float, np.ndarray]:
    """
    FR: Logarithme sûr.
    EN: Safe logarithm.
    
    DOES NOT propagate NaN.
    
    Sources: "Numerical Recipes in C" by Press et al. (1992), Ch. 1
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.log(np.maximum(np.real(x), NUMERICAL_TOLERANCE))
    return np.where(np.isfinite(result), result, default)


def safe_exp(x: Union[float, np.ndarray],
              default: float = 0.0,
              clip_max: float = 700.0) -> Union[float, np.ndarray]:
    """
    FR: Exponentielle sûre.
    EN: Safe exponential.
    
    DOES NOT propagate NaN.
    
    Sources: "Numerical Recipes in C" by Press et al. (1992), Ch. 1
    """
    x_clipped = np.clip(x, -clip_max, clip_max)
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.exp(x_clipped)
    return np.where(np.isfinite(result), result, default)


# =============================================================================
# FONCTIONS DE STATISTIQUES
# =============================================================================

def compute_pv_rms(data: np.ndarray,
                   handle_nan: bool = True) -> Tuple[float, float]:
    """
    FR: Calcule le PV (Peak-to-Valley) et le RMS.
    EN: Computes PV (Peak-to-Valley) and RMS.
    
    DOES NOT propagate NaN.
    
    Sources: "Data Reduction and Error Analysis" by P. R. Bevington (1969)
    """
    if data is None:
        raise ValueError("data cannot be None")
    if handle_nan:
        data = handle_nan(data, method='zero')
    return float(np.max(data) - np.min(data)), float(np.std(data))


def compute_statistics(data: np.ndarray,
                        handle_nan: bool = True) -> Dict[str, float]:
    """
    FR: Calcule les statistiques complètes d'un tableau.
    EN: Computes complete statistics of an array.
    
    DOES NOT propagate NaN.
    """
    if data is None:
        raise ValueError("data cannot be None")
    if handle_nan:
        data = handle_nan(data, method='zero')
    return {
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'pv': float(np.max(data) - np.min(data)),
        'rms': float(np.std(data))
    }


# =============================================================================
# FONCTIONS DE NORMALISATION
# =============================================================================

def normalize_array(array: np.ndarray,
                     method: str = 'max',
                     handle_nan: bool = True) -> np.ndarray:
    """
    FR: Normalise un tableau.
    EN: Normalizes an array.
    
    DOES NOT propagate NaN.
    
    Sources: "Numerical Recipes in C" by Press et al. (1992), Ch. 1
    """
    if array is None:
        raise ValueError("array cannot be None")
    if handle_nan:
        array = handle_nan(array, method='zero')
    
    if method == 'max':
        max_val = np.max(array)
        return array / max_val if max_val > 0 else array
    elif method == 'sum':
        total = np.sum(array)
        return array / total if total > 0 else array
    elif method == 'rms':
        rms = np.std(array)
        return array / rms if rms > 0 else array
    elif method == 'mean':
        return array - np.mean(array)
    elif method == 'zero_min':
        return array - np.min(array)
    else:
        raise ValueError(f"Méthode de normalisation inconnue: {method}")


# =============================================================================
# FONCTIONS DE CONVERSION D'UNITÉS
# =============================================================================

def nm_to_rad(phase_nm: Union[float, np.ndarray],
              wavelength_nm: float = DEFAULT_WAVELENGTH_NM) -> Union[float, np.ndarray]:
    """FR: Convertit une phase de nanomètres en radians."""
    return phase_nm * (TWO_PI / wavelength_nm)


def rad_to_nm(phase_rad: Union[float, np.ndarray],
              wavelength_nm: float = DEFAULT_WAVELENGTH_NM) -> Union[float, np.ndarray]:
    """FR: Convertit une phase de radians en nanomètres."""
    return phase_rad * (wavelength_nm / TWO_PI)


def nm_to_lambda(phase_nm: Union[float, np.ndarray],
                 wavelength_nm: float = DEFAULT_WAVELENGTH_NM) -> Union[float, np.ndarray]:
    """FR: Convertit une phase de nanomètres en longueurs d'onde."""
    return phase_nm / wavelength_nm


def lambda_to_nm(phase_lambda: Union[float, np.ndarray],
                  wavelength_nm: float = DEFAULT_WAVELENGTH_NM) -> Union[float, np.ndarray]:
    """FR: Convertit une phase de longueurs d'onde en nanomètres."""
    return phase_lambda * wavelength_nm


def rad_to_mrad(phase_rad: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """FR: Convertit des radians en milliradians."""
    return phase_rad * RAD_TO_MRAD


def mrad_to_rad(phase_mrad: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """FR: Convertit des milliradians en radians."""
    return phase_mrad * MRAD_TO_RAD


def mm_to_um(length_mm: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """FR: Convertit des millimètres en micromètres."""
    return length_mm * 1000.0


def um_to_mm(length_um: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """FR: Convertit des micromètres en millimètres."""
    return length_um / 1000.0


# =============================================================================
# FONCTIONS DE GÉNÉRATION DE MODES
# =============================================================================

def _binomial_coefficient(n: int, k: int) -> float:
    """FR: Calcule le coefficient binomial C(n, k)."""
    if k < 0 or k > n:
        return 0.0
    if k == 0 or k == n:
        return 1.0
    result = 1.0
    for i in range(1, k + 1):
        result = result * (n - k + i) / i
    return result


def _zernike_radial_polynomial(n: int, m: int, R: np.ndarray) -> np.ndarray:
    """FR: Calcule le polynôme radial de Zernike R_n^m(r)."""
    result = np.zeros_like(R)
    for k in range((n - m) // 2 + 1):
        c1 = _binomial_coefficient(n - k, k)
        c2 = _binomial_coefficient(n - 2 * k, (n - m) // 2 - k)
        sign = (-1) ** k
        term = sign * c1 * c2 * (R ** (n - 2 * k))
        result += term
    return result


def _apply_zernike_normalization(Z: np.ndarray,
                                  n: int,
                                  m: int,
                                  normalization: Union[str, NormalizationType] = NormalizationType.NOLL) -> np.ndarray:
    """FR: Applique la normalisation au polynôme de Zernike."""
    if isinstance(normalization, str):
        normalization = NormalizationType(normalization.lower())
    
    if normalization == NormalizationType.PV:
        max_val = np.max(np.abs(Z))
        return Z / max_val if max_val > 0 else Z
    elif normalization == NormalizationType.RMS:
        rms = np.std(Z)
        return Z / rms if rms > 0 else Z
    elif normalization == NormalizationType.NOLL:
        if n == 0 and m == 0:
            return Z
        else:
            return Z * np.sqrt((n + 1) / np.pi)
    elif normalization == NormalizationType.WYANT:
        if m == 0:
            return Z * np.sqrt(n + 1)
        else:
            return Z * np.sqrt(2 * (n + 1))
    else:
        return Z


def generate_zernike_polynomial(n: int,
                              m: int,
                              x: Union[float, np.ndarray],
                              y: Union[float, np.ndarray],
                              ordering: Union[str, ZernikeOrdering] = ZernikeOrdering.NOLL,
                              normalization: Union[str, NormalizationType] = NormalizationType.NOLL,
                              wavelength_nm: float = DEFAULT_WAVELENGTH_NM) -> np.ndarray:
    """
    FR: Génère un polynôme de Zernike.
    
    Sources:
        - "Zernike polynomials and atmospheric turbulence" by J. W. Noll (1976)
        - "Principles of Adaptive Optics" by R. K. Tyson (1991), Ch. 2
    """
    if isinstance(ordering, str):
        ordering = ZernikeOrdering(ordering.lower())
    
    R = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    theta = np.where(R > 0, theta, 0.0)
    
    if n < abs(m) or (n - abs(m)) % 2 != 0:
        return np.zeros_like(R)
    
    radial_poly = _zernike_radial_polynomial(n, abs(m), R)
    
    if m >= 0:
        azimuthal_term = np.cos(abs(m) * theta)
    else:
        azimuthal_term = np.sin(abs(m) * theta)
    
    Z = radial_poly * azimuthal_term
    return _apply_zernike_normalization(Z, n, m, normalization)


def generate_zernike_modes(max_degree: int,
                          ordering: Union[str, ZernikeOrdering] = ZernikeOrdering.NOLL,
                          normalization: Union[str, NormalizationType] = NormalizationType.NOLL,
                          num_points: int = 512) -> List[Tuple[int, int, np.ndarray]]:
    """FR: Génère tous les modes de Zernike jusqu'à un degré maximal."""
    if isinstance(ordering, str):
        ordering = ZernikeOrdering(ordering.lower())
    if isinstance(normalization, str):
        normalization = NormalizationType(normalization.lower())
    
    modes = []
    x, y, X, Y = create_grid(num_points, diameter_mm=2.0)
    X_norm = X / 1.0
    Y_norm = Y / 1.0
    
    for n in range(max_degree + 1):
        for m in range(-n, n + 1, 2):
            Z = generate_zernike_polynomial(n, m, X_norm, Y_norm, ordering, normalization)
            modes.append((n, m, Z))
    
    return modes


def generate_legendre_polynomial(n: int,
                                x: Union[float, np.ndarray]) -> np.ndarray:
    """
    FR: Génère un polynôme de Legendre P_n(x).
    
    Sources: "Handbook of Mathematical Functions" by Abramowitz & Stegun (1964), Ch. 22
    """
    try:
        from scipy.special import eval_legendre
        return eval_legendre(n, x)
    except ImportError:
        if n == 0:
            return np.ones_like(x)
        elif n == 1:
            return x
        else:
            P_prev_prev = np.ones_like(x)
            P_prev = x
            for i in range(2, n + 1):
                P_current = ((2 * i - 1) * x * P_prev - (i - 1) * P_prev_prev) / i
                P_prev_prev = P_prev
                P_prev = P_current
            return P_prev


def generate_hermite_polynomial(n: int,
                                x: Union[float, np.ndarray]) -> np.ndarray:
    """FR: Génère un polynôme d'Hermite H_n(x)."""
    try:
        from scipy.special import hermite
        return hermite(n)(x)
    except ImportError:
        if n == 0:
            return np.ones_like(x)
        elif n == 1:
            return 2 * x
        else:
            H_prev_prev = np.ones_like(x)
            H_prev = 2 * x
            for i in range(2, n + 1):
                H_current = 2 * x * H_prev - 2 * (i - 1) * H_prev_prev
                H_prev_prev = H_prev
                H_prev = H_current
            return H_prev


def generate_laguerre_polynomial(n: int,
                                 m: int,
                                 x: Union[float, np.ndarray]) -> np.ndarray:
    """FR: Génère un polynôme de Laguerre généralisé L_n^m(x)."""
    try:
        from scipy.special import genlaguerre
        return genlaguerre(n, m)(x)
    except ImportError:
        logger.warning("scipy not available. Using approximate Laguerre polynomial.")
        return x**n


# =============================================================================
# FONCTIONS D'INTERPOLATION
# =============================================================================

def interpolate_1d(x: np.ndarray,
                   y: np.ndarray,
                   x_new: np.ndarray,
                   method: str = 'linear',
                   fill_value: float = 0.0) -> np.ndarray:
    """FR: Interpole un tableau 1D."""
    from scipy.interpolate import interp1d
    interp_func = interp1d(x, y, kind=method, bounds_error=False, fill_value=fill_value)
    return interp_func(x_new)


def interpolate_2d(X: np.ndarray,
                   Y: np.ndarray,
                   Z: np.ndarray,
                   X_new: np.ndarray,
                   Y_new: np.ndarray,
                   method: str = 'linear',
                   fill_value: float = 0.0) -> np.ndarray:
    """FR: Interpole un tableau 2D."""
    from scipy.interpolate import griddata
    points = np.column_stack((X.ravel(), Y.ravel()))
    values = Z.ravel()
    new_points = np.column_stack((X_new.ravel(), Y_new.ravel()))
    Z_new = griddata(points, values, new_points, method=method, fill_value=fill_value)
    return Z_new.reshape(X_new.shape)


def smooth_gaussian(data: np.ndarray,
                    sigma: float = 1.0) -> np.ndarray:
    """FR: Lisse un tableau avec un filtre gaussien."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(data, sigma=sigma)


# =============================================================================
# FONCTIONS DE PHASE
# =============================================================================

def wrap_phase(phase: np.ndarray,
               units: str = 'rad') -> np.ndarray:
    """FR: Enroule la phase entre -π et π (rad) ou -λ/2 et λ/2 (nm)."""
    if units == 'rad':
        return np.arctan2(np.sin(phase), np.cos(phase))
    elif units == 'nm':
        wavelength_nm = DEFAULT_WAVELENGTH_NM
        phase_rad = nm_to_rad(phase, wavelength_nm)
        phase_wrapped_rad = np.arctan2(np.sin(phase_rad), np.cos(phase_rad))
        return rad_to_nm(phase_wrapped_rad, wavelength_nm)
    else:
        raise ValueError(f"Unités inconnues: {units}")


def unwrap_phase(phase: np.ndarray) -> np.ndarray:
    """FR: Déroule la phase."""
    return np.unwrap(phase)


# =============================================================================
# FONCTIONS DE GÉNÉRATION ALÉATOIRE
# =============================================================================

def set_random_seed(seed: int) -> None:
    """FR: Définit la graine pour les nombres aléatoires."""
    np.random.seed(seed)


def generate_random_array(shape: Tuple[int, ...],
                          distribution: str = 'uniform',
                          low: float = 0.0,
                          high: float = 1.0,
                          seed: Optional[int] = None) -> np.ndarray:
    """FR: Génère un tableau de nombres aléatoires."""
    if seed is not None:
        set_random_seed(seed)
    if distribution == 'uniform':
        return np.random.uniform(low, high, size=shape)
    elif distribution == 'normal':
        return np.random.normal(loc=(low + high) / 2, scale=(high - low) / 6, size=shape)
    elif distribution == 'poisson':
        return np.random.poisson(lam=low, size=shape)
    else:
        raise ValueError(f"Distribution inconnue: {distribution}")


# =============================================================================
# FONCTIONS UTILITAIRES DIVERSES
# =============================================================================

def clip_array(array: np.ndarray,
               min_val: Optional[float] = None,
               max_val: Optional[float] = None) -> np.ndarray:
    """FR: Clip un tableau entre des valeurs minimales et maximales."""
    if min_val is not None and max_val is not None:
        return np.clip(array, min_val, max_val)
    elif min_val is not None:
        return np.maximum(array, min_val)
    elif max_val is not None:
        return np.minimum(array, max_val)
    else:
        return array


def circular_mask(shape: Tuple[int, int],
                   radius: float,
                   center: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """FR: Crée un masque circulaire."""
    ny, nx = shape
    if center is None:
        center = (nx / 2, ny / 2)
    y, x = np.ogrid[:ny, :nx]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    return dist_from_center <= radius


# =============================================================================
# TESTS UNITAIRES
# =============================================================================

class TestMathAndPhysicsTools:
    """FR: Tests unitaires pour MathAndPhysicsTools.py."""

    def test_create_grid(self):
        """FR: Test la création de grille."""
        x, y, X, Y = create_grid(10, diameter_mm=2.0)
        assert x.shape == (10,)
        assert X.shape == (10, 10)

    def test_handle_nan(self):
        """FR: Test la gestion des NaN."""
        data = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0]])
        result_zero = handle_nan(data, method='zero')
        assert not np.any(np.isnan(result_zero))

    def test_safe_divide(self):
        """FR: Test la division sûre."""
        result = safe_divide(np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, np.nan]), default=0.0)
        assert result[0] == 1.0
        assert result[1] == 0.0
        assert result[2] == 0.0

    def test_compute_pv_rms(self):
        """FR: Test le calcul du PV et du RMS."""
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        pv, rms = compute_pv_rms(data)
        assert pv == 5.0
        assert rms > 0

    def test_conversions(self):
        """FR: Test les conversions d'unités."""
        phase_nm = 100.0
        wavelength_nm = 633.0
        phase_rad = nm_to_rad(phase_nm, wavelength_nm)
        assert np.isclose(rad_to_nm(phase_rad, wavelength_nm), phase_nm, atol=1e-10)


if __name__ == "__main__":
    import unittest
    unittest.main()

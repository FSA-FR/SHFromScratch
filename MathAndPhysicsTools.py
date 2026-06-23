"""
MathAndPhysicsTools.py

FR: Module central pour TOUTES les fonctions mathématiques et physiques réutilisables.
    
    Ce module contient TOUTES les fonctions outils nécessaires pour le package SHFromScratch.
    Aucune fonction outil ne doit exister dans d'autres fichiers.
    
    Fonctionnalités principales :
    - Génération de bases de modes (Zernike, Legendre, Hermite-Gauss, Laguerre-Gauss)
    - Normalisation de phase (PV, RMS, Noll)
    - Conversions d'unités (phase, énergie, puissance, intensité, longueurs)
    - Création de grilles d'échantillonnage (cartésiennes, polaires)
    - Gestion des NaN et des erreurs numériques
    - Chargement de données depuis des fichiers
    - Calculs statistiques
    - Interpolation
    
    Règle d'or :
    - TOUTE fonction outil DOIT être ici.
    - Aucune fonction outil ne doit exister dans d'autres fichiers.
    - Les fonctions doivent gérer les NaN sans les propager.
    
    Unités gérées :
    - Phase : nm, rad, λ, mrad
    - Énergie : J, mJ, a.u.
    - Puissance : W, mW, a.u.
    - Intensité : W/m², W/cm², a.u.
    - Longueurs : mm, µm

EN: Central module for ALL reusable mathematical and physical functions.
    
    This module contains ALL utility functions needed for the SHFromScratch package.
    No utility functions should exist in other files.
    
    Main features:
    - Mode bases generation (Zernike, Legendre, Hermite-Gauss, Laguerre-Gauss)
    - Phase normalization (PV, RMS, Noll)
    - Unit conversions (phase, energy, power, intensity, lengths)
    - Sampling grid creation (Cartesian, polar)
    - NaN handling and numerical error management
    - Data loading from files
    - Statistical calculations
    - Interpolation
    
    Golden rule:
    - EVERY utility function MUST be here.
    - No utility functions should exist in other files.
    - Functions must handle NaN without propagating them.

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch

Dependencies:
    - numpy
    - scipy (for interpolation, optional)
    - csv (for file loading)

Sources:
    - "Zernike polynomials and atmospheric turbulence" by R. J. Noll (1976), JOSA 66(3), 207-211
    - "Basic wavefront aberration theory for optical metrology" by J. C. Wyant & K. Creath (1992), Applied Optics 31(20), 3923-3930
    - "Handbook of Mathematical Functions" by M. Abramowitz & I. A. Stegun (1964), Dover
    - "Lasers" by A. E. Siegman (1986), University Science Books
    - "Optical Imaging and Aberrations" by V. N. Mahajan (2001), SPIE Press
    - "Principles of Optics" by M. Born & E. Wolf (1999), Cambridge University Press
"""

import numpy as np
import logging
import csv
from typing import Tuple, Optional, Union, List, Dict
from enum import Enum


# =============================================================================
# CONFIGURATION GLOBALE
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MathAndPhysicsTools")

# Tolérance numérique
NUMERICAL_TOLERANCE = 1e-12

# Longueur d'onde par défaut (He-Ne laser)
DEFAULT_WAVELENGTH_NM = 633.0

# Constantes mathématiques
PI = np.pi
TWO_PI = 2 * np.pi
E = np.e

# Constantes physiques
SPEED_OF_LIGHT_M_PER_S = 299792458.0
PLANCK_CONSTANT_J_S = 6.62607015e-34


# =============================================================================
# ENUMS
# =============================================================================

class ZernikeOrdering(Enum):
    """FR: Ordre des polynômes de Zernike. EN: Zernike polynomial ordering."""
    NOLL = "Noll"
    WYANT = "Wyant"


class NormalizationType(Enum):
    """FR: Type de normalisation. EN: Normalization type."""
    PV = "PV"
    RMS = "RMS"
    NOLL = "Noll"


# =============================================================================
# 1. GESTION DES NaN ET ERREURS NUMÉRIQUES
# =============================================================================

def handle_nan(data: Union[np.ndarray, float, int, complex],
               method: str = 'zero',
               replacement_value: float = 0.0) -> Union[np.ndarray, float]:
    """
    FR: Gère les valeurs NaN et Inf.
    EN: Handles NaN and Inf values.
    
    Args:
        data: Données à traiter.
        method: 'zero', 'mean', 'median', 'nearest', 'ignore'.
        replacement_value: Valeur de remplacement pour 'zero'.
    
    Returns:
        Données sans NaN/Inf.
    """
    if isinstance(data, (int, float, complex)):
        if np.isnan(data) or np.isinf(data):
            if method == 'zero':
                return replacement_value if isinstance(data, (int, float)) else complex(replacement_value, 0)
            elif method == 'ignore':
                return None
            else:
                return replacement_value
        return data
    
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    data = np.where(np.isinf(data), np.nan, data)
    
    if method == 'zero':
        return np.where(np.isnan(data), replacement_value, data)
    elif method == 'mean':
        mean_val = np.nanmean(data)
        if np.isnan(mean_val):
            return np.full_like(data, replacement_value)
        return np.where(np.isnan(data), mean_val, data)
    elif method == 'median':
        median_val = np.nanmedian(data)
        if np.isnan(median_val):
            return np.full_like(data, replacement_value)
        return np.where(np.isnan(data), median_val, data)
    elif method == 'nearest':
        try:
            from scipy.ndimage import distance_transform_edt
            if np.all(np.isnan(data)):
                return np.full_like(data, replacement_value)
            indices = distance_transform_edt(np.isnan(data))
            nearest_indices = np.unravel_index(np.argmin(indices, axis=None), indices.shape)
            return np.where(np.isnan(data), data[nearest_indices], data)
        except ImportError:
            return np.where(np.isnan(data), replacement_value, data)
    elif method == 'ignore':
        if np.all(np.isnan(data)):
            return None
        return data
    else:
        raise ValueError(f"Méthode inconnue: {method}")


def safe_divide(a: Union[np.ndarray, float],
                 b: Union[np.ndarray, float],
                 replacement: float = 0.0) -> Union[np.ndarray, float]:
    """FR: Division sûre. EN: Safe division."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = a / b
    result = np.where((b == 0) | np.isnan(result) | np.isinf(result), replacement, result)
    return result


def safe_sqrt(x: Union[np.ndarray, float], replacement: float = 0.0) -> Union[np.ndarray, float]:
    """FR: Racine carrée sûre. EN: Safe square root."""
    result = np.sqrt(np.maximum(x, 0))
    return np.where(x < 0, replacement, result)


def safe_log(x: Union[np.ndarray, float], replacement: float = -np.inf) -> Union[np.ndarray, float]:
    """FR: Logarithme sûr. EN: Safe logarithm."""
    result = np.log(np.maximum(x, NUMERICAL_TOLERANCE))
    return np.where(x <= 0, replacement, result)


def safe_exp(x: Union[np.ndarray, float], max_value: float = 700.0) -> Union[np.ndarray, float]:
    """FR: Exponentielle sûre. EN: Safe exponential."""
    x_clipped = np.clip(x, -max_value, max_value)
    return np.exp(x_clipped)


# =============================================================================
# 2. STATISTIQUES
# =============================================================================

def compute_pv_rms(data: np.ndarray) -> Tuple[float, float]:
    """FR: Calcule PV et RMS. EN: Computes PV and RMS."""
    data_clean = handle_nan(data, method='zero')
    pv = float(np.max(data_clean) - np.min(data_clean))
    rms = float(np.sqrt(np.mean(data_clean**2)))
    return pv, rms


def compute_statistics(data: np.ndarray) -> Dict[str, float]:
    """FR: Calcule les statistiques. EN: Computes statistics."""
    data_clean = handle_nan(data, method='zero')
    return {
        'min': float(np.min(data_clean)),
        'max': float(np.max(data_clean)),
        'mean': float(np.mean(data_clean)),
        'std': float(np.std(data_clean)),
        'pv': float(np.max(data_clean) - np.min(data_clean)),
        'rms': float(np.sqrt(np.mean(data_clean**2)))
    }


def normalize_array(data: np.ndarray, method: str = 'max', target_value: float = 1.0) -> np.ndarray:
    """FR: Normalise un tableau. EN: Normalizes an array."""
    data_clean = handle_nan(data, method='zero')
    if method == 'max':
        max_val = np.max(data_clean)
        if max_val == 0:
            return data_clean
        return data_clean * (target_value / max_val)
    elif method == 'sum':
        sum_val = np.sum(data_clean)
        if sum_val == 0:
            return data_clean
        return data_clean * (target_value / sum_val)
    elif method == 'rms':
        rms_val = np.sqrt(np.mean(data_clean**2))
        if rms_val == 0:
            return data_clean
        return data_clean * (target_value / rms_val)
    elif method == 'mean':
        mean_val = np.mean(data_clean)
        if mean_val == 0:
            return data_clean
        return data_clean * (target_value / mean_val)
    else:
        raise ValueError(f"Méthode inconnue: {method}")


# =============================================================================
# 3. BASES DE MODES
# =============================================================================

def generate_zernike_polynomial(n: int, m: int, x: np.ndarray, y: np.ndarray,
                                ordering: ZernikeOrdering = ZernikeOrdering.NOLL,
                                normalization: NormalizationType = NormalizationType.NOLL) -> np.ndarray:
    """FR: Génère un polynôme de Zernike. EN: Generates a Zernike polynomial."""
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    R_nm = _radial_polynomial(n, abs(m), r)
    if m >= 0:
        Z = R_nm * np.cos(abs(m) * theta)
    else:
        Z = R_nm * np.sin(abs(m) * theta)
    
    if normalization == NormalizationType.NOLL:
        Z = Z * np.sqrt(2 * (n + 1) / (1 + (m == 0)))
    elif normalization == NormalizationType.PV:
        Z = Z * (2 if m != 0 else 1)
    elif normalization == NormalizationType.RMS:
        Z = Z * np.sqrt(2 * (n + 1))
    return Z


def generate_zernike_modes(n_modes: int,
                           ordering: ZernikeOrdering = ZernikeOrdering.NOLL,
                           grid_x: Optional[np.ndarray] = None,
                           grid_y: Optional[np.ndarray] = None,
                           num_points: int = 512,
                           diameter_mm: float = 10.0) -> List[Tuple[int, int, np.ndarray]]:
    """FR: Génère tous les modes de Zernike. EN: Generates all Zernike modes."""
    if grid_x is None or grid_y is None:
        grid_x, grid_y = create_grid(diameter_mm, num_points)
    r = np.sqrt(grid_x**2 + grid_y**2)
    r_max = np.max(r)
    x_norm = grid_x / r_max
    y_norm = grid_y / r_max
    
    modes = []
    for mode_idx in range(n_modes):
        if ordering == ZernikeOrdering.NOLL:
            n, m = _noll_to_zernike_indices(mode_idx)
        else:
            n, m = _wyant_to_zernike_indices(mode_idx)
        Z = generate_zernike_polynomial(n, m, x_norm, y_norm, ordering, NormalizationType.NOLL)
        modes.append((n, m, Z))
    return modes


def _noll_to_zernike_indices(mode_idx: int) -> Tuple[int, int]:
    """FR: Convertit un indice Noll en (n, m)."""
    n = 0
    while (n + 1) * (n + 2) // 2 <= mode_idx:
        n += 1
    m = 2 * mode_idx - n * (n + 2)
    return n, m


def _wyant_to_zernike_indices(mode_idx: int) -> Tuple[int, int]:
    """FR: Convertit un indice Wyant en (n, m)."""
    n = int(np.floor(np.sqrt(2 * mode_idx + 1)))
    m = 2 * mode_idx - n * (n + 1)
    return n, m


def _radial_polynomial(n: int, m: int, r: np.ndarray) -> np.ndarray:
    """FR: Calcule le polynôme radial. EN: Computes radial polynomial."""
    R = np.zeros_like(r)
    for k in range((n - m) // 2 + 1):
        term = (
            ((-1) ** k)
            * np.math.factorial(n - 2 * k)
            / (
                np.math.factorial(k)
                * np.math.factorial((n + m) // 2 - k)
                * np.math.factorial((n - m) // 2 - k)
            )
            * (r ** (n - 2 * k))
        )
        R += term
    return R


def generate_legendre_polynomial(n: int, x: np.ndarray) -> np.ndarray:
    """FR: Génère un polynôme de Legendre. EN: Generates a Legendre polynomial."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x
    else:
        P_prev = np.ones_like(x)
        P_curr = x
        for k in range(2, n + 1):
            P_next = ((2 * k - 1) * x * P_curr - (k - 1) * P_prev) / k
            P_prev, P_curr = P_curr, P_next
        return P_curr


def generate_legendre_modes(n_modes: int,
                            grid_x: Optional[np.ndarray] = None,
                            grid_y: Optional[np.ndarray] = None,
                            num_points: int = 512,
                            diameter_mm: float = 10.0) -> List[Tuple[int, np.ndarray]]:
    """FR: Génère tous les modes de Legendre. EN: Generates all Legendre modes."""
    if grid_x is None or grid_y is None:
        grid_x, grid_y = create_grid(diameter_mm, num_points)
    x_norm = 2 * (grid_x - np.min(grid_x)) / (np.max(grid_x) - np.min(grid_x)) - 1
    modes = []
    for n in range(n_modes):
        P_n = generate_legendre_polynomial(n, x_norm)
        modes.append((n, P_n))
    return modes


def generate_hermite_polynomial(n: int, x: np.ndarray) -> np.ndarray:
    """FR: Génère un polynôme de Hermite. EN: Generates a Hermite polynomial."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    else:
        H_prev = np.ones_like(x)
        H_curr = 2 * x
        for k in range(2, n + 1):
            H_next = 2 * x * H_curr - 2 * (k - 1) * H_prev
            H_prev, H_curr = H_curr, H_next
        return H_curr


def generate_laguerre_polynomial(p: int, l: int, x: np.ndarray) -> np.ndarray:
    """FR: Génère un polynôme de Laguerre généralisé. EN: Generates a generalized Laguerre polynomial."""
    if p == 0:
        return np.ones_like(x)
    elif p == 1:
        return (l + 1) - x
    else:
        L_prev = np.ones_like(x)
        L_curr = (l + 1) - x
        for k in range(2, p + 1):
            L_next = ((2 * k + l - 1 - x) * L_curr - (k + l - 1) * L_prev) / k
            L_prev, L_curr = L_curr, L_next
        return L_curr


# =============================================================================
# 4. NORMALISATION DE PHASE
# =============================================================================

def normalize_phase(phase: np.ndarray,
                    normalization: NormalizationType = NormalizationType.RMS,
                    target_value: float = 1.0,
                    wavelength_nm: float = DEFAULT_WAVELENGTH_NM) -> np.ndarray:
    """FR: Normalise une carte de phase. EN: Normalizes a phase map."""
    phase_clean = handle_nan(phase, method='zero')
    if normalization == NormalizationType.PV:
        current_pv = np.max(phase_clean) - np.min(phase_clean)
        if current_pv == 0:
            return phase_clean
        return phase_clean * (target_value * wavelength_nm) / current_pv
    elif normalization == NormalizationType.RMS:
        current_rms = np.sqrt(np.mean(phase_clean**2))
        if current_rms == 0:
            return phase_clean
        return phase_clean * (target_value * wavelength_nm) / current_rms
    else:
        current_rms = np.sqrt(np.mean(phase_clean**2))
        if current_rms == 0:
            return phase_clean
        return phase_clean * (target_value * wavelength_nm) / current_rms


# =============================================================================
# 5. CONVERSIONS D'UNITÉS - PHASE
# =============================================================================

def nm_to_rad(phase_nm: Union[np.ndarray, float], wavelength_nm: float = DEFAULT_WAVELENGTH_NM) -> Union[np.ndarray, float]:
    """FR: Convertit une phase en nm en rad. EN: Converts phase from nm to rad."""
    return handle_nan((2 * PI * phase_nm) / wavelength_nm, method='zero')


def rad_to_nm(phase_rad: Union[np.ndarray, float], wavelength_nm: float = DEFAULT_WAVELENGTH_NM) -> Union[np.ndarray, float]:
    """FR: Convertit une phase en rad en nm. EN: Converts phase from rad to nm."""
    return handle_nan((phase_rad * wavelength_nm) / TWO_PI, method='zero')


def lambda_to_nm(wavelength_lambda: Union[np.ndarray, float], reference_nm: float = DEFAULT_WAVELENGTH_NM) -> Union[np.ndarray, float]:
    """FR: Convertit une longueur en λ en nm. EN: Converts length from λ to nm."""
    return handle_nan(wavelength_lambda * reference_nm, method='zero')


def nm_to_lambda(wavelength_nm: Union[np.ndarray, float], reference_nm: float = DEFAULT_WAVELENGTH_NM) -> Union[np.ndarray, float]:
    """FR: Convertit une longueur en nm en λ. EN: Converts length from nm to λ."""
    return handle_nan(wavelength_nm / reference_nm, method='zero')


def rad_to_mrad(phase_rad: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """FR: Convertit une phase en rad en mrad. EN: Converts phase from rad to mrad."""
    return handle_nan(phase_rad * 1e3, method='zero')


def mrad_to_rad(phase_mrad: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """FR: Convertit une phase en mrad en rad. EN: Converts phase from mrad to rad."""
    return handle_nan(phase_mrad * 1e-3, method='zero')


# =============================================================================
# 6. CONVERSIONS D'UNITÉS - ÉNERGIE ET PUISSANCE
# =============================================================================

def J_to_mJ(energy_J: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """FR: Convertit une énergie en J en mJ. EN: Converts energy from J to mJ."""
    return handle_nan(energy_J * 1000, method='zero')


def mJ_to_J(energy_mJ: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """FR: Convertit une énergie en mJ en J. EN: Converts energy from mJ to J."""
    return handle_nan(energy_mJ / 1000, method='zero')


def W_to_mW(power_W: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """FR: Convertit une puissance en W en mW. EN: Converts power from W to mW."""
    return handle_nan(power_W * 1000, method='zero')


def mW_to_W(power_mW: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """FR: Convertit une puissance en mW en W. EN: Converts power from mW to W."""
    return handle_nan(power_mW / 1000, method='zero')


def W_m2_to_W_cm2(intensity_W_m2: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """FR: Convertit une intensité en W/m² en W/cm². EN: Converts intensity from W/m² to W/cm²."""
    return handle_nan(intensity_W_m2 / 10000, method='zero')


def W_cm2_to_W_m2(intensity_W_cm2: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """FR: Convertit une intensité en W/cm² en W/m². EN: Converts intensity from W/cm² to W/m²."""
    return handle_nan(intensity_W_cm2 * 10000, method='zero')


# =============================================================================
# 7. CONVERSIONS D'UNITÉS - ÉNERGIE, PUISSANCE, INTENSITÉ
# =============================================================================

def energy_to_power(energy: float, energy_unit: str, pulse_duration_s: float, power_unit: str = "W") -> float:
    """FR: Convertit une énergie en puissance. EN: Converts energy to power."""
    if energy_unit == "mJ":
        energy_J = mJ_to_J(energy)
    elif energy_unit == "J":
        energy_J = energy
    else:
        raise ValueError(f"Unité d'énergie inconnue: {energy_unit}")
    power_W = energy_J / pulse_duration_s
    if power_unit == "mW":
        return W_to_mW(power_W)
    elif power_unit == "W":
        return power_W
    else:
        raise ValueError(f"Unité de puissance inconnue: {power_unit}")


def power_to_energy(power: float, power_unit: str, pulse_duration_s: float, energy_unit: str = "J") -> float:
    """FR: Convertit une puissance en énergie. EN: Converts power to energy."""
    if power_unit == "mW":
        power_W = mW_to_W(power)
    elif power_unit == "W":
        power_W = power
    else:
        raise ValueError(f"Unité de puissance inconnue: {power_unit}")
    energy_J = power_W * pulse_duration_s
    if energy_unit == "mJ":
        return J_to_mJ(energy_J)
    elif energy_unit == "J":
        return energy_J
    else:
        raise ValueError(f"Unité d'énergie inconnue: {energy_unit}")


def power_to_intensity(power: float, power_unit: str, diameter_mm: float, intensity_unit: str = "W/m2") -> float:
    """FR: Convertit une puissance en intensité. EN: Converts power to intensity."""
    if power_unit == "mW":
        power_W = mW_to_W(power)
    elif power_unit == "W":
        power_W = power
    else:
        raise ValueError(f"Unité de puissance inconnue: {power_unit}")
    radius_m = diameter_mm / 2000
    area_m2 = PI * radius_m**2
    intensity_W_m2 = power_W / area_m2
    if intensity_unit == "W/cm2":
        return W_m2_to_W_cm2(intensity_W_m2)
    elif intensity_unit == "W/m2":
        return intensity_W_m2
    else:
        raise ValueError(f"Unité d'intensité inconnue: {intensity_unit}")


def intensity_to_power(intensity: float, intensity_unit: str, diameter_mm: float, power_unit: str = "W") -> float:
    """FR: Convertit une intensité en puissance. EN: Converts intensity to power."""
    if intensity_unit == "W/cm2":
        intensity_W_m2 = W_cm2_to_W_m2(intensity)
    elif intensity_unit == "W/m2":
        intensity_W_m2 = intensity
    else:
        raise ValueError(f"Unité d'intensité inconnue: {intensity_unit}")
    radius_m = diameter_mm / 2000
    area_m2 = PI * radius_m**2
    power_W = intensity_W_m2 * area_m2
    if power_unit == "mW":
        return W_to_mW(power_W)
    elif power_unit == "W":
        return power_W
    else:
        raise ValueError(f"Unité de puissance inconnue: {power_unit}")


# =============================================================================
# 8. CONVERSIONS D'UNITÉS - LONGUEURS
# =============================================================================

def mm_to_um(length_mm: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """FR: Convertit une longueur en mm en µm. EN: Converts length from mm to µm."""
    return handle_nan(length_mm * 1000, method='zero')


def um_to_mm(length_um: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """FR: Convertit une longueur en µm en mm. EN: Converts length from µm to mm."""
    return handle_nan(length_um / 1000, method='zero')


# =============================================================================
# 9. GRILLES D'ÉCHANTILLONNAGE
# =============================================================================

def create_grid(size_mm: float = 10.0, num_points: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """FR: Crée une grille 2D centrée en mm. EN: Creates a 2D centered grid in mm."""
    x = np.linspace(-size_mm / 2, size_mm / 2, num_points)
    y = np.linspace(-size_mm / 2, size_mm / 2, num_points)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    return xx, yy


def create_polar_grid(r_min: float = 0.0, r_max: float = 5.0,
                      theta_min: float = 0.0, theta_max: float = TWO_PI,
                      num_r: int = 256, num_theta: int = 512) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FR: Crée une grille polaire. EN: Creates a polar grid."""
    r = np.linspace(r_min, r_max, num_r)
    theta = np.linspace(theta_min, theta_max, num_theta)
    R, Theta = np.meshgrid(r, theta, indexing='ij')
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    return X, Y, R


def cartesian_to_polar(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """FR: Convertit des coordonnées cartésiennes en polaires. EN: Converts Cartesian to polar coordinates."""
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return handle_nan(r, method='zero'), handle_nan(theta, method='zero')


def polar_to_cartesian(r: np.ndarray, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """FR: Convertit des coordonnées polaires en cartésiennes. EN: Converts polar to Cartesian coordinates."""
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return handle_nan(x, method='zero'), handle_nan(y, method='zero')


# =============================================================================
# 10. CHARGEMENT DE DONNÉES
# =============================================================================

def load_data_from_file(file_path: str, delimiter: Optional[str] = None) -> np.ndarray:
    """FR: Charge des données depuis un fichier (txt ou csv). EN: Loads data from a file (txt or csv)."""
    if not file_path.endswith(('.txt', '.csv')):
        raise ValueError("Seuls les fichiers .txt et .csv sont supportés.")
    try:
        if file_path.endswith('.csv'):
            with open(file_path, 'r') as f:
                reader = csv.reader(f, delimiter=delimiter)
                data = [[float(val) for val in row] for row in reader]
        else:
            with open(file_path, 'r') as f:
                if delimiter is None:
                    first_line = f.readline()
                    f.seek(0)
                    if ';' in first_line:
                        delimiter = ';'
                    elif ',' in first_line:
                        delimiter = ','
                    elif '\t' in first_line:
                        delimiter = '\t'
                    else:
                        delimiter = ' '
                data = np.loadtxt(file_path, delimiter=delimiter)
                return data
        return np.array(data)
    except FileNotFoundError:
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")


# =============================================================================
# 11. FONCTIONS UTILITAIRES
# =============================================================================

def get_area_mm2(diameter_mm: float) -> float:
    """FR: Calcule la surface d'un cercle en mm². EN: Computes the area of a circle in mm²."""
    radius_mm = diameter_mm / 2
    return PI * radius_mm**2


def resample_to_grid(data: np.ndarray, new_shape: Tuple[int, int],
                      old_grid_x: Optional[np.ndarray] = None,
                      old_grid_y: Optional[np.ndarray] = None,
                      new_grid_x: Optional[np.ndarray] = None,
                      new_grid_y: Optional[np.ndarray] = None) -> np.ndarray:
    """FR: Rééchantillonne des données 2D sur une nouvelle grille. EN: Resamples 2D data to a new grid."""
    try:
        from scipy.interpolate import griddata
    except ImportError:
        logger.warning("scipy not available. Using nearest neighbor.")
        if old_grid_x is None or old_grid_y is None:
            old_grid_x, old_grid_y = create_grid(1.0, data.shape[1])
        if new_grid_x is None or new_grid_y is None:
            new_grid_x, new_grid_y = create_grid(1.0, new_shape[1])
        x_ratio = old_grid_x.shape[1] / new_shape[1]
        y_ratio = old_grid_x.shape[0] / new_shape[0]
        resampled = np.zeros(new_shape)
        for i in range(new_shape[0]):
            for j in range(new_shape[1]):
                old_i = int(i * y_ratio)
                old_j = int(j * x_ratio)
                if 0 <= old_i < data.shape[0] and 0 <= old_j < data.shape[1]:
                    resampled[i, j] = data[old_i, old_j]
        return resampled
    
    if old_grid_x is None or old_grid_y is None:
        old_grid_x, old_grid_y = create_grid(1.0, data.shape[1])
    if new_grid_x is None or new_grid_y is None:
        new_grid_x, new_grid_y = create_grid(1.0, new_shape[1])
    points = np.column_stack((old_grid_x.ravel(), old_grid_y.ravel()))
    values = data.ravel()
    new_points = np.column_stack((new_grid_x.ravel(), new_grid_y.ravel()))
    resampled = griddata(points, values, new_points, method='cubic', fill_value=0.0)
    return resampled.reshape(new_shape)


def interpolate_1d(x: np.ndarray, y: np.ndarray, x_new: np.ndarray, method: str = 'linear') -> np.ndarray:
    """FR: Interpole des données 1D. EN: Interpolates 1D data."""
    try:
        from scipy.interpolate import interp1d
        f = interp1d(x, y, kind=method, bounds_error=False, fill_value='extrapolate')
        return f(x_new)
    except ImportError:
        logger.warning("scipy not available. Using linear interpolation.")
        y_new = np.zeros_like(x_new)
        for i, x_val in enumerate(x_new):
            idx = np.searchsorted(x, x_val, side='right') - 1
            if idx < 0:
                y_new[i] = y[0]
            elif idx >= len(x) - 1:
                y_new[i] = y[-1]
            else:
                x0, x1 = x[idx], x[idx + 1]
                y0, y1 = y[idx], y[idx + 1]
                if x1 != x0:
                    y_new[i] = y0 + (y1 - y0) * (x_val - x0) / (x1 - x0)
                else:
                    y_new[i] = y0
        return y_new


# =============================================================================
# TESTS UNITAIRES
# =============================================================================

class TestMathAndPhysicsTools:
    """FR: Tests unitaires pour MathAndPhysicsTools.py."""

    def test_handle_nan(self):
        """FR: Test handle_nan."""
        data = np.array([1, np.nan, 3, np.inf])
        result = handle_nan(data, method='zero')
        assert np.all(np.isfinite(result))

    def test_safe_divide(self):
        """FR: Test safe_divide."""
        assert safe_divide(1, 0) == 0
        result = safe_divide(np.array([1, 2]), np.array([0, 1]))
        assert result[0] == 0 and result[1] == 2

    def test_nm_to_rad(self):
        """FR: Test nm_to_rad."""
        assert abs(nm_to_rad(633.0, 633.0) - TWO_PI) < 1e-10

    def test_rad_to_nm(self):
        """FR: Test rad_to_nm."""
        assert abs(rad_to_nm(TWO_PI, 633.0) - 633.0) < 1e-10

    def test_J_to_mJ(self):
        """FR: Test J_to_mJ."""
        assert J_to_mJ(1.0) == 1000.0

    def test_create_grid(self):
        """FR: Test create_grid."""
        x, y = create_grid(10.0, 128)
        assert x.shape == (128, 128)
        assert x.min() == -5.0 and x.max() == 5.0

    def test_compute_pv_rms(self):
        """FR: Test compute_pv_rms."""
        phase = np.array([[0, 100], [50, 150]])
        pv, rms = compute_pv_rms(phase)
        assert pv == 150.0

    def test_generate_zernike_polynomial(self):
        """FR: Test generate_zernike_polynomial."""
        x, y = create_grid(10.0, 128)
        x_norm = x / 5.0
        y_norm = y / 5.0
        Z = generate_zernike_polynomial(1, 1, x_norm, y_norm)
        assert Z.shape == (128, 128)


if __name__ == "__main__":
    import unittest
    unittest.main()

"""
MathAndPhysicsTools.py
FR: Module regroupant les fonctions mathématiques et physiques réutilisables pour le package SHFromScratch.
    Inclut les bases de modes (Zernike, Legendre, Hermite-Gauss, Laguerre-Gauss), les normalisations (PV, RMS),
    les conversions d'unités (énergie, puissance, intensité), et les outils de grille d'échantillonnage.

EN: Module grouping reusable mathematical and physical functions for the SHFromScratch package.
    Includes mode bases (Zernike, Legendre, Hermite-Gauss, Laguerre-Gauss), normalizations (PV, RMS),
    unit conversions (energy, power, intensity), and sampling grid tools.

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch
"""

import numpy as np
import logging
from typing import Tuple, Optional, Union
import csv


# =============================================================================
# 1. BASES DE MODES / MODE BASES
# =============================================================================

def generate_zernike_modes(
    n_modes: int,
    ordination: str = "Noll",
    grid_x: Optional[np.ndarray] = None,
    grid_y: Optional[np.ndarray] = None,
    max_radial: Optional[int] = None,
    max_azimuthal: Optional[int] = None,
) -> np.ndarray:
    """
    FR: Génère une matrice de modes de Zernike 2D pour une grille donnée.
        Les modes sont générés selon l'ordination spécifiée (Noll ou Wyant).

    EN: Generates a 2D matrix of Zernike modes for a given grid.
        Modes are generated according to the specified indexing (Noll or Wyant).

    Args:
        n_modes (int): Nombre total de modes de Zernike à générer.
        ordination (str): Type d'ordination, "Noll" (défaut) ou "Wyant".
        grid_x (np.ndarray, optional): Grille en x (2D). Si None, une grille par défaut est créée.
        grid_y (np.ndarray, optional): Grille en y (2D). Si None, une grille par défaut est créée.
        max_radial (int, optional): Ordre radial maximal. Si None, calculé automatiquement.
        max_azimuthal (int, optional): Ordre azimutal maximal. Si None, calculé automatiquement.

    Returns:
        np.ndarray: Matrice de taille (n_modes, height, width) contenant les modes de Zernike.

    Raises:
        ValueError: Si n_modes <= 0 ou si ordination est invalide.

    Sources:
        - Noll, R. J. (1976). "Zernike polynomials and atmospheric turbulence." JOSA, 66(3), 207-211.
        - Wyant, J. C., & Creath, K. (1992). "Basic wavefront aberration theory for optical metrology."
          Applied Optics, 31(20), 3923-3930.
    """
    if n_modes <= 0:
        raise ValueError("n_modes doit être supérieur à 0.")
    if ordination not in ["Noll", "Wyant"]:
        raise ValueError("ordination doit être 'Noll' ou 'Wyant'.")

    # Calcul des ordres maximaux si non spécifiés
    if max_radial is None or max_azimuthal is None:
        max_n = int(np.ceil((np.sqrt(8 * n_modes + 1) - 1) / 2))
        max_radial = max_n if max_radial is None else max_radial
        max_azimuthal = max_n if max_azimuthal is None else max_azimuthal

    # Création d'une grille par défaut si non fournie
    if grid_x is None or grid_y is None:
        size = 10.0  # mm
        num_points = 512
        grid_x, grid_y = create_grid(size, num_points)

    # Normalisation de la grille pour les polynômes de Zernike (rayon unitaire)
    r = np.sqrt(grid_x**2 + grid_y**2)
    theta = np.arctan2(grid_y, grid_x)
    r_max = np.max(r)
    r = r / r_max  # Normalisation à [0, 1]

    # Initialisation de la matrice des modes
    modes = np.zeros((n_modes, grid_x.shape[0], grid_x.shape[1]))

    # Génération des modes selon l'ordination
    for mode_idx in range(n_modes):
        if ordination == "Noll":
            n, m = _noll_to_zernike_indices(mode_idx)
        else:  # Wyant
            n, m = _wyant_to_zernike_indices(mode_idx)

        if n > max_radial or abs(m) > max_azimuthal:
            modes[mode_idx] = np.zeros_like(grid_x)
            continue

        # Calcul du polynôme radial
        R_nm = _radial_polynomial(n, abs(m), r)

        # Calcul du mode de Zernike
        if m >= 0:
            modes[mode_idx] = R_nm * np.cos(abs(m) * theta)
        else:
            modes[mode_idx] = R_nm * np.sin(abs(m) * theta)

    return modes


def _noll_to_zernike_indices(mode_idx: int) -> Tuple[int, int]:
    """
    FR: Convertit un indice Noll en indices (n, m) de Zernike.

    EN: Converts a Noll index to Zernike (n, m) indices.

    Args:
        mode_idx (int): Indice du mode selon l'ordination de Noll.

    Returns:
        Tuple[int, int]: (n, m) où n est l'ordre radial et m l'ordre azimutal.

    Sources:
        - Noll, R. J. (1976). "Zernike polynomials and atmospheric turbulence." JOSA, 66(3), 207-211.
    """
    n = 0
    while (n + 1) * (n + 2) // 2 <= mode_idx:
        n += 1
    m = 2 * mode_idx - n * (n + 2)
    return n, m


def _wyant_to_zernike_indices(mode_idx: int) -> Tuple[int, int]:
    """
    FR: Convertit un indice Wyant en indices (n, m) de Zernike.

    EN: Converts a Wyant index to Zernike (n, m) indices.

    Args:
        mode_idx (int): Indice du mode selon l'ordination de Wyant.

    Returns:
        Tuple[int, int]: (n, m) où n est l'ordre radial et m l'ordre azimutal.

    Sources:
        - Wyant, J. C., & Creath, K. (1992). "Basic wavefront aberration theory for optical metrology."
          Applied Optics, 31(20), 3923-3930.
    """
    n = int(np.floor(np.sqrt(2 * mode_idx + 1)))
    m = 2 * mode_idx - n * (n + 1)
    return n, m


def _radial_polynomial(n: int, m: int, r: np.ndarray) -> np.ndarray:
    """
    FR: Calcule le polynôme radial R_n^m(r) pour les polynômes de Zernike.

    EN: Computes the radial polynomial R_n^m(r) for Zernike polynomials.

    Args:
        n (int): Ordre radial.
        m (int): Ordre azimutal.
        r (np.ndarray): Grille radiale normalisée (0 <= r <= 1).

    Returns:
        np.ndarray: Polynôme radial évalué sur la grille r.

    Sources:
        - Noll, R. J. (1976). "Zernike polynomials and atmospheric turbulence." JOSA, 66(3), 207-211.
    """
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


def generate_legendre_modes(
    n_modes: int,
    grid_x: Optional[np.ndarray] = None,
    grid_y: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    FR: Génère une matrice de modes de Legendre 2D pour une grille donnée.

    EN: Generates a matrix of 2D Legendre modes for a given grid.

    Args:
        n_modes (int): Nombre de modes de Legendre à générer.
        grid_x (np.ndarray, optional): Grille en x (2D). Si None, une grille par défaut est créée.
        grid_y (np.ndarray, optional): Grille en y (2D). Si None, une grille par défaut est créée.

    Returns:
        np.ndarray: Matrice de taille (n_modes, height, width) contenant les modes de Legendre 2D.

    Raises:
        ValueError: Si n_modes <= 0.

    Sources:
        - Abramowitz, M., & Stegun, I. A. (1964). "Handbook of Mathematical Functions." Dover.
    """
    if n_modes <= 0:
        raise ValueError("n_modes doit être supérieur à 0.")

    if grid_x is None or grid_y is None:
        size = 10.0  # mm
        num_points = 512
        grid_x, grid_y = create_grid(size, num_points)

    # Normalisation de la grille pour les polynômes de Legendre (x, y ∈ [-1, 1])
    x_norm = 2 * (grid_x - np.min(grid_x)) / (np.max(grid_x) - np.min(grid_x)) - 1
    y_norm = 2 * (grid_y - np.min(grid_y)) / (np.max(grid_y) - np.min(grid_y)) - 1

    # Initialisation de la matrice des modes
    modes = np.zeros((n_modes, grid_x.shape[0], grid_x.shape[1]))

    # Génération des modes de Legendre 2D
    for mode_idx in range(n_modes):
        n = mode_idx  # Ordre du polynôme de Legendre
        P_n = _legendre_polynomial(n, x_norm)
        modes[mode_idx] = P_n

    return modes


def _legendre_polynomial(n: int, x: np.ndarray) -> np.ndarray:
    """
    FR: Calcule le polynôme de Legendre P_n(x).

    EN: Computes the Legendre polynomial P_n(x).

    Args:
        n (int): Ordre du polynôme.
        x (np.ndarray): Grille 1D ou 2D.

    Returns:
        np.ndarray: Polynôme de Legendre P_n(x).

    Sources:
        - Abramowitz, M., & Stegun, I. A. (1964). "Handbook of Mathematical Functions." Dover.
    """
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


def generate_hermite_gauss_modes(
    n_modes: int,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    FR: Génère une matrice de modes de Hermite-Gauss pour une grille donnée.

    EN: Generates a matrix of Hermite-Gauss modes for a given grid.

    Args:
        n_modes (int): Nombre de modes à générer (n + m <= n_modes).
        grid_x (np.ndarray): Grille en x (2D).
        grid_y (np.ndarray): Grille en y (2D).
        sigma (float): Écart-type pour la gaussienne. Default: 1.0.

    Returns:
        np.ndarray: Matrice de taille (n_modes, height, width) contenant les modes de Hermite-Gauss.

    Sources:
        - Siegman, A. E. (1986). "Lasers." University Science Books.
    """
    modes = []
    for n in range(n_modes + 1):
        for m in range(n_modes + 1 - n):
            if n + m > n_modes:
                continue
            H_n = _hermite_polynomial(n, grid_x / sigma)
            H_m = _hermite_polynomial(m, grid_y / sigma)
            gauss = np.exp(-(grid_x**2 + grid_y**2) / (2 * sigma**2))
            mode = H_n * H_m * gauss
            modes.append(mode)
    return np.array(modes)[:n_modes]


def _hermite_polynomial(n: int, x: np.ndarray) -> np.ndarray:
    """
    FR: Calcule le polynôme de Hermite H_n(x).

    EN: Computes the Hermite polynomial H_n(x).

    Args:
        n (int): Ordre du polynôme.
        x (np.ndarray): Grille 1D ou 2D.

    Returns:
        np.ndarray: Polynôme de Hermite H_n(x).

    Sources:
        - Abramowitz, M., & Stegun, I. A. (1964). "Handbook of Mathematical Functions." Dover.
    """
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


def generate_laguerre_gauss_modes(
    n_modes: int,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    sigma: float = 1.0,
    p_max: int = 5,
    l_max: int = 5,
) -> np.ndarray:
    """
    FR: Génère une matrice de modes de Laguerre-Gauss pour une grille donnée.

    EN: Generates a matrix of Laguerre-Gauss modes for a given grid.

    Args:
        n_modes (int): Nombre de modes à générer.
        grid_x (np.ndarray): Grille en x (2D).
        grid_y (np.ndarray): Grille en y (2D).
        sigma (float): Écart-type pour la gaussienne. Default: 1.0.
        p_max (int): Ordre radial maximal. Default: 5.
        l_max (int): Ordre azimutal maximal. Default: 5.

    Returns:
        np.ndarray: Matrice de taille (n_modes, height, width) contenant les modes de Laguerre-Gauss.

    Sources:
        - Siegman, A. E. (1986). "Lasers." University Science Books.
    """
    r = np.sqrt(grid_x**2 + grid_y**2)
    theta = np.arctan2(grid_y, grid_x)
    modes = []

    for p in range(p_max + 1):
        for l in range(-l_max, l_max + 1):
            if len(modes) >= n_modes:
                break
            L_p_l = _generalized_laguerre_polynomial(p, abs(l), r**2 / sigma**2)
            gauss = np.exp(-r**2 / (2 * sigma**2))
            if l >= 0:
                angular = np.cos(l * theta)
            else:
                angular = np.sin(abs(l) * theta)
            mode = L_p_l * gauss * angular
            modes.append(mode)

    return np.array(modes)[:n_modes]


def _generalized_laguerre_polynomial(p: int, l: int, x: np.ndarray) -> np.ndarray:
    """
    FR: Calcule le polynôme de Laguerre généralisé L_p^l(x).

    EN: Computes the generalized Laguerre polynomial L_p^l(x).

    Args:
        p (int): Ordre radial.
        l (int): Ordre azimutal.
        x (np.ndarray): Grille 1D ou 2D.

    Returns:
        np.ndarray: Polynôme de Laguerre généralisé L_p^l(x).

    Sources:
        - Abramowitz, M., & Stegun, I. A. (1964). "Handbook of Mathematical Functions." Dover.
    """
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
# 2. NORMALISATIONS / NORMALIZATIONS
# =============================================================================

def normalize_phase(
    phase: np.ndarray,
    normalization: str = "RMS",
    target_value: float = 1.0,
    wavelength_nm: float = 633.0,
) -> np.ndarray:
    """
    FR: Normalise une carte de phase selon PV (Peak-to-Valley) ou RMS (Root Mean Square).

    EN: Normalizes a phase map according to PV (Peak-to-Valley) or RMS (Root Mean Square).

    Args:
        phase (np.ndarray): Carte de phase en nm.
        normalization (str): Type de normalisation, "PV" ou "RMS" (défaut: "RMS").
        target_value (float): Valeur cible pour la normalisation (ex: 1.0 pour 1λ).
        wavelength_nm (float): Longueur d'onde en nm (défaut: 633.0).

    Returns:
        np.ndarray: Phase normalisée en nm.

    Raises:
        ValueError: Si normalization n'est pas "PV" ou "RMS".

    Sources:
        - Mahajan, V. N. (2001). "Optical Imaging and Aberrations." SPIE Press.
    """
    if normalization not in ["PV", "RMS"]:
        raise ValueError("normalization doit être 'PV' ou 'RMS'.")

    if normalization == "PV":
        current_pv = np.max(phase) - np.min(phase)
        if current_pv == 0:
            return phase
        normalized_phase = phase * (target_value * wavelength_nm) / current_pv
    else:  # RMS
        current_rms = np.sqrt(np.mean(phase**2))
        if current_rms == 0:
            return phase
        normalized_phase = phase * (target_value * wavelength_nm) / current_rms

    return normalized_phase


# =============================================================================
# 3. CONVERSIONS D'UNITÉS / UNIT CONVERSIONS
# =============================================================================

# --- Conversions de phase ---
def nm_to_rad(phase_nm: Union[np.ndarray, float], wavelength_nm: float) -> Union[np.ndarray, float]:
    """
    FR: Convertit une phase en nm en rad.

    EN: Converts a phase from nm to rad.

    Args:
        phase_nm (np.ndarray or float): Phase en nm.
        wavelength_nm (float): Longueur d'onde en nm.

    Returns:
        np.ndarray or float: Phase en rad.

    Formula:
        phase_rad = (2π * phase_nm) / wavelength_nm
    """
    return (2 * np.pi * phase_nm) / wavelength_nm


def rad_to_nm(phase_rad: Union[np.ndarray, float], wavelength_nm: float) -> Union[np.ndarray, float]:
    """
    FR: Convertit une phase en rad en nm.

    EN: Converts a phase from rad to nm.

    Args:
        phase_rad (np.ndarray or float): Phase en rad.
        wavelength_nm (float): Longueur d'onde en nm.

    Returns:
        np.ndarray or float: Phase en nm.

    Formula:
        phase_nm = (phase_rad * wavelength_nm) / (2π)
    """
    return (phase_rad * wavelength_nm) / (2 * np.pi)


def lambda_to_nm(wavelength_lambda: Union[np.ndarray, float], reference_nm: float = 633.0) -> Union[np.ndarray, float]:
    """
    FR: Convertit une longueur d'onde en λ (unité de longueur d'onde) en nm.

    EN: Converts a wavelength from λ (wavelength unit) to nm.

    Args:
        wavelength_lambda (np.ndarray or float): Longueur d'onde en λ.
        reference_nm (float): Longueur d'onde de référence en nm. Default: 633.0.

    Returns:
        np.ndarray or float: Longueur d'onde en nm.
    """
    return wavelength_lambda * reference_nm


def nm_to_lambda(wavelength_nm: Union[np.ndarray, float], reference_nm: float = 633.0) -> Union[np.ndarray, float]:
    """
    FR: Convertit une longueur d'onde en nm en λ (unité de longueur d'onde).

    EN: Converts a wavelength from nm to λ (wavelength unit).

    Args:
        wavelength_nm (np.ndarray or float): Longueur d'onde en nm.
        reference_nm (float): Longueur d'onde de référence en nm. Default: 633.0.

    Returns:
        np.ndarray or float: Longueur d'onde en λ.
    """
    return wavelength_nm / reference_nm


def rad_to_mrad(phase_rad: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    FR: Convertit une phase en rad en mrad.

    EN: Converts a phase from rad to mrad.

    Args:
        phase_rad (np.ndarray or float): Phase en rad.

    Returns:
        np.ndarray or float: Phase en mrad.
    """
    return phase_rad * 1e3


def mrad_to_rad(phase_mrad: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    FR: Convertit une phase en mrad en rad.

    EN: Converts a phase from mrad to rad.

    Args:
        phase_mrad (np.ndarray or float): Phase en mrad.

    Returns:
        np.ndarray or float: Phase en rad.
    """
    return phase_mrad * 1e-3


# --- Conversions d'énergie et de puissance ---

def J_to_mJ(energy_J: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    FR: Convertit une énergie en Joules (J) en milliJoules (mJ).

    EN: Converts energy from Joules (J) to milliJoules (mJ).

    Args:
        energy_J (np.ndarray or float): Énergie en Joules.

    Returns:
        np.ndarray or float: Énergie en milliJoules.

    Formula:
        energy_mJ = energy_J * 1000
    """
    return energy_J * 1000


def mJ_to_J(energy_mJ: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    FR: Convertit une énergie en milliJoules (mJ) en Joules (J).

    EN: Converts energy from milliJoules (mJ) to Joules (J).

    Args:
        energy_mJ (np.ndarray or float): Énergie en milliJoules.

    Returns:
        np.ndarray or float: Énergie en Joules.

    Formula:
        energy_J = energy_mJ / 1000
    """
    return energy_mJ / 1000


def W_to_mW(power_W: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    FR: Convertit une puissance en Watts (W) en milliWatts (mW).

    EN: Converts power from Watts (W) to milliWatts (mW).

    Args:
        power_W (np.ndarray or float): Puissance en Watts.

    Returns:
        np.ndarray or float: Puissance en milliWatts.

    Formula:
        power_mW = power_W * 1000
    """
    return power_W * 1000


def mW_to_W(power_mW: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    FR: Convertit une puissance en milliWatts (mW) en Watts (W).

    EN: Converts power from milliWatts (mW) to Watts (W).

    Args:
        power_mW (np.ndarray or float): Puissance en milliWatts.

    Returns:
        np.ndarray or float: Puissance en Watts.

    Formula:
        power_W = power_mW / 1000
    """
    return power_mW / 1000


def W_m2_to_W_cm2(intensity_W_m2: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    FR: Convertit une intensité en W/m² en W/cm².

    EN: Converts intensity from W/m² to W/cm².

    Args:
        intensity_W_m2 (np.ndarray or float): Intensité en W/m².

    Returns:
        np.ndarray or float: Intensité en W/cm².

    Formula:
        intensity_W_cm2 = intensity_W_m2 / 10000
    """
    return intensity_W_m2 / 10000


def W_cm2_to_W_m2(intensity_W_cm2: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    FR: Convertit une intensité en W/cm² en W/m².

    EN: Converts intensity from W/cm² to W/m².

    Args:
        intensity_W_cm2 (np.ndarray or float): Intensité en W/cm².

    Returns:
        np.ndarray or float: Intensité en W/m².

    Formula:
        intensity_W_m2 = intensity_W_cm2 * 10000
    """
    return intensity_W_cm2 * 10000


# --- Conversions entre énergie, puissance et intensité ---

def energy_to_power(
    energy: float,
    energy_unit: str,
    pulse_duration_s: float,
    power_unit: str = "W",
) -> float:
    """
    FR: Convertit une énergie en puissance en utilisant la durée d'impulsion.

    EN: Converts energy to power using pulse duration.

    Args:
        energy (float): Énergie.
        energy_unit (str): Unité de l'énergie ("J" ou "mJ").
        pulse_duration_s (float): Durée de l'impulsion en secondes.
        power_unit (str): Unité de la puissance souhaitée ("W" ou "mW"). Default: "W".

    Returns:
        float: Puissance dans l'unité spécifiée.

    Formula:
        power_W = energy_J / pulse_duration_s
    """
    # Convertir l'énergie en Joules
    if energy_unit == "mJ":
        energy_J = mJ_to_J(energy)
    elif energy_unit == "J":
        energy_J = energy
    else:
        raise ValueError(f"Unité d'énergie inconnue : {energy_unit}. Utilisez 'J' ou 'mJ'.")

    # Calculer la puissance en Watts
    power_W = energy_J / pulse_duration_s

    # Convertir en mW si nécessaire
    if power_unit == "mW":
        return W_to_mW(power_W)
    elif power_unit == "W":
        return power_W
    else:
        raise ValueError(f"Unité de puissance inconnue : {power_unit}. Utilisez 'W' ou 'mW'.")


def power_to_energy(
    power: float,
    power_unit: str,
    pulse_duration_s: float,
    energy_unit: str = "J",
) -> float:
    """
    FR: Convertit une puissance en énergie en utilisant la durée d'impulsion.

    EN: Converts power to energy using pulse duration.

    Args:
        power (float): Puissance.
        power_unit (str): Unité de la puissance ("W" ou "mW").
        pulse_duration_s (float): Durée de l'impulsion en secondes.
        energy_unit (str): Unité de l'énergie souhaitée ("J" ou "mJ"). Default: "J".

    Returns:
        float: Énergie dans l'unité spécifiée.

    Formula:
        energy_J = power_W * pulse_duration_s
    """
    # Convertir la puissance en Watts
    if power_unit == "mW":
        power_W = mW_to_W(power)
    elif power_unit == "W":
        power_W = power
    else:
        raise ValueError(f"Unité de puissance inconnue : {power_unit}. Utilisez 'W' ou 'mW'.")

    # Calculer l'énergie en Joules
    energy_J = power_W * pulse_duration_s

    # Convertir en mJ si nécessaire
    if energy_unit == "mJ":
        return J_to_mJ(energy_J)
    elif energy_unit == "J":
        return energy_J
    else:
        raise ValueError(f"Unité d'énergie inconnue : {energy_unit}. Utilisez 'J' ou 'mJ'.")


def power_to_intensity(
    power: float,
    power_unit: str,
    diameter_mm: float,
    intensity_unit: str = "W/m2",
) -> float:
    """
    FR: Convertit une puissance en intensité en utilisant le diamètre du faisceau.

    EN: Converts power to intensity using beam diameter.

    Args:
        power (float): Puissance du faisceau.
        power_unit (str): Unité de la puissance ("W" ou "mW").
        diameter_mm (float): Diamètre du faisceau en mm.
        intensity_unit (str): Unité d'intensité souhaitée ("W/m2" ou "W/cm2"). Default: "W/m2".

    Returns:
        float: Intensité dans l'unité spécifiée.

    Formula:
        intensity_W_m2 = power_W / (π * (radius_m)^2)
    """
    # Convertir la puissance en Watts
    if power_unit == "mW":
        power_W = mW_to_W(power)
    elif power_unit == "W":
        power_W = power
    else:
        raise ValueError(f"Unité de puissance inconnue : {power_unit}. Utilisez 'W' ou 'mW'.")

    # Calculer la surface en m²
    radius_m = diameter_mm / 2000  # Conversion mm → m
    area_m2 = np.pi * radius_m**2

    # Calculer l'intensité en W/m²
    intensity_W_m2 = power_W / area_m2

    # Convertir en W/cm² si nécessaire
    if intensity_unit == "W/cm2":
        return W_m2_to_W_cm2(intensity_W_m2)
    elif intensity_unit == "W/m2":
        return intensity_W_m2
    else:
        raise ValueError(f"Unité d'intensité inconnue : {intensity_unit}. Utilisez 'W/m2' ou 'W/cm2'.")


def intensity_to_power(
    intensity: float,
    intensity_unit: str,
    diameter_mm: float,
    power_unit: str = "W",
) -> float:
    """
    FR: Convertit une intensité en puissance en utilisant le diamètre du faisceau.

    EN: Converts intensity to power using beam diameter.

    Args:
        intensity (float): Intensité du faisceau.
        intensity_unit (str): Unité de l'intensité ("W/m2" ou "W/cm2").
        diameter_mm (float): Diamètre du faisceau en mm.
        power_unit (str): Unité de puissance souhaitée ("W" ou "mW"). Default: "W".

    Returns:
        float: Puissance dans l'unité spécifiée.

    Formula:
        power_W = intensity_W_m2 * (π * (radius_m)^2)
    """
    # Convertir l'intensité en W/m²
    if intensity_unit == "W/cm2":
        intensity_W_m2 = W_cm2_to_W_m2(intensity)
    elif intensity_unit == "W/m2":
        intensity_W_m2 = intensity
    else:
        raise ValueError(f"Unité d'intensité inconnue : {intensity_unit}. Utilisez 'W/m2' ou 'W/cm2'.")

    # Calculer la surface en m²
    radius_m = diameter_mm / 2000  # Conversion mm → m
    area_m2 = np.pi * radius_m**2

    # Calculer la puissance en Watts
    power_W = intensity_W_m2 * area_m2

    # Convertir en mW si nécessaire
    if power_unit == "mW":
        return W_to_mW(power_W)
    elif power_unit == "W":
        return power_W
    else:
        raise ValueError(f"Unité de puissance inconnue : {power_unit}. Utilisez 'W' ou 'mW'.")


# =============================================================================
# 4. GRILLES D'ÉCHANTILLONNAGE / SAMPLING GRIDS
# =============================================================================

def create_grid(
    size_mm: float,
    num_points: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    FR: Crée une grille 2D centrée en mm.

    EN: Creates a 2D centered grid in mm.

    Args:
        size_mm (float): Taille de la grille en mm.
        num_points (int): Nombre de points par dimension (défaut: 512).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Grilles x et y en mm.
    """
    x = np.linspace(-size_mm / 2, size_mm / 2, num_points)
    y = np.linspace(-size_mm / 2, size_mm / 2, num_points)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    return xx, yy


def resample_to_grid(
    data: np.ndarray,
    new_shape: Tuple[int, int],
    old_grid_x: Optional[np.ndarray] = None,
    old_grid_y: Optional[np.ndarray] = None,
    new_grid_x: Optional[np.ndarray] = None,
    new_grid_y: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    FR: Rééchantillonne des données 2D sur une nouvelle grille en utilisant l'interpolation.

    EN: Resamples 2D data to a new grid using interpolation.

    Args:
        data (np.ndarray): Données 2D à rééchantillonner.
        new_shape (Tuple[int, int]): Nouvelle forme (height, width).
        old_grid_x (np.ndarray, optional): Grille x originale. Si None, supposée linéaire.
        old_grid_y (np.ndarray, optional): Grille y originale. Si None, supposée linéaire.
        new_grid_x (np.ndarray, optional): Nouvelle grille x. Si None, créée avec create_grid.
        new_grid_y (np.ndarray, optional): Nouvelle grille y. Si None, créée avec create_grid.

    Returns:
        np.ndarray: Données rééchantillonnées.

    Notes:
        Utilise scipy.interpolate.griddata si les grilles sont fournies, sinon utilise np.interp.
    """
    from scipy.interpolate import griddata

    if old_grid_x is None or old_grid_y is None:
        old_grid_x, old_grid_y = create_grid(1.0, data.shape[1])
    if new_grid_x is None or new_grid_y is None:
        new_grid_x, new_grid_y = create_grid(1.0, new_shape[1])

    # Aplatir les données et les grilles pour griddata
    points = np.column_stack((old_grid_x.ravel(), old_grid_y.ravel()))
    values = data.ravel()
    new_points = np.column_stack((new_grid_x.ravel(), new_grid_y.ravel()))

    # Rééchantillonnage
    resampled = griddata(points, values, new_points, method='cubic', fill_value=0.0)
    return resampled.reshape(new_shape)


# =============================================================================
# 5. CHARGEMENT DE DONNÉES / DATA LOADING
# =============================================================================

def load_data_from_file(
    file_path: str,
    delimiter: str = None,
) -> np.ndarray:
    """
    FR: Charge une carte d'intensité ou de phase depuis un fichier (txt ou csv).

    EN: Loads an intensity or phase map from a file (txt or csv).

    Args:
        file_path (str): Chemin vers le fichier.
        delimiter (str, optional): Délimiteur pour les fichiers txt/csv. Default: None (auto-detect).

    Returns:
        np.ndarray: Données chargées sous forme de tableau 2D.

    Raises:
        ValueError: Si le fichier n'est pas au format txt ou csv.
        FileNotFoundError: Si le fichier n'existe pas.
    """
    if not file_path.endswith(('.txt', '.csv')):
        raise ValueError("Seuls les fichiers .txt et .csv sont supportés.")

    try:
        if file_path.endswith('.csv'):
            with open(file_path, 'r') as f:
                reader = csv.reader(f, delimiter=delimiter)
                data = [[float(val) for val in row] for row in reader]
        else:  # .txt
            with open(file_path, 'r') as f:
                if delimiter is None:
                    # Essayer de détecter le délimiteur
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
# 6. FONCTIONS UTILITAIRES / UTILITY FUNCTIONS
# =============================================================================

def compute_pv_rms(phase: np.ndarray) -> Tuple[float, float]:
    """
    FR: Calcule le PV (Peak-to-Valley) et le RMS (Root Mean Square) d'une carte de phase.

    EN: Computes the PV (Peak-to-Valley) and RMS (Root Mean Square) of a phase map.

    Args:
        phase (np.ndarray): Carte de phase en nm.

    Returns:
        Tuple[float, float]: (PV, RMS) en nm.
    """
    pv = np.max(phase) - np.min(phase)
    rms = np.sqrt(np.mean(phase**2))
    return pv, rms


def get_area_mm2(diameter_mm: float) -> float:
    """
    FR: Calcule la surface du faisceau en mm².

    EN: Computes the beam area in mm².

    Args:
        diameter_mm (float): Diamètre du faisceau en mm.

    Returns:
        float: Surface en mm².
    """
    radius_mm = diameter_mm / 2
    return np.pi * radius_mm**2


# =============================================================================
# 7. TESTS UNITAIRES / UNIT TESTS
# =============================================================================

class TestMathAndPhysicsTools:
    """
    FR: Classe de tests unitaires pour MathAndPhysicsTools.py.
    EN: Unit test class for MathAndPhysicsTools.py.
    """

    def test_generate_zernike_modes(self):
        """Test la génération des modes de Zernike."""
        grid_x, grid_y = create_grid(10.0, 128)
        modes = generate_zernike_modes(10, "Noll", grid_x, grid_y)
        assert modes.shape == (10, 128, 128), f"Expected shape (10, 128, 128), got {modes.shape}"

    def test_normalize_phase_RMS(self):
        """Test la normalisation RMS d'une phase."""
        phase = np.random.rand(100, 100) * 100
        normalized = normalize_phase(phase, "RMS", 1.0, 633.0)
        rms = np.sqrt(np.mean(normalized**2))
        assert abs(rms - 633.0) < 1e-5, f"Expected RMS=633.0, got {rms}"

    def test_normalize_phase_PV(self):
        """Test la normalisation PV d'une phase."""
        phase = np.random.rand(100, 100) * 100
        normalized = normalize_phase(phase, "PV", 1.0, 633.0)
        pv = np.max(normalized) - np.min(normalized)
        assert abs(pv - 633.0) < 1e-5, f"Expected PV=633.0, got {pv}"

    def test_nm_to_rad(self):
        """Test la conversion nm → rad."""
        phase_nm = np.array([633.0, 1266.0])
        phase_rad = nm_to_rad(phase_nm, 633.0)
        expected = np.array([2 * np.pi, 4 * np.pi])
        np.testing.assert_allclose(phase_rad, expected, rtol=1e-5)

    def test_rad_to_nm(self):
        """Test la conversion rad → nm."""
        phase_rad = np.array([2 * np.pi, 4 * np.pi])
        phase_nm = rad_to_nm(phase_rad, 633.0)
        expected = np.array([633.0, 1266.0])
        np.testing.assert_allclose(phase_nm, expected, rtol=1e-5)

    def test_J_to_mJ(self):
        """Test la conversion J → mJ."""
        energy_J = 0.001
        energy_mJ = J_to_mJ(energy_J)
        assert energy_mJ == 1.0, f"Expected 1.0 mJ, got {energy_mJ}"

    def test_mJ_to_J(self):
        """Test la conversion mJ → J."""
        energy_mJ = 1.0
        energy_J = mJ_to_J(energy_mJ)
        assert energy_J == 0.001, f"Expected 0.001 J, got {energy_J}"

    def test_W_to_mW(self):
        """Test la conversion W → mW."""
        power_W = 0.001
        power_mW = W_to_mW(power_W)
        assert power_mW == 1.0, f"Expected 1.0 mW, got {power_mW}"

    def test_mW_to_W(self):
        """Test la conversion mW → W."""
        power_mW = 1.0
        power_W = mW_to_W(power_mW)
        assert power_W == 0.001, f"Expected 0.001 W, got {power_W}"

    def test_W_m2_to_W_cm2(self):
        """Test la conversion W/m² → W/cm²."""
        intensity_W_m2 = 10000.0
        intensity_W_cm2 = W_m2_to_W_cm2(intensity_W_m2)
        assert intensity_W_cm2 == 1.0, f"Expected 1.0 W/cm², got {intensity_W_cm2}"

    def test_W_cm2_to_W_m2(self):
        """Test la conversion W/cm² → W/m²."""
        intensity_W_cm2 = 1.0
        intensity_W_m2 = W_cm2_to_W_m2(intensity_W_cm2)
        assert intensity_W_m2 == 10000.0, f"Expected 10000.0 W/m², got {intensity_W_m2}"

    def test_energy_to_power(self):
        """Test la conversion énergie → puissance."""
        energy_mJ = 1.0  # 1 mJ
        pulse_duration_s = 0.001  # 1 ms
        power_W = energy_to_power(energy_mJ, "mJ", pulse_duration_s, "W")
        assert power_W == 1.0, f"Expected 1.0 W, got {power_W}"

    def test_power_to_energy(self):
        """Test la conversion puissance → énergie."""
        power_W = 1.0  # 1 W
        pulse_duration_s = 0.001  # 1 ms
        energy_mJ = power_to_energy(power_W, "W", pulse_duration_s, "mJ")
        assert energy_mJ == 1.0, f"Expected 1.0 mJ, got {energy_mJ}"

    def test_power_to_intensity(self):
        """Test la conversion puissance → intensité."""
        power_W = 1.0  # 1 W
        diameter_mm = 10.0  # 10 mm
        intensity_W_m2 = power_to_intensity(power_W, "W", diameter_mm, "W/m2")
        expected = 1.0 / (np.pi * (0.005)**2)  # 1 W / (π * (5e-3 m)^2)
        assert abs(intensity_W_m2 - expected) < 1e-5, f"Expected {expected} W/m², got {intensity_W_m2}"

    def test_intensity_to_power(self):
        """Test la conversion intensité → puissance."""
        intensity_W_m2 = 10000.0  # 10000 W/m²
        diameter_mm = 10.0  # 10 mm
        power_W = intensity_to_power(intensity_W_m2, "W/m2", diameter_mm, "W")
        expected = 10000.0 * (np.pi * (0.005)**2)  # 10000 W/m² * (π * (5e-3 m)^2)
        assert abs(power_W - expected) < 1e-5, f"Expected {expected} W, got {power_W}"

    def test_create_grid(self):
        """Test la création d'une grille."""
        grid_x, grid_y = create_grid(10.0, 128)
        assert grid_x.shape == (128, 128), f"Expected shape (128, 128), got {grid_x.shape}"
        assert np.min(grid_x) == -5.0 and np.max(grid_x) == 5.0, "Grid x range incorrect"
        assert np.min(grid_y) == -5.0 and np.max(grid_y) == 5.0, "Grid y range incorrect"

    def test_compute_pv_rms(self):
        """Test le calcul du PV et RMS."""
        phase = np.array([[0, 1], [2, 3]])
        pv, rms = compute_pv_rms(phase)
        assert pv == 3.0, f"Expected PV=3.0, got {pv}"
        assert abs(rms - np.sqrt(3.5)) < 1e-5, f"Expected RMS={np.sqrt(3.5)}, got {rms}"


if __name__ == "__main__":
    import unittest
    unittest.main()
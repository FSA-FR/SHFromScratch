# -*- coding: utf-8 -*-
"""
MathAndPhysicsTools Module
--------------------------

FR: Module regroupant les fonctions mathématiques et physiques réutilisables pour le package SHFromScratch.
    Inclut les bases de modes (Zernike, Legendre), les normalisations, les conversions d'unités,
    et les outils de grille d'échantillonnage.

EN: Module grouping reusable mathematical and physical functions for the SHFromScratch package.
    Includes mode bases (Zernike, Legendre), normalizations, unit conversions,
    and sampling grid tools.

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch
License: MIT
"""

import numpy as np
import logging
import unittest
from typing import Tuple, Optional, Union

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================
# FR: Fonctions de bases de modes
# EN: Mode bases functions
# =============================================

def generate_zernike_modes(
    n_modes: int,
    ordination: str = "Noll",
    grid_x: Optional[np.ndarray] = None,
    grid_y: Optional[np.ndarray] = None,
    max_radial: Optional[int] = None,
    max_azimuthal: Optional[int] = None,
) -> np.ndarray:
    """
    FR: Génère une matrice de modes de Zernike pour une grille donnée.
        Les modes sont générés selon l'ordination spécifiée (Noll ou Wyant).
        
    EN: Generates a matrix of Zernike modes for a given grid.
        Modes are generated according to the specified indexing (Noll or Wyant).

    Args:
        n_modes (int): Nombre total de modes de Zernike à générer.
        ordination (str): Type d'ordination, "Noll" (défaut) ou "Wyant".
        grid_x (np.ndarray, optional): Grille en x (2D). Si None, une grille par défaut est créée.
        grid_y (np.ndarray, optional): Grille en y (2D). Si None, une grille par défaut est créée.
        max_radial (int, optional): Ordre radial maximal. Si None, calculé automatiquement à partir de n_modes.
        max_azimuthal (int, optional): Ordre azimutal maximal. Si None, calculé automatiquement.

    Returns:
        np.ndarray: Matrice de taille (n_modes, height, width) contenant les modes de Zernike.

    Raises:
        ValueError: Si n_modes <= 0 ou si ordination est invalide.

    Sources:
        - Noll, R. J. (1976). "Zernike polynomials and atmospheric turbulence." JOSA, 66(3), 207-211.
          DOI: 10.1364/JOSA.66.000207
        - Wyant, J. C., & Creath, K. (1992). "Basic wavefront aberration theory for optical metrology."
          Applied optics and Optical Engineering, 11, 2-53.
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
    r_max = np.max(np.sqrt(grid_x**2 + grid_y**2))
    r = r / r_max  # Normalisation à [0, 1]

    # Initialisation de la matrice des modes
    modes = np.zeros((n_modes, grid_x.shape[0], grid_x.shape[1]))

    # Génération des modes selon l'ordination
    for mode_idx in range(n_modes):
        if ordination == "Noll":
            n, m = noll_to_zernike_indices(mode_idx)
        else:  # Wyant
            n, m = wyant_to_zernike_indices(mode_idx)

        if n > max_radial or abs(m) > max_azimuthal:
            modes[mode_idx] = np.zeros_like(grid_x)
            continue

        # Calcul du polynôme radial
        R_nm = radial_polynomial(n, abs(m), r)

        # Calcul du mode de Zernike
        if m >= 0:
            modes[mode_idx] = R_nm * np.cos(abs(m) * theta)
        else:
            modes[mode_idx] = R_nm * np.sin(abs(m) * theta)

    return modes


def noll_to_zernike_indices(mode_idx: int) -> Tuple[int, int]:
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


def wyant_to_zernike_indices(mode_idx: int) -> Tuple[int, int]:
    """
    FR: Convertit un indice Wyant en indices (n, m) de Zernike.
        
    EN: Converts a Wyant index to Zernike (n, m) indices.

    Args:
        mode_idx (int): Indice du mode selon l'ordination de Wyant.

    Returns:
        Tuple[int, int]: (n, m) où n est l'ordre radial et m l'ordre azimutal.

    Sources:
        - Wyant, J. C., & Creath, K. (1992). "Basic wavefront aberration theory for optical metrology."
          Applied optics and Optical Engineering, 11, 2-53.
    """
    n = int(np.floor(np.sqrt(2 * mode_idx + 1)))
    m = 2 * mode_idx - n * (n + 1)
    return n, m


def radial_polynomial(n: int, m: int, r: np.ndarray) -> np.ndarray:
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
    x_norm = 2 * grid_x / (np.max(grid_x) - np.min(grid_x)) - 1
    y_norm = 2 * grid_y / (np.max(grid_y) - np.min(grid_y)) - 1

    # Initialisation de la matrice des modes
    modes = np.zeros((n_modes, grid_x.shape[0], grid_x.shape[1]))

    # Génération des modes de Legendre 2D
    for mode_idx in range(n_modes):
        n = mode_idx  # Ordre du polynôme de Legendre
        P_n = np.polynomial.legendre.legval(x_norm.flatten(), [0] * n + [1]).reshape(x_norm.shape)
        P_m = np.polynomial.legendre.legval(y_norm.flatten(), [0] * n + [1]).reshape(y_norm.shape)
        modes[mode_idx] = P_n * P_m

    return modes


# =============================================
# FR: Fonctions de normalisation
# EN: Normalization functions
# =============================================

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


# =============================================
# FR: Fonctions de conversion d'unités
# EN: Unit conversion functions
# =============================================

def nm_to_rad(phase_nm: np.ndarray, wavelength_nm: float) -> np.ndarray:
    """
    FR: Convertit une phase en nm en rad.
        
    EN: Converts a phase from nm to rad.

    Args:
        phase_nm (np.ndarray): Phase en nm.
        wavelength_nm (float): Longueur d'onde en nm.

    Returns:
        np.ndarray: Phase en rad.

    Formula:
        phase_rad = (2π * phase_nm) / wavelength_nm
    """
    return (2 * np.pi * phase_nm) / wavelength_nm


def rad_to_nm(phase_rad: np.ndarray, wavelength_nm: float) -> np.ndarray:
    """
    FR: Convertit une phase en rad en nm.
        
    EN: Converts a phase from rad to nm.

    Args:
        phase_rad (np.ndarray): Phase en rad.
        wavelength_nm (float): Longueur d'onde en nm.

    Returns:
        np.ndarray: Phase en nm.

    Formula:
        phase_nm = (phase_rad * wavelength_nm) / (2π)
    """
    return (phase_rad * wavelength_nm) / (2 * np.pi)


def nm_to_lambda(phase_nm: np.ndarray, wavelength_nm: float) -> np.ndarray:
    """
    FR: Convertit une phase en nm en λ (longueur d'onde).
        
    EN: Converts a phase from nm to λ (wavelength).

    Args:
        phase_nm (np.ndarray): Phase en nm.
        wavelength_nm (float): Longueur d'onde en nm.

    Returns:
        np.ndarray: Phase en λ.
    """
    return phase_nm / wavelength_nm


def lambda_to_nm(phase_lambda: np.ndarray, wavelength_nm: float) -> np.ndarray:
    """
    FR: Convertit une phase en λ en nm.
        
    EN: Converts a phase from λ to nm.

    Args:
        phase_lambda (np.ndarray): Phase en λ.
        wavelength_nm (float): Longueur d'onde en nm.

    Returns:
        np.ndarray: Phase en nm.
    """
    return phase_lambda * wavelength_nm


def rad_to_mrad(phase_rad: np.ndarray) -> np.ndarray:
    """
    FR: Convertit une phase en rad en mrad.
        
    EN: Converts a phase from rad to mrad.

    Args:
        phase_rad (np.ndarray): Phase en rad.

    Returns:
        np.ndarray: Phase en mrad.
    """
    return phase_rad * 1e3


def mrad_to_rad(phase_mrad: np.ndarray) -> np.ndarray:
    """
    FR: Convertit une phase en mrad en rad.
        
    EN: Converts a phase from mrad to rad.

    Args:
        phase_mrad (np.ndarray): Phase en mrad.

    Returns:
        np.ndarray: Phase en rad.
    """
    return phase_mrad * 1e-3


# =============================================
# FR: Fonctions de grille d'échantillonnage
# EN: Sampling grid functions
# =============================================

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


# =============================================
# FR: Tests unitaires
# EN: Unit tests
# =============================================

class TestMathAndPhysicsTools(unittest.TestCase):
    """FR: Tests unitaires pour les fonctions de MathAndPhysicsTools."""

    def setUp(self):
        self.wavelength_nm = 633.0
        self.size_mm = 10.0
        self.num_points = 512

    def test_create_grid(self):
        grid_x, grid_y = create_grid(self.size_mm, self.num_points)
        self.assertEqual(grid_x.shape, (self.num_points, self.num_points))
        self.assertEqual(grid_y.shape, (self.num_points, self.num_points))
        self.assertAlmostEqual(np.min(grid_x), -self.size_mm / 2, places=5)
        self.assertAlmostEqual(np.max(grid_x), self.size_mm / 2, places=5)

    def test_nm_to_rad(self):
        phase_nm = np.array([633.0, 1266.0])  # 1λ et 2λ
        phase_rad = nm_to_rad(phase_nm, self.wavelength_nm)
        np.testing.assert_allclose(phase_rad, [2 * np.pi, 4 * np.pi], rtol=1e-5)

    def test_rad_to_nm(self):
        phase_rad = np.array([2 * np.pi, 4 * np.pi])
        phase_nm = rad_to_nm(phase_rad, self.wavelength_nm)
        np.testing.assert_allclose(phase_nm, [633.0, 1266.0], rtol=1e-5)

    def test_normalize_phase_RMS(self):
        phase = np.random.rand(100, 100) * 100  # Phase aléatoire en nm
        normalized = normalize_phase(phase, normalization="RMS", target_value=1.0, wavelength_nm=self.wavelength_nm)
        rms = np.sqrt(np.mean(normalized**2))
        self.assertAlmostEqual(rms, self.wavelength_nm, places=5)

    def test_normalize_phase_PV(self):
        phase = np.random.rand(100, 100) * 100  # Phase aléatoire en nm
        normalized = normalize_phase(phase, normalization="PV", target_value=1.0, wavelength_nm=self.wavelength_nm)
        pv = np.max(normalized) - np.min(normalized)
        self.assertAlmostEqual(pv, self.wavelength_nm, places=5)

    def test_noll_to_zernike_indices(self):
        # Test pour les premiers indices Noll
        self.assertEqual(noll_to_zernike_indices(0), (0, 0))  # Piston
        self.assertEqual(noll_to_zernike_indices(1), (1, -1))  # Tilt X
        self.assertEqual(noll_to_zernike_indices(2), (1, 1))   # Tilt Y
        self.assertEqual(noll_to_zernike_indices(3), (2, 0))   # Defocus

    def test_wyant_to_zernike_indices(self):
        # Test pour les premiers indices Wyant
        self.assertEqual(wyant_to_zernike_indices(0), (0, 0))  # Piston
        self.assertEqual(wyant_to_zernike_indices(1), (1, 1))   # Tilt Y
        self.assertEqual(wyant_to_zernike_indices(2), (1, -1))  # Tilt X
        self.assertEqual(wyant_to_zernike_indices(3), (2, 0))   # Defocus


if __name__ == "__main__":
    unittest.main()

"""
Southwell.py

FR: Module pour les algorithmes de reconstruction de Southwell.
    
    Ce module implémente les méthodes de reconstruction de front d'onde
    basées sur l'algorithme de Southwell, qui est une méthode modale
    pour reconstruire le front d'onde à partir des pentes locales.
    
    Unités :
    - Longueurs : mm
    - Longueur d'onde : nm
    - Phase : nm (principale), rad (pour les calculs)
    - Pentes : rad (radians)

EN: Module for Southwell reconstruction algorithms.
    
    This module implements wavefront reconstruction methods based on
    the Southwell algorithm, which is a modal method for reconstructing
    the wavefront from local slopes.
    
    Units:
    - Lengths: mm
    - Wavelength: nm
    - Phase: nm (main), rad (for calculations)
    - Slopes: rad (radians)

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch

Dependencies:
    - numpy
    - MathAndPhysicsTools (pour les polynômes de Zernike)

Sources:
    - "Wavefront Estimation from Wavefront Slope Measurements" by W. H. Southwell (1980)
    - "Principles of Adaptive Optics" by R. K. Tyson (1991), Ch. 6
"""

import numpy as np
import logging
from typing import Optional, Tuple
from enum import Enum

from MathAndPhysicsTools import (
    handle_nan,
    generate_zernike_modes,
    ZernikeOrdering,
    NormalizationType,
    DEFAULT_WAVELENGTH_NM,
    TWO_PI
)


# =============================================================================
# CONFIGURATION GLOBALE
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Southwell")

MAX_ITERATIONS = 100
CONVERGENCE_TOLERANCE = 1e-6
DEFAULT_MAX_ZERNIKE_DEGREE = 10


# =============================================================================
# ENUMS
# =============================================================================

class ReconstructionType(Enum):
    STANDARD = "standard"
    WEIGHTED = "weighted"


# =============================================================================
# CLASSE: SOUTHWELL RECONSTRUCTOR
# =============================================================================

class SouthwellReconstructor:
    """
    FR: Reconstruteur de front d'onde basé sur l'algorithme de Southwell.
    EN: Wavefront reconstructor based on the Southwell algorithm.
    
    Sources: "Wavefront Estimation from Wavefront Slope Measurements" by W. H. Southwell (1980)
    """

    def __init__(self,
                 wavelength_nm: float = DEFAULT_WAVELENGTH_NM,
                 max_zernike_degree: int = DEFAULT_MAX_ZERNIKE_DEGREE,
                 num_pixels: int = 512,
                 diameter_mm: float = 10.0,
                 reconstruction_type: ReconstructionType = ReconstructionType.STANDARD,
                 regularization_factor: float = 0.0):
        self.wavelength_nm = wavelength_nm
        self.max_zernike_degree = max_zernike_degree
        self.num_pixels = num_pixels
        self.diameter_mm = diameter_mm
        self.reconstruction_type = reconstruction_type
        self.regularization_factor = regularization_factor
        
        self.zernike_modes = generate_zernike_modes(
            max_degree=max_zernike_degree,
            ordering=ZernikeOrdering.NOLL,
            normalization=NormalizationType.NOLL,
            num_points=num_pixels
        )
        self.X_norm, self.Y_norm = self._create_normalized_grid()

    def _create_normalized_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """FR: Crée une grille normalisée. EN: Creates a normalized grid."""
        x = np.linspace(-1, 1, self.num_pixels)
        y = np.linspace(-1, 1, self.num_pixels)
        X, Y = np.meshgrid(x, y)
        return X, Y

    def reconstruct(self, slopes_x: np.ndarray, slopes_y: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        FR: Reconstrue le front d'onde à partir des pentes.
        EN: Reconstructs the wavefront from slopes.
        
        Sources: "Wavefront Estimation from Wavefront Slope Measurements" by W. H. Southwell (1980)
        """
        slopes_x_flat = slopes_x.ravel()
        slopes_y_flat = slopes_y.ravel()
        weights_flat = weights.ravel() if weights is not None else np.ones_like(slopes_x_flat)
        
        coefficients = self._calculate_zernike_coefficients(slopes_x_flat, slopes_y_flat, weights_flat)
        return self._reconstruct_from_coefficients(coefficients)

    def _calculate_zernike_coefficients(self, slopes_x: np.ndarray, slopes_y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """FR: Calcule les coefficients de Zernike. EN: Calculates Zernike coefficients."""
        num_modes = len(self.zernike_modes)
        num_measurements = len(slopes_x)
        
        A = np.zeros((num_measurements * 2, num_modes))
        b = np.concatenate([slopes_x, slopes_y])
        
        for j, (n, m, Z) in enumerate(self.zernike_modes):
            dZ_dx = np.gradient(Z, axis=1).ravel()[:num_measurements]
            dZ_dy = np.gradient(Z, axis=0).ravel()[:num_measurements]
            dZ_dx_rad = dZ_dx * (TWO_PI / self.wavelength_nm)
            dZ_dy_rad = dZ_dy * (TWO_PI / self.wavelength_nm)
            A[:num_measurements, j] = dZ_dx_rad
            A[num_measurements:, j] = dZ_dy_rad
        
        W = np.diag(np.concatenate([weights, weights]))
        A_weighted = W @ A
        b_weighted = W @ b
        
        if self.regularization_factor > 0:
            A_weighted = np.vstack([A_weighted, np.sqrt(self.regularization_factor) * np.eye(num_modes)])
            b_weighted = np.concatenate([b_weighted, np.zeros(num_modes)])
        
        coefficients, _, _, _ = np.linalg.lstsq(A_weighted, b_weighted, rcond=None)
        return coefficients

    def _reconstruct_from_coefficients(self, coefficients: np.ndarray) -> np.ndarray:
        """FR: Reconstruit à partir des coefficients. EN: Reconstructs from coefficients."""
        wavefront = np.zeros((self.num_pixels, self.num_pixels))
        for i, (n, m, Z) in enumerate(self.zernike_modes):
            if i < len(coefficients):
                wavefront += coefficients[i] * Z
        return handle_nan(wavefront * (self.wavelength_nm / TWO_PI), method='zero')

    def get_zernike_coefficients(self, wavefront: np.ndarray) -> np.ndarray:
        """FR: Calcule les coefficients de Zernike. EN: Calculates Zernike coefficients."""
        wavefront_rad = wavefront * (TWO_PI / self.wavelength_nm)
        coefficients = np.zeros(len(self.zernike_modes))
        for i, (n, m, Z) in enumerate(self.zernike_modes):
            numerator = np.sum(wavefront_rad * Z)
            denominator = np.sum(Z**2)
            coefficients[i] = numerator / denominator if denominator > 0 else 0.0
        return coefficients


# =============================================================================
# FONCTION UTILITAIRE
# =============================================================================

def reconstruct_wavefront_southwell(slopes_x: np.ndarray, slopes_y: np.ndarray,
                                     wavelength_nm: float = DEFAULT_WAVELENGTH_NM,
                                     max_zernike_degree: int = DEFAULT_MAX_ZERNIKE_DEGREE,
                                     num_pixels: int = 512,
                                     diameter_mm: float = 10.0) -> np.ndarray:
    """FR: Fonction utilitaire pour reconstruire un front d'onde. EN: Utility function to reconstruct a wavefront."""
    reconstructor = SouthwellReconstructor(
        wavelength_nm=wavelength_nm, max_zernike_degree=max_zernike_degree,
        num_pixels=num_pixels, diameter_mm=diameter_mm
    )
    return reconstructor.reconstruct(slopes_x, slopes_y)


# =============================================================================
# TESTS UNITAIRES
# =============================================================================

class TestSouthwell:
    """FR: Tests unitaires pour Southwell.py."""

    def test_creation(self):
        """FR: Test la création."""
        reconstructor = SouthwellReconstructor(wavelength_nm=633.0, max_zernike_degree=5)
        assert reconstructor.wavelength_nm == 633.0

    def test_reconstruct(self):
        """FR: Test la reconstruction."""
        reconstructor = SouthwellReconstructor(wavelength_nm=633.0, max_zernike_degree=5, num_pixels=64)
        slopes_x = np.zeros(64)
        slopes_y = np.zeros(64)
        wavefront = reconstructor.reconstruct(slopes_x, slopes_y)
        assert wavefront.shape == (64, 64)


if __name__ == "__main__":
    import unittest
    unittest.main()

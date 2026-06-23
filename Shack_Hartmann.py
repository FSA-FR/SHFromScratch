"""
Shack_Hartmann.py

FR: Module pour le capteur Shack-Hartmann.
    
    Ce module implémente un capteur Shack-Hartmann pour la mesure de front d'onde.
    
    Fonctionnalités principales :
    - Création d'un capteur Shack-Hartmann avec matrice de microlentilles
    - Simulation de la formation des tâches d'Airy
    - Détection des centroïdes des tâches
    - Calcul des pentes locales
    - Reconstruction du front d'onde (modale et zonale)
    
    Unités :
    - Longueurs : mm (pour les dimensions physiques)
    - Longueur d'onde : nm
    - Phase : nm (principale), rad (pour les calculs)
    - Pentes : rad (radians)
    - Positions : mm

EN: Module for the Shack-Hartmann sensor.
    
    This module implements a Shack-Hartmann sensor for wavefront measurement.
    
    Main features:
    - Creation of a Shack-Hartmann sensor with microlens array
    - Simulation of Airy spot formation
    - Spot centroid detection
    - Local slope calculation
    - Wavefront reconstruction (modal and zonal)
    
    Units:
    - Lengths: mm (for physical dimensions)
    - Wavelength: nm
    - Phase: nm (main), rad (for calculations)
    - Slopes: rad (radians)
    - Positions: mm

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch

Dependencies:
    - numpy
    - scipy (pour les fonctions d'optimisation, optionnel)
    - Microstructure (pour MicrolensArray)
    - Camera (pour PerfectCamera)
    - MathAndPhysicsTools (pour les fonctions outils)

Sources:
    - "Principles of Adaptive Optics" by R. K. Tyson (1991)
    - "Shack-Hartmann Wavefront Sensing" by J. W. Hardy (1978)
"""

import numpy as np
import logging
from typing import Optional, Tuple, List
from enum import Enum
from dataclasses import dataclass

from Microstructure import MicrolensArray, create_microlens_array, MicrolensShape, ArrayType
from Camera import PerfectCamera, create_perfect_camera
from MathAndPhysicsTools import (
    handle_nan,
    generate_zernike_polynomial,
    generate_zernike_modes,
    ZernikeOrdering,
    NormalizationType,
    DEFAULT_WAVELENGTH_NM,
    PI,
    TWO_PI
)


# =============================================================================
# CONFIGURATION GLOBALE
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Shack_Hartmann")


# =============================================================================
# CONSTANTES
# =============================================================================

TYPICAL_SH_DISTANCE_MM = 10.0
TYPICAL_NUM_MICROLENS_X = 50
TYPICAL_NUM_MICROLENS_Y = 50
TYPICAL_MICROLENS_DIAMETER_MM = 0.1
TYPICAL_MICROLENS_FOCAL_LENGTH_MM = 10.0


# =============================================================================
# ENUMS
# =============================================================================

class ReconstructionMethod(Enum):
    ZONAL = "zonal"
    MODAL = "modal"
    BOTH = "both"


class CentroidMethod(Enum):
    CENTER_OF_MASS = "center_of_mass"
    GAUSSIAN_FIT = "gaussian_fit"
    CORRELATION = "correlation"


# =============================================================================
# CLASSE: SPOT
# =============================================================================

@dataclass
class Spot:
    """FR: Tâche d'Airy détectée. EN: Detected Airy spot."""
    x_pixel: float = 0.0
    y_pixel: float = 0.0
    x_mm: float = 0.0
    y_mm: float = 0.0
    intensity: float = 0.0
    width_pixels: float = 1.0
    signal_to_noise: float = 0.0


# =============================================================================
# CLASSE: SHACK-HARTMANN SENSOR
# =============================================================================

class ShackHartmann:
    """
    FR: Capteur Shack-Hartmann.
    EN: Shack-Hartmann sensor.
    
    Sources:
        - "Principles of Adaptive Optics" by R. K. Tyson (1991)
        - "Shack-Hartmann Wavefront Sensing" by J. W. Hardy (1978)
    """

    def __init__(self,
                 name: str = "ShackHartmann",
                 num_microlens_x: int = TYPICAL_NUM_MICROLENS_X,
                 num_microlens_y: int = TYPICAL_NUM_MICROLENS_Y,
                 microlens_diameter_mm: float = TYPICAL_MICROLENS_DIAMETER_MM,
                 microlens_focal_length_mm: float = TYPICAL_MICROLENS_FOCAL_LENGTH_MM,
                 sh_distance_mm: float = TYPICAL_SH_DISTANCE_MM,
                 wavelength_nm: float = DEFAULT_WAVELENGTH_NM,
                 num_pixels_x: int = 1024,
                 num_pixels_y: int = 1024,
                 pixel_size_um: float = 5.0):
        self.name = name
        self.wavelength_nm = wavelength_nm
        self.sh_distance_mm = sh_distance_mm
        self.num_pixels_x = num_pixels_x
        self.num_pixels_y = num_pixels_y
        self.pixel_size_um = pixel_size_um
        
        self.microlens_array = create_microlens_array(
            name=f"{name}_MLArray",
            num_elements_x=num_microlens_x,
            num_elements_y=num_microlens_y,
            element_diameter_mm=microlens_diameter_mm,
            pitch_mm=microlens_diameter_mm,
            focal_length_mm=microlens_focal_length_mm,
            wavelength_nm=wavelength_nm
        )
        
        self.camera = create_perfect_camera(
            name=f"{name}_Camera",
            num_pixels_x=num_pixels_x,
            num_pixels_y=num_pixels_y,
            pixel_size_um=pixel_size_um,
            wavelength_nm=wavelength_nm
        )
        
        self.field_of_view_mm = (
            self.microlens_array.total_width_mm,
            self.microlens_array.total_height_mm
        )
        self.spot_size_pixels = self._calculate_spot_size_pixels()

    def _calculate_spot_size_pixels(self) -> float:
        """FR: Calcule la taille d'une tâche en pixels. EN: Calculates spot size in pixels."""
        D_mm = self.microlens_array.element_diameter_mm
        f_mm = self.microlens_array.focal_length_mm
        d_mm = self.sh_distance_mm - f_mm
        
        if abs(d_mm) < 1e-6:
            spot_size_mm = 2.44 * (self.wavelength_nm * 1e-6) * f_mm / D_mm
        else:
            spot_size_mm = D_mm + 2.44 * (self.wavelength_nm * 1e-6) * abs(d_mm) / D_mm
        
        pixel_size_mm = self.pixel_size_um * 1e-3
        return spot_size_mm / pixel_size_mm

    def _create_coordinate_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """FR: Crée une grille de coordonnées. EN: Creates a coordinate grid."""
        x = np.linspace(
            -self.microlens_array.total_width_mm / 2,
            self.microlens_array.total_width_mm / 2,
            self.num_pixels_x
        )
        y = np.linspace(
            -self.microlens_array.total_height_mm / 2,
            self.microlens_array.total_height_mm / 2,
            self.num_pixels_y
        )
        X, Y = np.meshgrid(x, y)
        return X, Y

    def simulate_spot_image(self, input_phase_map: np.ndarray) -> np.ndarray:
        """
        FR: Simule une image des tâches d'Airy.
        EN: Simulates an Airy spot image.
        
        Sources:
            - "Principles of Adaptive Optics" by R. K. Tyson (1991)
        """
        total_phase_map = self._calculate_total_phase_map(input_phase_map)
        spot_positions_mm, spot_intensities = self._calculate_spot_positions(total_phase_map)
        spot_image = self.camera.simulate_airy_spots(
            spot_positions=spot_positions_mm,
            spot_intensities=spot_intensities,
            spot_width_um=self.spot_size_pixels * self.pixel_size_um
        )
        return spot_image

    def _calculate_total_phase_map(self, input_phase_map: np.ndarray) -> np.ndarray:
        """FR: Calcule la phase totale. EN: Calculates total phase."""
        X, Y = self._create_coordinate_grid()
        ml_phase_map = self.microlens_array.get_total_phase_map(X, Y)
        return handle_nan(input_phase_map + ml_phase_map, method='zero')

    def _calculate_spot_positions(self, total_phase_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """FR: Calcule les positions des tâches. EN: Calculates spot positions."""
        ml_positions = self.microlens_array.get_microlens_positions()
        spot_positions_mm = np.zeros_like(ml_positions)
        spot_intensities = np.ones(len(ml_positions))
        
        X, Y = self._create_coordinate_grid()
        slopes_x, slopes_y = self._calculate_local_slopes(total_phase_map, X, Y)
        
        f_mm = self.microlens_array.focal_length_mm
        wavelength_mm = self.wavelength_nm * 1e-6
        
        for i, (x_ml, y_ml) in enumerate(ml_positions):
            x_idx = int(x_ml / self.microlens_array.pitch_mm)
            y_idx = int(y_ml / self.microlens_array.pitch_mm)
            
            if x_idx < slopes_x.shape[1] and y_idx < slopes_x.shape[0]:
                slope_x = slopes_x[y_idx, x_idx]
                slope_y = slopes_y[y_idx, x_idx]
                
                delta_x_mm = f_mm * slope_x
                delta_y_mm = f_mm * slope_y
                
                spot_positions_mm[i, 0] = x_ml + delta_x_mm
                spot_positions_mm[i, 1] = y_ml + delta_y_mm
        
        return spot_positions_mm, spot_intensities

    def _calculate_local_slopes(self, phase_map: np.ndarray, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """FR: Calcule les pentes locales. EN: Calculates local slopes."""
        wavelength_mm = self.wavelength_nm * 1e-6
        phase_rad = phase_map * (TWO_PI / self.wavelength_nm)
        
        dx = X[0, 1] - X[0, 0]
        dy = Y[1, 0] - Y[0, 0]
        
        slopes_x = np.gradient(phase_rad, dx, axis=1)
        slopes_y = np.gradient(phase_rad, dy, axis=0)
        
        return slopes_x, slopes_y

    def detect_spot_centroids(self, spot_image: np.ndarray, method: CentroidMethod = CentroidMethod.CENTER_OF_MASS) -> List[Spot]:
        """
        FR: Détecte les centroïdes des tâches.
        EN: Detects spot centroids.
        
        Sources:
            - "Adaptive Optics for Astronomical Telescopes" by F. Roddier (1999)
        """
        spots = []
        spot_image_normalized = spot_image / np.max(spot_image)
        ml_positions = self.microlens_array.get_microlens_positions()
        
        for i, (x_ml_mm, y_ml_mm) in enumerate(ml_positions):
            x_ml_pixel = int(x_ml_mm / (self.pixel_size_um * 1e-3))
            y_ml_pixel = int(y_ml_mm / (self.pixel_size_um * 1e-3))
            
            half_size = int(self.spot_size_pixels / 2)
            x_min = max(0, x_ml_pixel - half_size)
            x_max = min(self.num_pixels_x, x_ml_pixel + half_size + 1)
            y_min = max(0, y_ml_pixel - half_size)
            y_max = min(self.num_pixels_y, y_ml_pixel + half_size + 1)
            
            region = spot_image_normalized[y_min:y_max, x_min:x_max]
            
            if method == CentroidMethod.CENTER_OF_MASS:
                x_centroid, y_centroid = self._centroid_center_of_mass(region, x_min, y_min)
            elif method == CentroidMethod.GAUSSIAN_FIT:
                x_centroid, y_centroid = self._centroid_gaussian_fit(region, x_min, y_min)
            else:
                x_centroid, y_centroid = self._centroid_center_of_mass(region, x_min, y_min)
            
            max_intensity = float(np.max(region))
            width_pixels = self._calculate_spot_width(region)
            snr = self._calculate_snr(region)
            
            x_centroid_mm = x_centroid * (self.pixel_size_um * 1e-3)
            y_centroid_mm = y_centroid * (self.pixel_size_um * 1e-3)
            
            spots.append(Spot(x_pixel=x_centroid, y_pixel=y_centroid,
                             x_mm=x_centroid_mm, y_mm=y_centroid_mm,
                             intensity=max_intensity, width_pixels=width_pixels,
                             signal_to_noise=snr))
        
        return spots

    def _centroid_center_of_mass(self, region: np.ndarray, x_offset: int, y_offset: int) -> Tuple[float, float]:
        """FR: Centre de masse. EN: Center of mass."""
        thresholded = np.where(region > np.mean(region), region, 0.0)
        total = np.sum(thresholded)
        if total <= 0:
            return region.shape[1] / 2 + x_offset, region.shape[0] / 2 + y_offset
        y_indices, x_indices = np.indices(region.shape)
        return np.sum(x_indices * thresholded) / total + x_offset, np.sum(y_indices * thresholded) / total + y_offset

    def _centroid_gaussian_fit(self, region: np.ndarray, x_offset: int, y_offset: int) -> Tuple[float, float]:
        """FR: Ajustement gaussien. EN: Gaussian fit."""
        try:
            from scipy.optimize import curve_fit
            
            def gaussian_2d(x, y, x0, y0, sigma, A, offset):
                return offset + A * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
            
            y_indices, x_indices = np.indices(region.shape)
            x_data = x_indices.ravel()
            y_data = y_indices.ravel()
            z_data = region.ravel()
            
            x0_init = region.shape[1] / 2
            y0_init = region.shape[0] / 2
            sigma_init = self.spot_size_pixels / 2
            A_init = np.max(region)
            offset_init = np.min(region)
            
            try:
                popt, _ = curve_fit(gaussian_2d, (x_data, y_data), z_data,
                                   p0=[x0_init, y0_init, sigma_init, A_init, offset_init])
                return popt[0] + x_offset, popt[1] + y_offset
            except:
                return self._centroid_center_of_mass(region, x_offset, y_offset)
        except ImportError:
            return self._centroid_center_of_mass(region, x_offset, y_offset)

    def _calculate_spot_width(self, region: np.ndarray) -> float:
        """FR: Calcule la largeur de la tâche. EN: Calculates spot width."""
        max_val = np.max(region)
        half_max = max_val / 2
        above_half = region > half_max
        
        if np.any(above_half):
            y_indices, x_indices = np.where(above_half)
            width_x = np.max(x_indices) - np.min(x_indices)
            width_y = np.max(y_indices) - np.min(y_indices)
            return float(np.sqrt(width_x**2 + width_y**2))
        return self.spot_size_pixels

    def _calculate_snr(self, region: np.ndarray) -> float:
        """FR: Calcule le rapport signal/bruit. EN: Calculates SNR."""
        signal = np.max(region)
        background = region[region < 0.5 * np.max(region)]
        noise = np.std(background) if len(background) > 0 else 1.0
        return signal / noise if noise > 0 else float('inf')

    def calculate_slopes(self, spots: List[Spot]) -> Tuple[np.ndarray, np.ndarray]:
        """
        FR: Calcule les pentes locales.
        EN: Calculates local slopes.
        
        Sources:
            - "Principles of Adaptive Optics" by R. K. Tyson (1991)
        """
        ml_positions = self.microlens_array.get_microlens_positions()
        slopes_x = np.zeros(len(spots))
        slopes_y = np.zeros(len(spots))
        f_mm = self.microlens_array.focal_length_mm
        
        for i, (spot, ml_pos) in enumerate(zip(spots, ml_positions)):
            delta_x_mm = spot.x_mm - ml_pos[0]
            delta_y_mm = spot.y_mm - ml_pos[1]
            slopes_x[i] = delta_x_mm / f_mm
            slopes_y[i] = delta_y_mm / f_mm
        
        return slopes_x, slopes_y

    def reconstruct_wavefront(self, slopes_x: np.ndarray, slopes_y: np.ndarray,
                              method: ReconstructionMethod = ReconstructionMethod.MODAL,
                              max_zernike_degree: int = 10) -> np.ndarray:
        """
        FR: Reconstrue le front d'onde.
        EN: Reconstructs the wavefront.
        
        Sources:
            - "Principles of Adaptive Optics" by R. K. Tyson (1991)
        """
        if method == ReconstructionMethod.ZONAL:
            return self._reconstruct_zonal(slopes_x, slopes_y)
        elif method == ReconstructionMethod.MODAL:
            return self._reconstruct_modal(slopes_x, slopes_y, max_zernike_degree)
        else:
            return (self._reconstruct_zonal(slopes_x, slopes_y) + 
                    self._reconstruct_modal(slopes_x, slopes_y, max_zernike_degree)) / 2

    def _reconstruct_zonal(self, slopes_x: np.ndarray, slopes_y: np.ndarray) -> np.ndarray:
        """FR: Reconstruction zonale. EN: Zonal reconstruction."""
        X, Y = self._create_coordinate_grid()
        wavefront = np.zeros_like(X)
        ml_positions = self.microlens_array.get_microlens_positions()
        
        for i, (x_ml, y_ml) in enumerate(ml_positions):
            x_idx = int(x_ml / self.microlens_array.pitch_mm)
            y_idx = int(y_ml / self.microlens_array.pitch_mm)
            wavefront[y_idx, :] = np.cumsum(slopes_x[i] * self.microlens_array.pitch_mm)
            wavefront[:, x_idx] = np.cumsum(slopes_y[i] * self.microlens_array.pitch_mm)
        
        return handle_nan(wavefront * (self.wavelength_nm / TWO_PI), method='zero')

    def _reconstruct_modal(self, slopes_x: np.ndarray, slopes_y: np.ndarray, max_zernike_degree: int) -> np.ndarray:
        """FR: Reconstruction modale. EN: Modal reconstruction."""
        X, Y = self._create_coordinate_grid()
        X_norm = X / (self.microlens_array.total_width_mm / 2)
        Y_norm = Y / (self.microlens_array.total_height_mm / 2)
        
        zernike_modes = generate_zernike_modes(
            max_degree=max_zernike_degree,
            ordering=ZernikeOrdering.NOLL,
            normalization=NormalizationType.NOLL,
            num_points=self.num_pixels_x
        )
        
        num_modes = len(zernike_modes)
        coefficients = np.zeros(num_modes)
        ml_positions = self.microlens_array.get_microlens_positions()
        
        for i, (n, m, Z) in enumerate(zernike_modes):
            coefficient = 0.0
            for j, (x_ml, y_ml) in enumerate(ml_positions):
                x_idx = int(x_ml / self.microlens_array.pitch_mm)
                y_idx = int(y_ml / self.microlens_array.pitch_mm)
                if x_idx < Z.shape[1] and y_idx < Z.shape[0]:
                    dZ_dx = np.gradient(Z, axis=1)[y_idx, x_idx]
                    dZ_dy = np.gradient(Z, axis=0)[y_idx, x_idx]
                    coefficient += slopes_x[j] * dZ_dx + slopes_y[j] * dZ_dy
            coefficients[i] = coefficient
        
        wavefront = np.zeros_like(X_norm)
        for i, (n, m, Z) in enumerate(zernike_modes):
            wavefront += coefficients[i] * Z
        
        return handle_nan(wavefront * (self.wavelength_nm / TWO_PI), method='zero')


# =============================================================================
# FONCTION DE CRÉATION
# =============================================================================

def create_shack_hartmann(name: str = "ShackHartmann",
                          num_microlens_x: int = TYPICAL_NUM_MICROLENS_X,
                          num_microlens_y: int = TYPICAL_NUM_MICROLENS_Y,
                          microlens_diameter_mm: float = TYPICAL_MICROLENS_DIAMETER_MM,
                          microlens_focal_length_mm: float = TYPICAL_MICROLENS_FOCAL_LENGTH_MM,
                          sh_distance_mm: float = TYPICAL_SH_DISTANCE_MM,
                          wavelength_nm: float = DEFAULT_WAVELENGTH_NM,
                          num_pixels_x: int = 1024,
                          num_pixels_y: int = 1024,
                          pixel_size_um: float = 5.0) -> ShackHartmann:
    """FR: Crée un capteur Shack-Hartmann. EN: Creates a Shack-Hartmann sensor."""
    return ShackHartmann(
        name=name, num_microlens_x=num_microlens_x, num_microlens_y=num_microlens_y,
        microlens_diameter_mm=microlens_diameter_mm,
        microlens_focal_length_mm=microlens_focal_length_mm,
        sh_distance_mm=sh_distance_mm, wavelength_nm=wavelength_nm,
        num_pixels_x=num_pixels_x, num_pixels_y=num_pixels_y,
        pixel_size_um=pixel_size_um
    )


# =============================================================================
# TESTS UNITAIRES
# =============================================================================

class TestShackHartmann:
    """FR: Tests unitaires pour Shack_Hartmann.py."""

    def test_creation(self):
        """FR: Test la création."""
        sh = create_shack_hartmann(name="TestSH", num_microlens_x=5, num_microlens_y=5)
        assert sh.name == "TestSH"

    def test_spot_image(self):
        """FR: Test la simulation d'image de tâches."""
        sh = create_shack_hartmann(num_microlens_x=5, num_microlens_y=5)
        phase_map = np.zeros((sh.num_pixels_y, sh.num_pixels_x))
        spot_image = sh.simulate_spot_image(phase_map)
        assert spot_image.shape == (sh.num_pixels_y, sh.num_pixels_x)

    def test_detect_centroids(self):
        """FR: Test la détection des centroïdes."""
        sh = create_shack_hartmann(num_microlens_x=5, num_microlens_y=5)
        spot_image = np.random.rand(sh.num_pixels_y, sh.num_pixels_x)
        spots = sh.detect_spot_centroids(spot_image)
        assert len(spots) == 25


if __name__ == "__main__":
    import unittest
    unittest.main()

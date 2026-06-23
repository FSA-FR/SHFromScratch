"""
Camera.py

FR: Module pour la modélisation de capteurs optiques (caméras).
    Ce module permet de créer et simuler des caméras parfaites et réelles.
    
    Fonctionnalités principales :
    - Caméras parfaites (sans bruit)
    - Caméras réelles (CCD, CMOS, etc.)
    - Modélisation du bruit (quantique, thermique, de lecture)
    - Réponse spectrale
    - Simulation de tâches d'Airy
    
    Unités :
    - Longueurs : mm (pour les dimensions physiques)
    - Longueur d'onde : nm
    - Taille des pixels : µm (micromètres)
    - Intensité : ADU (Analog-to-Digital Units) ou électrons
    - Bruit : électrons RMS

EN: Module for modeling optical sensors (cameras).
    This module allows creating and simulating perfect and real cameras.
    
    Main features:
    - Perfect cameras (no noise)
    - Real cameras (CCD, CMOS, etc.)
    - Noise modeling (quantum, thermal, readout)
    - Spectral response
    - Airy spot simulation
    
    Units:
    - Lengths: mm (for physical dimensions)
    - Wavelength: nm
    - Pixel size: µm (micrometers)
    - Intensity: ADU (Analog-to-Digital Units) or electrons
    - Noise: electrons RMS

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch

Dependencies:
    - numpy
    - MathAndPhysicsTools (pour les fonctions outils)

Sources:
    - "Charge-Coupled Devices" by J. Janesick (2001)
    - "CMOS/CCD Sensors and Camera Systems" by G. C. Holst & T. L. Lomheim (2011)
"""

import numpy as np
import logging
from typing import Optional, Tuple, List
from enum import Enum
from dataclasses import dataclass

from MathAndPhysicsTools import handle_nan, DEFAULT_WAVELENGTH_NM


# =============================================================================
# CONFIGURATION GLOBALE
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Camera")


# =============================================================================
# CONSTANTES
# =============================================================================

ELECTRON_CHARGE_C = 1.602176634e-19
PLANCK_CONSTANT_J_S = 6.62607015e-34
SPEED_OF_LIGHT_M_PER_S = 299792458.0

TYPICAL_PIXEL_SIZE_UM = 5.0
TYPICAL_WELL_DEPTH_ELECTRONS = 100000
TYPICAL_READ_NOISE_ELECTRONS = 3.0
TYPICAL_DARK_CURRENT_ELECTRONS_PER_S = 0.1
TYPICAL_ADU_PER_ELECTRON = 1.0
TYPICAL_QUANTUM_EFFICIENCY = 0.8


# =============================================================================
# ENUMS
# =============================================================================

class CameraType(Enum):
    IDEAL = "ideal"
    CCD = "ccd"
    CMOS = "cmos"


class NoiseType(Enum):
    QUANTUM = "quantum"
    THERMAL = "thermal"
    READOUT = "readout"


class ResponseType(Enum):
    FLAT = "flat"
    GAUSSIAN = "gaussian"


# =============================================================================
# CLASSE: SPECTRAL RESPONSE
# =============================================================================

@dataclass
class SpectralResponse:
    """FR: Réponse spectrale. EN: Spectral response."""
    response_type: ResponseType = ResponseType.FLAT
    peak_wavelength_nm: float = DEFAULT_WAVELENGTH_NM
    fwhm_nm: float = 100.0
    quantum_efficiency: float = TYPICAL_QUANTUM_EFFICIENCY
    
    def get_qe(self, wavelength_nm: float) -> float:
        """FR: Retourne l'efficacité quantique. EN: Returns quantum efficiency."""
        if self.response_type == ResponseType.FLAT:
            return self.quantum_efficiency
        elif self.response_type == ResponseType.GAUSSIAN:
            sigma = self.fwhm_nm / (2 * np.sqrt(2 * np.log(2)))
            return self.quantum_efficiency * np.exp(
                -((wavelength_nm - self.peak_wavelength_nm) ** 2) / (2 * sigma ** 2)
            )
        return self.quantum_efficiency


# =============================================================================
# CLASSE: CAMERA
# =============================================================================

class Camera:
    """
    FR: Caméra optique.
    EN: Optical camera.
    
    Sources:
        - "Charge-Coupled Devices" by J. Janesick (2001)
        - "CMOS/CCD Sensors and Camera Systems" by G. C. Holst & T. L. Lomheim (2011)
    """

    def __init__(self,
                 name: str = "Camera",
                 camera_type: CameraType = CameraType.CCD,
                 num_pixels_x: int = 1024,
                 num_pixels_y: int = 1024,
                 pixel_size_um: float = TYPICAL_PIXEL_SIZE_UM,
                 well_depth_electrons: float = TYPICAL_WELL_DEPTH_ELECTRONS,
                 read_noise_electrons: float = TYPICAL_READ_NOISE_ELECTRONS,
                 dark_current_electrons_per_s: float = TYPICAL_DARK_CURRENT_ELECTRONS_PER_S,
                 adu_per_electron: float = TYPICAL_ADU_PER_ELECTRON,
                 spectral_response: Optional[SpectralResponse] = None,
                 wavelength_nm: float = DEFAULT_WAVELENGTH_NM,
                 exposure_time_s: float = 0.1):
        self.name = name
        self.camera_type = camera_type
        self.num_pixels_x = num_pixels_x
        self.num_pixels_y = num_pixels_y
        self.pixel_size_um = pixel_size_um
        self.well_depth_electrons = well_depth_electrons
        self.read_noise_electrons = read_noise_electrons
        self.dark_current_electrons_per_s = dark_current_electrons_per_s
        self.adu_per_electron = adu_per_electron
        self.wavelength_nm = wavelength_nm
        self.exposure_time_s = exposure_time_s
        
        self.sensor_width_mm = num_pixels_x * pixel_size_um * 1e-3
        self.sensor_height_mm = num_pixels_y * pixel_size_um * 1e-3
        self.full_well_capacity_adu = well_depth_electrons * adu_per_electron
        self.spectral_response = spectral_response if spectral_response is not None else SpectralResponse()
        self.gain = 1.0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', {self.num_pixels_x}x{self.num_pixels_y})"

    def capture_image(self, intensity_map: np.ndarray, wavelength_nm: Optional[float] = None) -> np.ndarray:
        """FR: Capture une image. EN: Captures an image."""
        if wavelength_nm is None:
            wavelength_nm = self.wavelength_nm
        
        electrons_map = self._intensity_to_electrons(intensity_map, wavelength_nm)
        noisy_electrons = self._add_noise(electrons_map)
        return self._electrons_to_adu(noisy_electrons)

    def _intensity_to_electrons(self, intensity_map: np.ndarray, wavelength_nm: float) -> np.ndarray:
        """FR: Convertit l'intensité en électrons. EN: Converts intensity to electrons."""
        qe = self.spectral_response.get_qe(wavelength_nm)
        pixel_area_m2 = (self.pixel_size_um * 1e-6) ** 2
        photon_energy_J = (PLANCK_CONSTANT_J_S * SPEED_OF_LIGHT_M_PER_S) / (wavelength_nm * 1e-9)
        photons_per_pixel = intensity_map * pixel_area_m2 * self.exposure_time_s / photon_energy_J
        electrons_map = photons_per_pixel * qe * self.gain
        return np.clip(electrons_map, 0, self.well_depth_electrons)

    def _add_noise(self, electrons_map: np.ndarray) -> np.ndarray:
        """FR: Ajoute du bruit. EN: Adds noise."""
        noisy_electrons = np.random.poisson(electrons_map)
        dark_electrons = self.dark_current_electrons_per_s * self.exposure_time_s
        noisy_electrons += np.random.poisson(dark_electrons, size=electrons_map.shape)
        read_noise = np.random.normal(0, self.read_noise_electrons, size=electrons_map.shape)
        noisy_electrons += read_noise
        return np.clip(noisy_electrons, 0, self.well_depth_electrons)

    def _electrons_to_adu(self, electrons_map: np.ndarray) -> np.ndarray:
        """FR: Convertit les électrons en ADU. EN: Converts electrons to ADU."""
        return electrons_map * self.adu_per_electron

    def simulate_dark_image(self) -> np.ndarray:
        """FR: Simule une image de noir. EN: Simulates a dark image."""
        intensity_map = np.zeros((self.num_pixels_y, self.num_pixels_x))
        return self.capture_image(intensity_map)

    def simulate_flat_image(self, intensity_value: float = 1.0) -> np.ndarray:
        """FR: Simule une image de flat field. EN: Simulates a flat field image."""
        intensity_map = np.ones((self.num_pixels_y, self.num_pixels_x)) * intensity_value
        return self.capture_image(intensity_map)

    def simulate_airy_spots(self, spot_positions: np.ndarray, spot_intensities: np.ndarray, spot_width_um: float = 10.0) -> np.ndarray:
        """FR: Simule des tâches d'Airy. EN: Simulates Airy spots."""
        intensity_map = np.zeros((self.num_pixels_y, self.num_pixels_x))
        pixel_size_mm = self.pixel_size_um * 1e-3
        
        for position, intensity in zip(spot_positions, spot_intensities):
            x_pixel = int(position[0] / pixel_size_mm)
            y_pixel = int(position[1] / pixel_size_mm)
            width_pixels = int(spot_width_um / self.pixel_size_um)
            
            for dx in range(-width_pixels, width_pixels + 1):
                for dy in range(-width_pixels, width_pixels + 1):
                    x_idx = x_pixel + dx
                    y_idx = y_pixel + dy
                    if 0 <= x_idx < self.num_pixels_x and 0 <= y_idx < self.num_pixels_y:
                        distance_pixels = np.sqrt(dx**2 + dy**2)
                        intensity_map[y_idx, x_idx] += intensity * np.exp(-distance_pixels**2 / (2 * (width_pixels / 2)**2))
        
        return self.capture_image(intensity_map)

    def get_image(self, intensity_map: Optional[np.ndarray] = None) -> np.ndarray:
        """FR: Retourne une image. EN: Returns an image."""
        if intensity_map is None:
            return self.simulate_flat_image()
        return self.capture_image(intensity_map)


# =============================================================================
# CLASSE: PERFECT CAMERA
# =============================================================================

class PerfectCamera(Camera):
    """FR: Caméra parfaite. EN: Perfect camera."""

    def __init__(self,
                 name: str = "PerfectCamera",
                 num_pixels_x: int = 1024,
                 num_pixels_y: int = 1024,
                 pixel_size_um: float = TYPICAL_PIXEL_SIZE_UM,
                 wavelength_nm: float = DEFAULT_WAVELENGTH_NM):
        super().__init__(
            name=name, camera_type=CameraType.IDEAL, num_pixels_x=num_pixels_x,
            num_pixels_y=num_pixels_y, pixel_size_um=pixel_size_um,
            well_depth_electrons=float('inf'), read_noise_electrons=0.0,
            dark_current_electrons_per_s=0.0, adu_per_electron=1.0,
            wavelength_nm=wavelength_nm
        )
        self.spectral_response = SpectralResponse(response_type=ResponseType.FLAT, quantum_efficiency=1.0)

    def capture_image(self, intensity_map: np.ndarray, wavelength_nm: Optional[float] = None) -> np.ndarray:
        """FR: Capture une image parfaite. EN: Captures a perfect image."""
        intensity_normalized = intensity_map / np.max(intensity_map)
        return intensity_normalized * 65535


# =============================================================================
# FONCTIONS DE CRÉATION
# =============================================================================

def create_camera(name: str = "Camera", camera_type: CameraType = CameraType.CCD,
                  num_pixels_x: int = 1024, num_pixels_y: int = 1024,
                  pixel_size_um: float = TYPICAL_PIXEL_SIZE_UM,
                  well_depth_electrons: float = TYPICAL_WELL_DEPTH_ELECTRONS,
                  read_noise_electrons: float = TYPICAL_READ_NOISE_ELECTRONS,
                  dark_current_electrons_per_s: float = TYPICAL_DARK_CURRENT_ELECTRONS_PER_S,
                  adu_per_electron: float = TYPICAL_ADU_PER_ELECTRON,
                  spectral_response: Optional[SpectralResponse] = None,
                  wavelength_nm: float = DEFAULT_WAVELENGTH_NM,
                  exposure_time_s: float = 0.1) -> Camera:
    """FR: Crée une caméra. EN: Creates a camera."""
    return Camera(name, camera_type, num_pixels_x, num_pixels_y, pixel_size_um,
                  well_depth_electrons, read_noise_electrons, dark_current_electrons_per_s,
                  adu_per_electron, spectral_response, wavelength_nm, exposure_time_s)


def create_perfect_camera(name: str = "PerfectCamera", num_pixels_x: int = 1024,
                           num_pixels_y: int = 1024, pixel_size_um: float = TYPICAL_PIXEL_SIZE_UM,
                           wavelength_nm: float = DEFAULT_WAVELENGTH_NM) -> PerfectCamera:
    """FR: Crée une caméra parfaite. EN: Creates a perfect camera."""
    return PerfectCamera(name, num_pixels_x, num_pixels_y, pixel_size_um, wavelength_nm)


# =============================================================================
# TESTS UNITAIRES
# =============================================================================

class TestCamera:
    """FR: Tests unitaires pour Camera.py."""

    def test_camera_creation(self):
        """FR: Test la création d'une caméra."""
        camera = create_camera(name="TestCamera", num_pixels_x=100, num_pixels_y=100)
        assert camera.name == "TestCamera"
        assert camera.num_pixels_x == 100

    def test_perfect_camera(self):
        """FR: Test la caméra parfaite."""
        camera = create_perfect_camera(name="PerfectCam", num_pixels_x=100)
        assert camera.camera_type == CameraType.IDEAL

    def test_capture_image(self):
        """FR: Test la capture d'une image."""
        camera = create_camera(num_pixels_x=100, num_pixels_y=100)
        intensity_map = np.ones((100, 100)) * 0.5
        image = camera.capture_image(intensity_map)
        assert image.shape == (100, 100)


if __name__ == "__main__":
    import unittest
    unittest.main()

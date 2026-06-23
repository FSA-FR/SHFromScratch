"""
Simulation.py

FR: Module principal pour la simulation complète d'un système Shack-Hartmann.
    
    Ce module permet de :
    - Créer une simulation complète (faisceau + matrice de microlentilles + caméra)
    - Simuler la propagation du faisceau à travers le système
    - Capturer les tâches d'Airy sur la caméra
    - Détecter les centroïdes et calculer les pentes
    - Reconstruire le front d'onde (modale et zonale)
    - Comparer les résultats avec le front d'onde parfait
    
    Unités :
    - Longueurs : mm
    - Longueur d'onde : nm
    - Phase : nm (principale), rad (pour les calculs)

EN: Main module for complete Shack-Hartmann system simulation.
    
    This module allows:
    - Creating a complete simulation (beam + microlens array + camera)
    - Simulating beam propagation through the system
    - Capturing Airy spots on the camera
    - Detecting centroids and calculating slopes
    - Reconstructing the wavefront (modal and zonal)
    - Comparing results with the perfect wavefront
    
    Units:
    - Lengths: mm
    - Wavelength: nm
    - Phase: nm (main), rad (for calculations)

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch

Dependencies:
    - numpy
    - Beam
    - Microstructure
    - Camera
    - Shack_Hartmann
    - Southwell
    - Visualization
    - MathAndPhysicsTools

Sources:
    - "Principles of Adaptive Optics" by R. K. Tyson (1991)
    - "Shack-Hartmann Wavefront Sensing" by J. W. Hardy (1978)
"""

import numpy as np
import logging
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from datetime import datetime
import time

from Beam import Beam, BeamProfile, create_beam
from Microstructure import MicrolensArray, create_microlens_array
from Camera import PerfectCamera, create_perfect_camera
from Shack_Hartmann import ShackHartmann, create_shack_hartmann, ReconstructionMethod, CentroidMethod, Spot
from Southwell import SouthwellReconstructor
from Visualization import display_2d_array, display_phase, display_airy_spots, display_reconstruction_results
from MathAndPhysicsTools import handle_nan, generate_zernike_modes, ZernikeOrdering, NormalizationType, compute_pv_rms, DEFAULT_WAVELENGTH_NM, TWO_PI


# =============================================================================
# CONFIGURATION GLOBALE
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Simulation")

DEFAULT_BEAM_DIAMETER_MM = 10.0
DEFAULT_BEAM_NUM_POINTS = 512
DEFAULT_ML_ARRAY_NUM_X = 50
DEFAULT_ML_ARRAY_NUM_Y = 50
DEFAULT_ML_DIAMETER_MM = 0.1
DEFAULT_ML_FOCAL_LENGTH_MM = 10.0
DEFAULT_SH_DISTANCE_MM = 10.0
DEFAULT_CAMERA_NUM_PIXELS = 1024
DEFAULT_CAMERA_PIXEL_SIZE_UM = 5.0


# =============================================================================
# CLASSE: SIMULATION RESULT
# =============================================================================

@dataclass
class SimulationResult:
    """FR: Résultat d'une simulation. EN: Result of a simulation."""
    name: str = "Simulation"
    input_phase: Optional[np.ndarray] = None
    spot_image: Optional[np.ndarray] = None
    spots: Optional[List[Spot]] = None
    slopes_x: Optional[np.ndarray] = None
    slopes_y: Optional[np.ndarray] = None
    zonal_wavefront: Optional[np.ndarray] = None
    modal_wavefront: Optional[np.ndarray] = None
    perfect_wavefront: Optional[np.ndarray] = None
    zonal_error: Optional[np.ndarray] = None
    modal_error: Optional[np.ndarray] = None
    perfect_error: Optional[np.ndarray] = None
    zonal_pv: float = 0.0
    zonal_rms: float = 0.0
    modal_pv: float = 0.0
    modal_rms: float = 0.0
    perfect_pv: float = 0.0
    perfect_rms: float = 0.0
    execution_time_s: float = 0.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


# =============================================================================
# CLASSE: SIMULATION
# =============================================================================

class Simulation:
    """
    FR: Simulation complète d'un système Shack-Hartmann.
    EN: Complete Shack-Hartmann system simulation.
    
    Sources:
        - "Principles of Adaptive Optics" by R. K. Tyson (1991)
    """

    def __init__(self,
                 name: str = "Simulation",
                 wavelength_nm: float = DEFAULT_WAVELENGTH_NM,
                 beam_diameter_mm: float = DEFAULT_BEAM_DIAMETER_MM,
                 num_pixels: int = DEFAULT_BEAM_NUM_POINTS,
                 ml_array_num_x: int = DEFAULT_ML_ARRAY_NUM_X,
                 ml_array_num_y: int = DEFAULT_ML_ARRAY_NUM_Y,
                 ml_diameter_mm: float = DEFAULT_ML_DIAMETER_MM,
                 ml_focal_length_mm: float = DEFAULT_ML_FOCAL_LENGTH_MM,
                 sh_distance_mm: float = DEFAULT_SH_DISTANCE_MM,
                 camera_num_pixels: int = DEFAULT_CAMERA_NUM_PIXELS,
                 camera_pixel_size_um: float = DEFAULT_CAMERA_PIXEL_SIZE_UM):
        self.name = name
        self.wavelength_nm = wavelength_nm
        
        self.beam = create_beam(name=f"{name}_Beam", wavelength_nm=wavelength_nm,
                                diameter_mm=beam_diameter_mm, num_points=num_pixels)
        
        self.sh_sensor = create_shack_hartmann(
            name=f"{name}_SH", num_microlens_x=ml_array_num_x, num_microlens_y=ml_array_num_y,
            microlens_diameter_mm=ml_diameter_mm, microlens_focal_length_mm=ml_focal_length_mm,
            sh_distance_mm=sh_distance_mm, wavelength_nm=wavelength_nm,
            num_pixels_x=camera_num_pixels, num_pixels_y=camera_num_pixels,
            pixel_size_um=camera_pixel_size_um
        )
        self.num_pixels = num_pixels
        self.results = []

    def _create_coordinate_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """FR: Crée une grille de coordonnées. EN: Creates a coordinate grid."""
        x = np.linspace(-self.beam.diameter_mm / 2, self.beam.diameter_mm / 2, self.num_pixels)
        y = np.linspace(-self.beam.diameter_mm / 2, self.beam.diameter_mm / 2, self.num_pixels)
        X, Y = np.meshgrid(x, y)
        return X, Y

    def _generate_random_wavefront(self, amplitude_nm: float) -> np.ndarray:
        """FR: Génère un front d'onde aléatoire. EN: Generates a random wavefront."""
        X, Y = self._create_coordinate_grid()
        X_norm = X / (self.beam.diameter_mm / 2)
        Y_norm = Y / (self.beam.diameter_mm / 2)
        
        wavefront = np.zeros_like(X_norm)
        max_degree = 5
        zernike_modes = generate_zernike_modes(max_degree, ZernikeOrdering.NOLL, NormalizationType.PV, self.num_pixels)
        np.random.seed()
        coefficients = np.random.normal(0, amplitude_nm / (max_degree + 1), len(zernike_modes))
        
        for i, (n, m, Z) in enumerate(zernike_modes):
            wavefront += coefficients[i] * Z
        return handle_nan(wavefront, method='zero')

    def run_single_simulation(self, input_phase_map: np.ndarray, name: str = "SingleSimulation") -> SimulationResult:
        """FR: Exécute une simulation unique. EN: Runs a single simulation."""
        start_time = time.time()
        
        self.beam.generate_electric_field(method=BeamProfile.UNIFORM)
        X, Y = self._create_coordinate_grid()
        phase_rad = input_phase_map * (TWO_PI / self.wavelength_nm)
        self.beam.electric_field = self.beam.electric_field * np.exp(1j * phase_rad)
        self.beam.phase = input_phase_map
        
        spot_image = self.sh_sensor.simulate_spot_image(input_phase_map)
        spots = self.sh_sensor.detect_spot_centroids(spot_image, method=CentroidMethod.CENTER_OF_MASS)
        slopes_x, slopes_y = self.sh_sensor.calculate_slopes(spots)
        
        zonal_wavefront = self.sh_sensor.reconstruct_wavefront(slopes_x, slopes_y, method=ReconstructionMethod.ZONAL)
        modal_wavefront = self.sh_sensor.reconstruct_wavefront(slopes_x, slopes_y, method=ReconstructionMethod.MODAL)
        
        zonal_error = input_phase_map - zonal_wavefront
        modal_error = input_phase_map - modal_wavefront
        perfect_error = np.zeros_like(input_phase_map)
        
        zonal_pv, zonal_rms = compute_pv_rms(zonal_error)
        modal_pv, modal_rms = compute_pv_rms(modal_error)
        perfect_pv, perfect_rms = compute_pv_rms(perfect_error)
        
        execution_time = time.time() - start_time
        
        result = SimulationResult(
            name=name, input_phase=input_phase_map, spot_image=spot_image, spots=spots,
            slopes_x=slopes_x, slopes_y=slopes_y, zonal_wavefront=zonal_wavefront,
            modal_wavefront=modal_wavefront, perfect_wavefront=input_phase_map,
            zonal_error=zonal_error, modal_error=modal_error, perfect_error=perfect_error,
            zonal_pv=zonal_pv, zonal_rms=zonal_rms, modal_pv=modal_pv, modal_rms=modal_rms,
            perfect_pv=perfect_pv, perfect_rms=perfect_rms, execution_time_s=execution_time
        )
        self.results.append(result)
        return result

    def display_result(self, result: SimulationResult, save_path: Optional[str] = None, show: bool = False) -> None:
        """FR: Affiche les résultats. EN: Displays the results."""
        if result.input_phase is not None:
            display_phase(result.input_phase, title=f"Phase d'entrée - {result.name}",
                         save_path=f"{save_path}_input_phase.png" if save_path else None, show=show)
        if result.spot_image is not None:
            display_airy_spots(result.spot_image, title=f"Tâches d'Airy - {result.name}",
                              save_path=f"{save_path}_spot_image.png" if save_path else None, show=show)
        if result.zonal_wavefront is not None and result.modal_wavefront is not None:
            display_reconstruction_results(result.zonal_wavefront, result.modal_wavefront,
                                           result.perfect_wavefront, title=f"Reconstruction - {result.name}",
                                           save_path=f"{save_path}_reconstruction.png" if save_path else None, show=show)


# =============================================================================
# FONCTION DE CRÉATION
# =============================================================================

def create_simulation(name: str = "Simulation", wavelength_nm: float = DEFAULT_WAVELENGTH_NM,
                      beam_diameter_mm: float = DEFAULT_BEAM_DIAMETER_MM,
                      num_pixels: int = DEFAULT_BEAM_NUM_POINTS,
                      ml_array_num_x: int = DEFAULT_ML_ARRAY_NUM_X,
                      ml_array_num_y: int = DEFAULT_ML_ARRAY_NUM_Y,
                      ml_diameter_mm: float = DEFAULT_ML_DIAMETER_MM,
                      ml_focal_length_mm: float = DEFAULT_ML_FOCAL_LENGTH_MM,
                      sh_distance_mm: float = DEFAULT_SH_DISTANCE_MM,
                      camera_num_pixels: int = DEFAULT_CAMERA_NUM_PIXELS,
                      camera_pixel_size_um: float = DEFAULT_CAMERA_PIXEL_SIZE_UM) -> Simulation:
    """FR: Crée une simulation. EN: Creates a simulation."""
    return Simulation(name, wavelength_nm, beam_diameter_mm, num_pixels,
                      ml_array_num_x, ml_array_num_y, ml_diameter_mm, ml_focal_length_mm,
                      sh_distance_mm, camera_num_pixels, camera_pixel_size_um)


# =============================================================================
# TESTS UNITAIRES
# =============================================================================

class TestSimulation:
    """FR: Tests unitaires pour Simulation.py."""

    def test_creation(self):
        """FR: Test la création."""
        sim = create_simulation(name="TestSimulation", num_pixels=128, ml_array_num_x=5, ml_array_num_y=5)
        assert sim.name == "TestSimulation"

    def test_single_simulation(self):
        """FR: Test une simulation unique."""
        sim = create_simulation(num_pixels=128, ml_array_num_x=5, ml_array_num_y=5)
        input_phase = np.zeros((128, 128))
        result = sim.run_single_simulation(input_phase)
        assert result is not None


if __name__ == "__main__":
    import unittest
    unittest.main()

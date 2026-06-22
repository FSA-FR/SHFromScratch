"""
Simulation.py

FR: Simulation complète d'un système Shack-Hartmann.
    Ce module permet de :
    - Générer un faisceau circulaire d'intensité gaussienne (diamètre 5mm, λ=635nm)
    - Appliquer des aberrations aléatoires définies par les 10 premiers modes de Zernike (ordonnancement Wyant)
    - Faire traverser le faisceau à travers une matrice de microlentilles plan-convexe en Fused Silica
      (épaisseur 2mm, pas 100µm, focale 2mm, qualité λ/10 PV, parallélisme 5 arcsec, S-D 20-10)
    - Mesurer les tâches d'Airy avec une caméra CMOS virtuelle (1449x1449 pixels, 3.45µm/pixel, 5x5mm²)
    - Calculer les pentes locales (déplacement des centroïdes par rapport à une référence parfaite)
    - Reconstruire la phase en mode ZONAL (Southwell) et MODAL (Zernike)
    - Comparer avec un Shack-Hartmann parfait (caméra sans bruit, microlentilles paraxiales parfaites)
    - Effectuer 10 tirages d'aberrations pour chaque amplitude (50, 250, 500, 1000 nm RMS) → 40 simulations
    - Tracer le comportement de l'erreur pour chaque tirage

    Chaque image générée aura :
    - Une échelle visuelle
    - Le PV (Peak-to-Valley) et le RMS des valeurs
    - Colormap : "Jet" pour la phase, "hot" pour l'intensité

    Unités :
    - Longueurs : mm (faisceau, matrice), µm (microlentilles, pixels)
    - Longueur d'onde : nm (635 nm)
    - Phase : nm (principale), rad (pour les calculs)
    - Pentes : rad (radians)

EN: Complete Shack-Hartmann simulation system.
    This module allows:
    - Generating a circular Gaussian beam (diameter 5mm, λ=635nm)
    - Applying random aberrations defined by the first 10 Zernike modes (Wyant ordering)
    - Propagating the beam through a plano-convex microlens array in Fused Silica
      (thickness 2mm, pitch 100µm, focal length 2mm, λ/10 PV quality, 5 arcsec parallelism, S-D 20-10)
    - Measuring Airy spots with a virtual CMOS camera (1449x1449 pixels, 3.45µm/pixel, 5x5mm²)
    - Calculating local slopes (centroid displacement relative to perfect reference)
    - Reconstructing phase in ZONAL mode (Southwell) and MODAL mode (Zernike)
    - Comparing with a perfect Shack-Hartmann (noise-free camera, perfect paraxial microlenses)
    - Performing 10 aberration draws for each amplitude (50, 250, 500, 1000 nm RMS) → 40 simulations
    - Plotting error behavior for each draw

    Each generated image will have:
    - A visual scale
    - PV (Peak-to-Valley) and RMS values
    - Colormap: "Jet" for phase, "hot" for intensity

    Units:
    - Lengths: mm (beam, array), µm (microlenses, pixels)
    - Wavelength: nm (635 nm)
    - Phase: nm (main), rad (for calculations)
    - Slopes: rad (radians)

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch

Dependencies:
    - numpy
    - matplotlib
    - Beam.py
    - Optiques.py (for Zernike polynomials, WaveFrontError)
    - Microstructure.py (for microlens arrays)
    - Camera.py (for CMOS camera)
    - Propagation.py (for beam propagation)
    - Shack_Hartmann.py (for slope calculation)
    - Southwell.py (for phase reconstruction)

Sources:
    - "Principles of Adaptive Optics" by R.K. Tyson (Academic Press, 1991)
      -> Shack-Hartmann principles and slope calculation
    - "Wavefront estimation from wavefront slope measurements" by W.H. Southwell (1980)
      -> Southwell algorithm for phase reconstruction
    - "Zernike polynomials and atmospheric turbulence" by J.W. Noll (1976)
      -> Zernike modes for wavefront aberrations
    - "ISO 10110-7: Scratch/Dig" (MIL-SPEC equivalent)
      -> S-D 20-10 surface quality specification
    - "Optical System Design" by R. Kingslake (1983)
      -> Plano-convex lens design and parameters
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging


# =============================================================================
# IMPORT DES MODULES LOCAUX
# =============================================================================

try:
    from Beam import Beam
    BEAM_AVAILABLE = True
except ImportError as e:
    BEAM_AVAILABLE = False
    logging.error(f"Beam module not available: {e}")

try:
    from Optiques import (
        WaveFrontError, generate_zernike_polynomial, ZernikeOrdering
    )
    OPTIQUES_AVAILABLE = True
except ImportError as e:
    OPTIQUES_AVAILABLE = False
    logging.error(f"Optiques module not available: {e}")

try:
    from Microstructure import create_microlens_array, MicrolensArray
    MICROSTRUCTURE_AVAILABLE = True
except ImportError as e:
    MICROSTRUCTURE_AVAILABLE = False
    logging.error(f"Microstructure module not available: {e}")

try:
    from Camera import RealCamera, IdealCamera, Camera
    CAMERA_AVAILABLE = True
except ImportError as e:
    CAMERA_AVAILABLE = False
    logging.error(f"Camera module not available: {e}")

try:
    from Propagation import Propagation
    PROPAGATION_AVAILABLE = True
except ImportError as e:
    PROPAGATION_AVAILABLE = False
    logging.error(f"Propagation module not available: {e}")

try:
    from Shack_Hartmann import (
        ShackHartmann, create_shack_hartmann, CentroidAlgorithm
    )
    SHACK_HARTMANN_AVAILABLE = True
except ImportError as e:
    SHACK_HARTMANN_AVAILABLE = False
    logging.error(f"Shack_Hartmann module not available: {e}")

try:
    from Southwell import (
        SouthwellReconstructor, create_southwell_reconstructor,
        ReconstructionAlgorithm, ReconstructionResult
    )
    SOUTHWELL_AVAILABLE = True
except ImportError as e:
    SOUTHWELL_AVAILABLE = False
    logging.error(f"Southwell module not available: {e}")


# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Simulation")


# =============================================================================
# CONSTANTES GLOBALES (SPÉCIFICATIONS UTILISATEUR)
# =============================================================================

# Faisceau
WAVELENGTH_NM = 635.0  # nm
BEAM_DIAMETER_MM = 5.0  # mm
BEAM_SIGMA_MM = BEAM_DIAMETER_MM / 4  # σ pour une intensité gaussienne

# Aberrations (10 premiers modes Zernike en ordonnancement Wyant)
# Ordonnancement Wyant : (n, m) où n est le degré radial, m est le degré azimuthal
# Les 10 premiers modes (index 0 à 9) :
WYANT_MODES = [
    (0, 0),   # 0: Piston
    (1, 1),   # 1: Tilt X
    (1, -1),  # 2: Tilt Y
    (2, 0),   # 3: Defocus
    (2, 2),   # 4: Astigmatisme 0°
    (2, -2),  # 5: Astigmatisme 45°
    (3, -1),  # 6: Coma Y
    (3, 1),   # 7: Coma X
    (3, -3),  # 8: Trefle 0°
    (3, 3)    # 9: Trefle 45°
]

# Amplitudes RMS pour les tirages (en nm)
AMPLITUDES_RMS_NM = [50, 250, 500, 1000]  # nm RMS
NUM_DRAWS_PER_AMPLITUDE = 10  # 10 tirages par amplitude
TOTAL_SIMULATIONS = len(AMPLITUDES_RMS_NM) * NUM_DRAWS_PER_AMPLITUDE  # 40

# Matrice de microlentilles
MATRIX_SIZE_MM = 5.0  # 5x5 mm²
MICROLENS_SIDE_UM = 100.0  # 100 µm de côté (carrées)
MICROLENS_SIDE_MM = MICROLENS_SIDE_UM * 1e-3  # 0.1 mm
FOCAL_LENGTH_MM = 2.0  # mm
THICKNESS_MM = 2.0  # mm
MATERIAL_NAME = "Fused_Silica"
SURFACE_QUALITY_PV_NM = WAVELENGTH_NM / 10  # 63.5 nm PV
PARALLELISM_ARCSEC = 5.0  # arcsecondes
SCRATCH_DIG = "20-10"  # Norme MIL-SPEC

# Caméra
CAMERA_SIZE_MM = 5.0  # 5x5 mm²
PIXEL_SIZE_UM = 3.45  # µm
PIXEL_SIZE_MM = PIXEL_SIZE_UM * 1e-3  # mm
EXPOSURE_TIME_S = 0.1  # secondes

# Calcul du nombre de pixels
NUM_PIXELS_X = int(round(CAMERA_SIZE_MM / PIXEL_SIZE_MM))
NUM_PIXELS_Y = int(round(CAMERA_SIZE_MM / PIXEL_SIZE_MM))  # 1449

# Nombre de microlentilles
NUM_MICROLENS_X = int(MATRIX_SIZE_MM / MICROLENS_SIDE_MM)  # 50
NUM_MICROLENS_Y = int(MATRIX_SIZE_MM / MICROLENS_SIDE_MM)  # 50

# Position de la caméra (au point focal)
CAMERA_POSITION_MM = FOCAL_LENGTH_MM


# =============================================================================
# CLASSE PRINCIPALE: SHACK-HARTMANN SIMULATION
# =============================================================================

class ShackHartmannSimulation:
    """
    FR: Simulation complète d'un système Shack-Hartmann.
        
        Ce module gère :
        - La génération du faisceau avec aberrations
        - La propagation à travers la matrice de microlentilles
        - La mesure avec la caméra CMOS
        - Le calcul des pentes locales
        - La reconstruction de phase (zonal et modal)
        - La comparaison avec un système parfait
        - L'analyse statistique sur plusieurs tirages
    
    EN: Complete Shack-Hartmann simulation system.
        
        This module handles:
        - Beam generation with aberrations
        - Propagation through microlens array
        - Measurement with CMOS camera
        - Local slope calculation
        - Phase reconstruction (zonal and modal)
        - Comparison with perfect system
        - Statistical analysis over multiple draws
    
    Attributes:
        name (str): Nom de la simulation.
        wavelength_nm (float): Longueur d'onde en nm.
        beam_diameter_mm (float): Diamètre du faisceau en mm.
        microlens_array (MicrolensArray): Matrice de microlentilles.
        real_camera (Camera): Caméra CMOS réelle (avec bruit).
        perfect_camera (Camera): Caméra idéale (sans bruit).
        reconstructor (SouthwellReconstructor): Reconstructeur de phase.
        output_dir (str): Répertoire de sortie.
    """

    def __init__(self,
                 name: str = "SH_Simulation",
                 output_dir: str = "simulation_output",
                 display: bool = True):
        """
        FR: Initialise la simulation.
        
        EN: Initializes the simulation.
        
        Args:
            name (str): Nom de la simulation.
            output_dir (str): Répertoire de sortie.
            display (bool): Afficher les images.
        """
        self.name = name
        self.output_dir = output_dir
        self.display = display
        
        # Créer le répertoire de sortie
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialiser les composants
        self._initialize_components()
        
        logger.info(f"Simulation '{name}' initialisée")
        logger.info(f"  Faisceau: λ={WAVELENGTH_NM}nm, diamètre={BEAM_DIAMETER_MM}mm")
        logger.info(f"  Matrice: {NUM_MICROLENS_X}x{NUM_MICROLENS_Y} microlentilles, "
                   f"côté={MICROLENS_SIDE_UM}µm, f={FOCAL_LENGTH_MM}mm")
        logger.info(f"  Caméra: {NUM_PIXELS_X}x{NUM_PIXELS_Y} pixels, "
                   f"taille={PIXEL_SIZE_UM}µm, {CAMERA_SIZE_MM}mm²")

    def _initialize_components(self) -> None:
        """
        FR: Initialise tous les composants de la simulation.
        
        EN: Initializes all simulation components.
        """
        # Matrice de microlentilles (réelle)
        self.microlens_array = self._create_microlens_array()
        
        # Matrice de microlentilles parfaite (pour comparaison)
        self.perfect_microlens_array = self._create_perfect_microlens_array()
        
        # Caméra CMOS (réelle, avec bruit)
        self.real_camera = self._create_real_camera()
        
        # Caméra idéale (sans bruit, pour comparaison)
        self.perfect_camera = self._create_perfect_camera()
        
        # Systèmes Shack-Hartmann
        self.real_sh = self._create_shack_hartmann(
            self.microlens_array, self.real_camera, "Real"
        )
        self.perfect_sh = self._create_shack_hartmann(
            self.perfect_microlens_array, self.perfect_camera, "Perfect"
        )
        
        # Reconstructeur de phase
        self.reconstructor = self._create_reconstructor()

    def _create_microlens_array(self) -> MicrolensArray:
        """
        FR: Crée la matrice de microlentilles réelle.
        
        EN: Creates the real microlens array.
        """
        if not MICROSTRUCTURE_AVAILABLE:
            raise ImportError("Microstructure module required")
        
        microlens_array = create_microlens_array(
            name="Matrice_Microlentilles",
            pitch_mm=0.0,  # Espacement bord-à-bord = 0 (jointives)
            element_diameter_mm=MICROLENS_SIDE_MM,  # 0.1 mm (côté)
            num_elements_x=NUM_MICROLENS_X,
            num_elements_y=NUM_MICROLENS_Y,
            focal_length_mm=FOCAL_LENGTH_MM,
            thickness_mm=THICKNESS_MM,
            material_name=MATERIAL_NAME,
            wavelength_nm=WAVELENGTH_NM,
            edge_to_edge_spacing_mm=0.0  # Jointives
        )
        
        # Ajouter la qualité de surface et le parallélisme
        microlens_array.set_global_wfe(WaveFrontError(
            surface_roughness_nm=SURFACE_QUALITY_PV_NM,
            parallelism_arcsec=PARALLELISM_ARCSEC,
            scratch_dig=SCRATCH_DIG
        ))
        
        return microlens_array

    def _create_perfect_microlens_array(self) -> MicrolensArray:
        """
        FR: Crée la matrice de microlentilles parfaite (sans défauts).
        
        EN: Creates the perfect microlens array (no defects).
        """
        if not MICROSTRUCTURE_AVAILABLE:
            raise ImportError("Microstructure module required")
        
        perfect_array = create_microlens_array(
            name="Matrice_Parfaite",
            pitch_mm=0.0,
            element_diameter_mm=MICROLENS_SIDE_MM,
            num_elements_x=NUM_MICROLENS_X,
            num_elements_y=NUM_MICROLENS_Y,
            focal_length_mm=FOCAL_LENGTH_MM,
            thickness_mm=THICKNESS_MM,
            material_name=MATERIAL_NAME,
            wavelength_nm=WAVELENGTH_NM,
            edge_to_edge_spacing_mm=0.0
        )
        
        return perfect_array

    def _create_real_camera(self) -> Camera:
        """
        FR: Crée la caméra CMOS réelle (avec bruit).
        
        EN: Creates the real CMOS camera (with noise).
        """
        if not CAMERA_AVAILABLE:
            raise ImportError("Camera module required")
        
        return RealCamera(
            name="Caméra_CMOS_Réelle",
            num_pixels_x=NUM_PIXELS_X,
            num_pixels_y=NUM_PIXELS_Y,
            pixel_size_um=PIXEL_SIZE_UM,
            material_name="Silicon",
            quantum_efficiency=0.8,  # Typique CMOS
            full_well_capacity=75000,  # 50,000-100,000
            readout_noise_e=4.0,  # 3-5 électrons RMS
            dark_current_e=0.1,  # 0.1 électrons/pixel/s
            exposure_time_s=EXPOSURE_TIME_S,
            wavelength_nm=WAVELENGTH_NM,
            display=False
        )

    def _create_perfect_camera(self) -> Camera:
        """
        FR: Crée la caméra idéale (sans bruit).
        
        EN: Creates the ideal camera (no noise).
        """
        if not CAMERA_AVAILABLE:
            raise ImportError("Camera module required")
        
        return IdealCamera(
            name="Caméra_Parfaite",
            num_pixels_x=NUM_PIXELS_X,
            num_pixels_y=NUM_PIXELS_Y,
            pixel_size_um=PIXEL_SIZE_UM,
            wavelength_nm=WAVELENGTH_NM,
            display=False
        )

    def _create_shack_hartmann(self,
                               microlens_array: MicrolensArray,
                               camera: Camera,
                               name_suffix: str) -> ShackHartmann:
        """
        FR: Crée un système Shack-Hartmann.
        
        EN: Creates a Shack-Hartmann system.
        
        Args:
            microlens_array: Matrice de microlentilles.
            camera: Caméra.
            name_suffix: Suffixe pour le nom.
        
        Returns:
            ShackHartmann: Système Shack-Hartmann.
        """
        if not SHACK_HARTMANN_AVAILABLE:
            raise ImportError("Shack_Hartmann module required")
        
        return ShackHartmann(
            name=f"SH_{name_suffix}",
            microlens_array=microlens_array,
            camera=camera,
            wavelength_nm=WAVELENGTH_NM,
            focal_length_mm=FOCAL_LENGTH_MM,
            centroid_algorithm=CentroidAlgorithm.WEIGHTED_CENTROID,
            display=False
        )

    def _create_reconstructor(self) -> SouthwellReconstructor:
        """
        FR: Crée le reconstructeur de phase.
        
        EN: Creates the phase reconstructor.
        """
        if not SOUTHWELL_AVAILABLE:
            raise ImportError("Southwell module required")
        
        return create_southwell_reconstructor(
            name="Reconstructeur_Simulation",
            wavelength_nm=WAVELENGTH_NM,
            pixel_size_mm=PIXEL_SIZE_MM
        )

    def generate_aberrated_beam(self,
                                amplitude_rms: float,
                                draw_idx: int) -> Beam:
        """
        FR: Génère un faisceau avec des aberrations aléatoires.
            
            Les aberrations sont définies par les 10 premiers modes de Zernike
            (ordonnancement Wyant) avec des coefficients aléatoires entre 0 et amplitude_rms.
        
        EN: Generates a beam with random aberrations.
            
            Aberrations are defined by the first 10 Zernike modes (Wyant ordering)
            with random coefficients between 0 and amplitude_rms.
        
        Args:
            amplitude_rms: Amplitude RMS maximale en nm.
            draw_idx: Index du tirage (pour le seed).
        
        Returns:
            Beam: Faisceau avec aberrations.
        """
        if not BEAM_AVAILABLE or not OPTIQUES_AVAILABLE:
            raise ImportError("Beam or Optiques module required")
        
        # Créer le faisceau gaussien
        beam = Beam(
            wavelength_nm=WAVELENGTH_NM,
            diameter_mm=BEAM_DIAMETER_MM,
            num_points=NUM_PIXELS_X
        )
        
        # Générer le champ électrique gaussien
        electric_field = beam.generate_electric_field(
            method="gaussian",
            sigma_mm=BEAM_SIGMA_MM
        )
        
        # Générer les coefficients aléatoires pour les modes de Zernike
        # Utiliser un seed unique pour chaque tirage
        seed = 42 + int(amplitude_rms) + draw_idx * 100
        np.random.seed(seed)
        
        coefficients = {}
        for mode in WYANT_MODES:
            # Coefficient aléatoire entre 0 et amplitude_rms (RMS)
            # Note: Pour une distribution uniforme, l'amplitude RMS sera amplitude_rms / sqrt(2)
            # Mais on veut que l'amplitude RMS totale soit amplitude_rms
            # Donc on divise par sqrt(10) pour avoir la bonne amplitude RMS totale
            # (car il y a 10 modes indépendants)
            coeff = np.random.uniform(0, amplitude_rms * np.sqrt(10))
            coefficients[mode] = coeff
        
        # Créer la carte de phase
        x = np.linspace(-beam.diameter_mm/2, beam.diameter_mm/2, beam.num_points)
        y = np.linspace(-beam.diameter_mm/2, beam.diameter_mm/2, beam.num_points)
        X, Y = np.meshgrid(x, y)
        
        # Normaliser les coordonnées pour les polynômes de Zernike (entre -1 et 1)
        X_norm = X / (beam.diameter_mm / 2)
        Y_norm = Y / (beam.diameter_mm / 2)
        R = np.sqrt(X_norm**2 + Y_norm**2)
        
        phase_map_nm = np.zeros_like(X)
        for (n, m), coeff in coefficients.items():
            # Générer le polynôme de Zernike (ordonnancement Wyant)
            Z = generate_zernike_polynomial(
                n, m, X_norm, Y_norm,
                ordering=ZernikeOrdering.WYANT
            )
            phase_map_nm += coeff * Z
        
        # Convertir en radians pour le champ électrique
        phase_rad = phase_map_nm * 2 * np.pi / WAVELENGTH_NM
        
        # Appliquer les aberrations au champ électrique
        aberrated_field = np.abs(electric_field) * np.exp(
            1j * (np.angle(electric_field) + phase_rad)
        )
        
        beam.electric_field = aberrated_field
        beam.intensity = beam.compute_intensity_from_electric_field(aberrated_field)
        beam.phase = beam.extract_phase_from_electric_field(aberrated_field)
        
        # Calculer PV et RMS de la phase vraie (pour référence)
        beam.true_pv = float(np.max(phase_map_nm) - np.min(phase_map_nm))
        beam.true_rms = float(np.std(phase_map_nm))
        beam.true_phase = phase_map_nm
        beam.coefficients = coefficients
        
        return beam

    def simulate_single(self,
                       amplitude_rms: float,
                       draw_idx: int) -> Dict[str, Any]:
        """
        FR: Effectue une simulation complète pour un tirage donné.
            
            Étapes :
            1. Générer le faisceau avec aberrations
            2. Simuler le système Shack-Hartmann réel
            3. Simuler le système Shack-Hartmann parfait
            4. Reconstruire la phase (zonal et modal)
            5. Calculer les erreurs
            6. Sauvegarder les images
        
        EN: Performs a complete simulation for a given draw.
            
            Steps:
            1. Generate beam with aberrations
            2. Simulate real Shack-Hartmann system
            3. Simulate perfect Shack-Hartmann system
            4. Reconstruct phase (zonal and modal)
            5. Calculate errors
            6. Save images
        
        Args:
            amplitude_rms: Amplitude RMS en nm.
            draw_idx: Index du tirage.
        
        Returns:
            Dict: Résultats de la simulation.
        """
        start_time = time.time()
        
        # Créer les répertoires de sortie
        amp_dir = os.path.join(self.output_dir, f"amplitude_{amplitude_rms}nm")
        draw_dir = os.path.join(amp_dir, f"draw_{draw_idx}")
        os.makedirs(draw_dir, exist_ok=True)
        
        # 1. Générer le faisceau avec aberrations
        beam = self.generate_aberrated_beam(amplitude_rms, draw_idx)
        
        result = {
            'amplitude_rms': amplitude_rms,
            'draw_idx': draw_idx,
            'true_pv': beam.true_pv,
            'true_rms': beam.true_rms,
            'coefficients': beam.coefficients,
            'timestamp': datetime.now().isoformat()
        }
        
        # 2. Simuler le système Shack-Hartmann réel
        self.real_sh.simulate(beam)
        real_slopes_x, real_slopes_y = self.real_sh.get_slope_maps()
        real_stats = self.real_sh.get_slope_statistics()
        
        result['real_slopes'] = {
            'pv_x': real_stats['slopes_x']['pv'],
            'rms_x': real_stats['slopes_x']['rms'],
            'pv_y': real_stats['slopes_y']['pv'],
            'rms_y': real_stats['slopes_y']['rms']
        }
        
        # 3. Simuler le système Shack-Hartmann parfait
        self.perfect_sh.simulate(beam)
        perfect_slopes_x, perfect_slopes_y = self.perfect_sh.get_slope_maps()
        
        # 4. Reconstruire la phase (zonal et modal)
        # Mode zonal (Southwell avec régularisation)
        zonal_result = self.reconstructor.reconstruct(
            real_slopes_x, real_slopes_y,
            algorithm=ReconstructionAlgorithm.SOUTHWELL_REGULARIZED,
            alpha=0.01
        )
        
        # Mode modal (Zernike)
        modal_result = self.reconstructor.reconstruct(
            real_slopes_x, real_slopes_y,
            algorithm=ReconstructionAlgorithm.MODAL_ZERNIKE,
            max_zernike_degree=10
        )
        
        # Reconstruction parfaite (pour référence)
        perfect_zonal_result = self.reconstructor.reconstruct(
            perfect_slopes_x, perfect_slopes_y,
            algorithm=ReconstructionAlgorithm.SOUTHWELL_REGULARIZED,
            alpha=0.01
        )
        
        # 5. Calculer les erreurs (par rapport à la phase vraie)
        # Note: Les phases reconstruites peuvent avoir un offset constant
        # On soustrait la moyenne pour comparer
        
        # Phase vraie (normalisée)
        true_phase_normalized = beam.true_phase - np.mean(beam.true_phase)
        
        # Erreurs zonal
        zonal_phase_normalized = zonal_result.phase - np.mean(zonal_result.phase)
        zonal_error = zonal_phase_normalized - true_phase_normalized
        
        result['zonal'] = {
            'pv': float(zonal_result.pv),
            'rms': float(zonal_result.rms),
            'error_pv': float(np.max(np.abs(zonal_error))),
            'error_rms': float(np.std(zonal_error)),
            'error_mean': float(np.mean(zonal_error)),
            'computation_time': zonal_result.computation_time
        }
        
        # Erreurs modal
        modal_phase_normalized = modal_result.phase - np.mean(modal_result.phase)
        modal_error = modal_phase_normalized - true_phase_normalized
        
        result['modal'] = {
            'pv': float(modal_result.pv),
            'rms': float(modal_result.rms),
            'error_pv': float(np.max(np.abs(modal_error))),
            'error_rms': float(np.std(modal_error)),
            'error_mean': float(np.mean(modal_error)),
            'computation_time': modal_result.computation_time
        }
        
        # Erreurs parfaites
        perfect_phase_normalized = perfect_zonal_result.phase - np.mean(perfect_zonal_result.phase)
        perfect_error = perfect_phase_normalized - true_phase_normalized
        
        result['perfect'] = {
            'pv': float(perfect_zonal_result.pv),
            'rms': float(perfect_zonal_result.rms),
            'error_pv': float(np.max(np.abs(perfect_error))),
            'error_rms': float(np.std(perfect_error)),
            'error_mean': float(np.mean(perfect_error)),
            'computation_time': perfect_zonal_result.computation_time
        }
        
        # 6. Sauvegarder les images
        if self.display:
            self._save_simulation_images(
                beam, draw_dir, amplitude_rms, draw_idx,
                real_slopes_x, real_slopes_y,
                zonal_result, modal_result, perfect_zonal_result,
                zonal_error, modal_error, perfect_error
            )
        
        # Temps total
        result['total_time'] = time.time() - start_time
        
        logger.info(f"Simulation {amplitude_rms}nm/D{draw_idx} terminée en {result['total_time']:.2f}s")
        
        return result

    def _save_simulation_images(self,
                                beam: Beam,
                                draw_dir: str,
                                amplitude_rms: float,
                                draw_idx: int,
                                real_slopes_x: np.ndarray,
                                real_slopes_y: np.ndarray,
                                zonal_result: ReconstructionResult,
                                modal_result: ReconstructionResult,
                                perfect_result: ReconstructionResult,
                                zonal_error: np.ndarray,
                                modal_error: np.ndarray,
                                perfect_error: np.ndarray) -> None:
        """
        FR: Sauvegarde toutes les images de la simulation.
        
        EN: Saves all simulation images.
        """
        # 1. Faisceau incident (intensité + phase)
        self._save_beam_images(beam, draw_dir)
        
        # 2. Après matrice de microlentilles
        self._save_after_microlenses_images(beam, self.real_sh, draw_dir)
        
        # 3. Plan focal (tâches d'Airy)
        self._save_focal_plane_images(self.real_sh, draw_dir)
        
        # 4. Cartes des pentes
        self._save_slopes_images(real_slopes_x, real_slopes_y, draw_dir)
        
        # 5. Phases reconstruites
        self._save_reconstructed_phase_images(
            zonal_result, modal_result, perfect_result,
            draw_dir
        )
        
        # 6. Cartes d'erreur
        self._save_error_images(
            zonal_error, modal_error, perfect_error,
            draw_dir
        )

    def _save_beam_images(self, beam: Beam, draw_dir: str) -> None:
        """FR: Sauvegarde les images du faisceau incident."""
        # Intensité
        plt.figure(figsize=(10, 8))
        plt.imshow(beam.intensity, cmap='hot',
                  extent=[-beam.diameter_mm/2, beam.diameter_mm/2,
                          -beam.diameter_mm/2, beam.diameter_mm/2])
        plt.title(f"Faisceau incident - Intensité\n"
                 f"PV={beam.true_pv:.2f} nm, RMS={beam.true_rms:.2f} nm")
        plt.xlabel("x (mm)")
        plt.ylabel("y (mm)")
        plt.colorbar(label="Intensité (a.u.)")
        plt.savefig(os.path.join(draw_dir, "beam_intensity.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Phase
        plt.figure(figsize=(10, 8))
        plt.imshow(beam.phase, cmap='Jet',
                  extent=[-beam.diameter_mm/2, beam.diameter_mm/2,
                          -beam.diameter_mm/2, beam.diameter_mm/2])
        plt.title(f"Faisceau incident - Phase\n"
                 f"PV={beam.true_pv:.2f} nm, RMS={beam.true_rms:.2f} nm")
        plt.xlabel("x (mm)")
        plt.ylabel("y (mm)")
        plt.colorbar(label="Phase (nm)")
        plt.savefig(os.path.join(draw_dir, "beam_phase.png"), dpi=150, bbox_inches='tight')
        plt.close()

    def _save_after_microlenses_images(self, beam: Beam, sh: ShackHartmann, draw_dir: str) -> None:
        """FR: Sauvegarde les images après la matrice de microlentilles."""
        # On ne peut pas directement visualiser après les microlentilles
        # car le faisceau est modifié par la matrice
        # Mais on peut afficher la phase de la matrice
        
        # Phase de la matrice de microlentilles
        try:
            from MathAndPhysicsTools import create_grid
            x = np.linspace(-sh.microlens_array.total_width_mm/2,
                          sh.microlens_array.total_width_mm/2, 512)
            y = np.linspace(-sh.microlens_array.total_height_mm/2,
                          sh.microlens_array.total_height_mm/2, 512)
            X, Y = np.meshgrid(x, y)
            
            phase_map = sh.microlens_array.get_total_phase_map(X, Y)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(phase_map, cmap='Jet',
                      extent=[x[0], x[-1], y[0], y[-1]])
            plt.title(f"Phase de la matrice de microlentilles")
            plt.xlabel("x (mm)")
            plt.ylabel("y (mm)")
            plt.colorbar(label="Phase (nm)")
            plt.savefig(os.path.join(draw_dir, "microlens_phase.png"), dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"Impossible de sauvegarder la phase de la matrice: {e}")

    def _save_focal_plane_images(self, sh: ShackHartmann, draw_dir: str) -> None:
        """FR: Sauvegarde les images du plan focal (tâches d'Airy)."""
        if sh.spot_image is None:
            return
        
        # Tâches d'Airy
        plt.figure(figsize=(10, 8))
        extent = [
            -sh.camera.sensor_width_mm/2,
            sh.camera.sensor_width_mm/2,
            -sh.camera.sensor_height_mm/2,
            sh.camera.sensor_height_mm/2
        ]
        plt.imshow(sh.spot_image, cmap='hot', extent=extent, origin='lower')
        plt.title(f"Plan focal - Tâches d'Airy")
        plt.xlabel("x (mm)")
        plt.ylabel("y (mm)")
        plt.colorbar(label="Intensité (ADU)")
        plt.savefig(os.path.join(draw_dir, "focal_plane_spots.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Tâches avec centroïdes
        if sh.centroids is not None:
            centroid_map = sh.get_centroid_map()
            plt.figure(figsize=(10, 8))
            plt.imshow(centroid_map, cmap='hot', extent=extent, origin='lower')
            plt.title(f"Tâches d'Airy avec centroïdes")
            plt.xlabel("x (mm)")
            plt.ylabel("y (mm)")
            plt.colorbar(label="Intensité normalisée")
            plt.savefig(os.path.join(draw_dir, "focal_plane_centroids.png"), dpi=150, bbox_inches='tight')
            plt.close()

    def _save_slopes_images(self, slopes_x: np.ndarray, slopes_y: np.ndarray, draw_dir: str) -> None:
        """FR: Sauvegarde les cartes des pentes locales."""
        # Obtenir les positions des microlentilles
        microlens_positions = self.real_sh.get_microlens_positions()
        microlens_x = microlens_positions[:, 0]
        microlens_y = microlens_positions[:, 1]
        
        x_min, x_max = np.min(microlens_x), np.max(microlens_x)
        y_min, y_max = np.min(microlens_y), np.max(microlens_y)
        
        # Pentes X
        plt.figure(figsize=(10, 8))
        plt.imshow(slopes_x, cmap='Jet',
                  extent=[x_min, x_max, y_min, y_max], origin='lower')
        plt.title(f"Pentes locales X (dφ/dx)")
        plt.xlabel("x (mm)")
        plt.ylabel("y (mm)")
        plt.colorbar(label="Pente (rad)")
        plt.savefig(os.path.join(draw_dir, "slopes_x.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Pentes Y
        plt.figure(figsize=(10, 8))
        plt.imshow(slopes_y, cmap='Jet',
                  extent=[x_min, x_max, y_min, y_max], origin='lower')
        plt.title(f"Pentes locales Y (dφ/dy)")
        plt.xlabel("x (mm)")
        plt.ylabel("y (mm)")
        plt.colorbar(label="Pente (rad)")
        plt.savefig(os.path.join(draw_dir, "slopes_y.png"), dpi=150, bbox_inches='tight')
        plt.close()

    def _save_reconstructed_phase_images(self,
                                         zonal_result: ReconstructionResult,
                                         modal_result: ReconstructionResult,
                                         perfect_result: ReconstructionResult,
                                         draw_dir: str) -> None:
        """FR: Sauvegarde les images des phases reconstruites."""
        # Phase zonal
        plt.figure(figsize=(10, 8))
        plt.imshow(zonal_result.phase, cmap='Jet')
        plt.title(f"Phase reconstruite - Mode Zonal\n"
                 f"PV={zonal_result.pv:.2f} nm, RMS={zonal_result.rms:.2f} nm")
        plt.colorbar(label="Phase (nm)")
        plt.savefig(os.path.join(draw_dir, "phase_zonal.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Phase modal
        plt.figure(figsize=(10, 8))
        plt.imshow(modal_result.phase, cmap='Jet')
        plt.title(f"Phase reconstruite - Mode Modal\n"
                 f"PV={modal_result.pv:.2f} nm, RMS={modal_result.rms:.2f} nm")
        plt.colorbar(label="Phase (nm)")
        plt.savefig(os.path.join(draw_dir, "phase_modal.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Phase parfaite
        plt.figure(figsize=(10, 8))
        plt.imshow(perfect_result.phase, cmap='Jet')
        plt.title(f"Phase reconstruite - Parfait\n"
                 f"PV={perfect_result.pv:.2f} nm, RMS={perfect_result.rms:.2f} nm")
        plt.colorbar(label="Phase (nm)")
        plt.savefig(os.path.join(draw_dir, "phase_perfect.png"), dpi=150, bbox_inches='tight')
        plt.close()

    def _save_error_images(self,
                           zonal_error: np.ndarray,
                           modal_error: np.ndarray,
                           perfect_error: np.ndarray,
                           draw_dir: str) -> None:
        """FR: Sauvegarde les cartes d'erreur."""
        # Erreur zonal
        plt.figure(figsize=(10, 8))
        plt.imshow(zonal_error, cmap='Jet')
        plt.title(f"Erreur - Mode Zonal\n"
                 f"PV={np.max(np.abs(zonal_error)):.2f} nm, RMS={np.std(zonal_error):.2f} nm")
        plt.colorbar(label="Erreur (nm)")
        plt.savefig(os.path.join(draw_dir, "error_zonal.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Erreur modal
        plt.figure(figsize=(10, 8))
        plt.imshow(modal_error, cmap='Jet')
        plt.title(f"Erreur - Mode Modal\n"
                 f"PV={np.max(np.abs(modal_error)):.2f} nm, RMS={np.std(modal_error):.2f} nm")
        plt.colorbar(label="Erreur (nm)")
        plt.savefig(os.path.join(draw_dir, "error_modal.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Erreur parfaite
        plt.figure(figsize=(10, 8))
        plt.imshow(perfect_error, cmap='Jet')
        plt.title(f"Erreur - Parfait\n"
                 f"PV={np.max(np.abs(perfect_error)):.2f} nm, RMS={np.std(perfect_error):.2f} nm")
        plt.colorbar(label="Erreur (nm)")
        plt.savefig(os.path.join(draw_dir, "error_perfect.png"), dpi=150, bbox_inches='tight')
        plt.close()

    def run_full_simulation(self) -> Dict[str, Any]:
        """
        FR: Exécute la simulation complète avec tous les tirages.
            
            Effectue 40 simulations (4 amplitudes × 10 tirages) et génère :
            - Les images pour chaque tirage
            - Les graphiques globaux (erreur vs amplitude)
            - Les statistiques complètes
        
        EN: Runs the complete simulation with all draws.
            
            Performs 40 simulations (4 amplitudes × 10 draws) and generates:
            - Images for each draw
            - Global plots (error vs amplitude)
            - Complete statistics
        
        Returns:
            Dict: Résultats complets de la simulation.
        """
        logger.info(f"Début de la simulation complète ({TOTAL_SIMULATIONS} simulations)...")
        
        # Stockage des résultats
        all_results = {
            'metadata': {
                'date': datetime.now().isoformat(),
                'wavelength_nm': WAVELENGTH_NM,
                'beam_diameter_mm': BEAM_DIAMETER_MM,
                'microlens_array': {
                    'size': f"{NUM_MICROLENS_X}x{NUM_MICROLENS_Y}",
                    'side_um': MICROLENS_SIDE_UM,
                    'focal_length_mm': FOCAL_LENGTH_MM,
                    'material': MATERIAL_NAME,
                    'surface_quality_pv_nm': SURFACE_QUALITY_PV_NM,
                    'parallelism_arcsec': PARALLELISM_ARCSEC,
                    'scratch_dig': SCRATCH_DIG
                },
                'camera': {
                    'pixels': f"{NUM_PIXELS_X}x{NUM_PIXELS_Y}",
                    'pixel_size_um': PIXEL_SIZE_UM,
                    'size_mm': CAMERA_SIZE_MM,
                    'exposure_time_s': EXPOSURE_TIME_S
                }
            },
            'amplitudes': AMPLITUDES_RMS_NM,
            'num_draws_per_amplitude': NUM_DRAWS_PER_AMPLITUDE,
            'results': [],
            'statistics': {}
        }
        
        # Exécuter toutes les simulations
        for amplitude_rms in AMPLITUDES_RMS_NM:
            amplitude_results = []
            
            for draw_idx in range(NUM_DRAWS_PER_AMPLITUDE):
                result = self.simulate_single(amplitude_rms, draw_idx)
                amplitude_results.append(result)
                
                # Mise à jour de la progression
                total_done = (AMPLITUDES_RMS_NM.index(amplitude_rms) * NUM_DRAWS_PER_AMPLITUDE) + draw_idx + 1
                logger.info(f"Progression: {total_done}/{TOTAL_SIMULATIONS} "
                          f"({total_done/TOTAL_SIMULATIONS*100:.1f}%) - "
                          f"Amplitude={amplitude_rms}nm, Tirage={draw_idx}")
            
            all_results['results'].extend(amplitude_results)
            
            # Calculer les statistiques pour cette amplitude
            amp_stats = self._calculate_amplitude_statistics(amplitude_results)
            all_results['statistics'][f"amplitude_{amplitude_rms}nm"] = amp_stats
        
        # Calculer les statistiques globales
        all_results['global_statistics'] = self._calculate_global_statistics(all_results['results'])
        
        # Générer les graphiques globaux
        if self.display:
            self._generate_global_plots(all_results)
        
        # Sauvegarder les résultats complets
        self._save_results(all_results)
        
        logger.info(f"Simulation complète terminée !")
        logger.info(f"Résultats sauvegardés dans {self.output_dir}/")
        
        return all_results

    def _calculate_amplitude_statistics(self, results: List[Dict]) -> Dict:
        """FR: Calcule les statistiques pour une amplitude donnée."""
        zonal_errors_rms = [r['zonal']['error_rms'] for r in results]
        modal_errors_rms = [r['modal']['error_rms'] for r in results]
        perfect_errors_rms = [r['perfect']['error_rms'] for r in results]
        
        zonal_errors_pv = [r['zonal']['error_pv'] for r in results]
        modal_errors_pv = [r['modal']['error_pv'] for r in results]
        perfect_errors_pv = [r['perfect']['error_pv'] for r in results]
        
        true_pvs = [r['true_pv'] for r in results]
        true_rms = [r['true_rms'] for r in results]
        
        return {
            'num_draws': len(results),
            'true_phase': {
                'mean_pv': float(np.mean(true_pvs)),
                'std_pv': float(np.std(true_pvs)),
                'mean_rms': float(np.mean(true_rms)),
                'std_rms': float(np.std(true_rms))
            },
            'zonal': {
                'error_rms': {
                    'mean': float(np.mean(zonal_errors_rms)),
                    'std': float(np.std(zonal_errors_rms)),
                    'min': float(np.min(zonal_errors_rms)),
                    'max': float(np.max(zonal_errors_rms))
                },
                'error_pv': {
                    'mean': float(np.mean(zonal_errors_pv)),
                    'std': float(np.std(zonal_errors_pv)),
                    'min': float(np.min(zonal_errors_pv)),
                    'max': float(np.max(zonal_errors_pv))
                },
                'reconstruction': {
                    'mean_pv': float(np.mean([r['zonal']['pv'] for r in results])),
                    'std_pv': float(np.std([r['zonal']['pv'] for r in results])),
                    'mean_rms': float(np.mean([r['zonal']['rms'] for r in results])),
                    'std_rms': float(np.std([r['zonal']['rms'] for r in results]))
                }
            },
            'modal': {
                'error_rms': {
                    'mean': float(np.mean(modal_errors_rms)),
                    'std': float(np.std(modal_errors_rms)),
                    'min': float(np.min(modal_errors_rms)),
                    'max': float(np.max(modal_errors_rms))
                },
                'error_pv': {
                    'mean': float(np.mean(modal_errors_pv)),
                    'std': float(np.std(modal_errors_pv)),
                    'min': float(np.min(modal_errors_pv)),
                    'max': float(np.max(modal_errors_pv))
                },
                'reconstruction': {
                    'mean_pv': float(np.mean([r['modal']['pv'] for r in results])),
                    'std_pv': float(np.std([r['modal']['pv'] for r in results])),
                    'mean_rms': float(np.mean([r['modal']['rms'] for r in results])),
                    'std_rms': float(np.std([r['modal']['rms'] for r in results]))
                }
            },
            'perfect': {
                'error_rms': {
                    'mean': float(np.mean(perfect_errors_rms)),
                    'std': float(np.std(perfect_errors_rms)),
                    'min': float(np.min(perfect_errors_rms)),
                    'max': float(np.max(perfect_errors_rms))
                },
                'error_pv': {
                    'mean': float(np.mean(perfect_errors_pv)),
                    'std': float(np.std(perfect_errors_pv)),
                    'min': float(np.min(perfect_errors_pv)),
                    'max': float(np.max(perfect_errors_pv))
                }
            }
        }

    def _calculate_global_statistics(self, results: List[Dict]) -> Dict:
        """FR: Calcule les statistiques globales."""
        # Extraire toutes les données
        amplitudes = [r['amplitude_rms'] for r in results]
        draws = [r['draw_idx'] for r in results]
        
        zonal_errors_rms = [r['zonal']['error_rms'] for r in results]
        modal_errors_rms = [r['modal']['error_rms'] for r in results]
        perfect_errors_rms = [r['perfect']['error_rms'] for r in results]
        
        true_pvs = [r['true_pv'] for r in results]
        true_rms = [r['true_rms'] for r in results]
        
        # Créer une matrice amplitude × draw
        unique_amplitudes = sorted(set(amplitudes))
        
        return {
            'all_amplitudes': unique_amplitudes,
            'all_draws': list(range(NUM_DRAWS_PER_AMPLITUDE)),
            'zonal': {
                'errors_rms': zonal_errors_rms,
                'errors_pv': [r['zonal']['error_pv'] for r in results],
                'reconstruction_pv': [r['zonal']['pv'] for r in results],
                'reconstruction_rms': [r['zonal']['rms'] for r in results]
            },
            'modal': {
                'errors_rms': modal_errors_rms,
                'errors_pv': [r['modal']['error_pv'] for r in results],
                'reconstruction_pv': [r['modal']['pv'] for r in results],
                'reconstruction_rms': [r['modal']['rms'] for r in results]
            },
            'perfect': {
                'errors_rms': perfect_errors_rms,
                'errors_pv': [r['perfect']['error_pv'] for r in results],
                'reconstruction_pv': [r['perfect']['pv'] for r in results],
                'reconstruction_rms': [r['perfect']['rms'] for r in results]
            },
            'true_phase': {
                'pv': true_pvs,
                'rms': true_rms
            }
        }

    def _generate_global_plots(self, all_results: Dict) -> None:
        """FR: Génère les graphiques globaux."""
        
        # 1. Erreur RMS en fonction de l'amplitude (moyenne ± écart-type)
        self._plot_error_vs_amplitude(all_results, error_type='rms')
        
        # 2. Erreur PV en fonction de l'amplitude
        self._plot_error_vs_amplitude(all_results, error_type='pv')
        
        # 3. Comparaison zonal vs modal
        self._plot_zonal_vs_modal(all_results)
        
        # 4. Tracé du comportement de l'erreur pour chaque tirage
        self._plot_individual_draws(all_results)
        
        # 5. Histogrammes des erreurs
        self._plot_error_histograms(all_results)

    def _plot_error_vs_amplitude(self, all_results: Dict, error_type: str = 'rms') -> None:
        """FR: Trace l'erreur en fonction de l'amplitude."""
        amplitudes = all_results['amplitudes']
        stats = all_results['statistics']
        
        # Extraire les données
        zonal_means = []
        zonal_stds = []
        modal_means = []
        modal_stds = []
        perfect_means = []
        perfect_stds = []
        
        for amp in amplitudes:
            amp_key = f"amplitude_{amp}nm"
            zonal_means.append(stats[amp_key]['zonal'][f'error_{error_type}']['mean'])
            zonal_stds.append(stats[amp_key]['zonal'][f'error_{error_type}']['std'])
            
            modal_means.append(stats[amp_key]['modal'][f'error_{error_type}']['mean'])
            modal_stds.append(stats[amp_key]['modal'][f'error_{error_type}']['std'])
            
            perfect_means.append(stats[amp_key]['perfect'][f'error_{error_type}']['mean'])
            perfect_stds.append(stats[amp_key]['perfect'][f'error_{error_type}']['std'])
        
        # Créer le graphique
        plt.figure(figsize=(12, 8))
        
        # Zonal
        plt.errorbar(amplitudes, zonal_means, yerr=zonal_stds,
                    label='Mode Zonal', marker='o', capsize=5)
        
        # Modal
        plt.errorbar(amplitudes, modal_means, yerr=modal_stds,
                    label='Mode Modal', marker='s', capsize=5)
        
        # Parfait
        plt.errorbar(amplitudes, perfect_means, yerr=perfect_stds,
                    label='Parfait', marker='^', capsize=5)
        
        plt.xlabel("Amplitude RMS des aberrations (nm)")
        plt.ylabel(f"Erreur {error_type.upper()} moyenne (nm)")
        plt.title(f"Erreur {error_type.upper()} en fonction de l'amplitude des aberrations")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"error_{error_type}_vs_amplitude.png"), dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_zonal_vs_modal(self, all_results: Dict) -> None:
        """FR: Compare les performances zonal vs modal."""
        amplitudes = all_results['amplitudes']
        stats = all_results['statistics']
        
        # Extraire les données
        zonal_rms = []
        modal_rms = []
        zonal_pv = []
        modal_pv = []
        
        for amp in amplitudes:
            amp_key = f"amplitude_{amp}nm"
            zonal_rms.append(stats[amp_key]['zonal']['error_rms']['mean'])
            modal_rms.append(stats[amp_key]['modal']['error_rms']['mean'])
            zonal_pv.append(stats[amp_key]['zonal']['error_pv']['mean'])
            modal_pv.append(stats[amp_key]['modal']['error_pv']['mean'])
        
        # Créer le graphique
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # RMS
        axes[0].plot(amplitudes, zonal_rms, 'o-', label='Mode Zonal')
        axes[0].plot(amplitudes, modal_rms, 's-', label='Mode Modal')
        axes[0].set_xlabel("Amplitude RMS des aberrations (nm)")
        axes[0].set_ylabel("Erreur RMS moyenne (nm)")
        axes[0].set_title("Comparaison Mode Zonal vs Modal - Erreur RMS")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # PV
        axes[1].plot(amplitudes, zonal_pv, 'o-', label='Mode Zonal')
        axes[1].plot(amplitudes, modal_pv, 's-', label='Mode Modal')
        axes[1].set_xlabel("Amplitude RMS des aberrations (nm)")
        axes[1].set_ylabel("Erreur PV moyenne (nm)")
        axes[1].set_title("Comparaison Mode Zonal vs Modal - Erreur PV")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "zonal_vs_modal_comparison.png"), dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_individual_draws(self, all_results: Dict) -> None:
        """FR: Trace le comportement de l'erreur pour chaque tirage."""
        results_list = all_results['results']
        
        # Extraire les données
        amplitudes = [r['amplitude_rms'] for r in results_list]
        draws = [r['draw_idx'] for r in results_list]
        zonal_errors = [r['zonal']['error_rms'] for r in results_list]
        modal_errors = [r['modal']['error_rms'] for r in results_list]
        perfect_errors = [r['perfect']['error_rms'] for r in results_list]
        
        # Créer un index unique pour chaque tirage
        indices = np.arange(len(results_list))
        
        # Créer le graphique
        plt.figure(figsize=(14, 8))
        
        # Tracer chaque tirage
        for amp in all_results['amplitudes']:
            amp_indices = [i for i, a in enumerate(amplitudes) if a == amp]
            amp_zonal = [zonal_errors[i] for i in amp_indices]
            amp_modal = [modal_errors[i] for i in amp_indices]
            amp_perfect = [perfect_errors[i] for i in amp_indices]
            
            # Tracer avec des couleurs différentes pour chaque amplitude
            color = plt.cm.viridis(AMPLITUDES_RMS_NM.index(amp) / len(AMPLITUDES_RMS_NM))
            
            plt.scatter(amp_indices, amp_zonal, color=color, marker='o', label=f'{amp}nm Zonal' if amp == AMPLITUDES_RMS_NM[0] else "")
            plt.scatter(amp_indices, amp_modal, color=color, marker='s', label=f'{amp}nm Modal' if amp == AMPLITUDES_RMS_NM[0] else "")
            plt.scatter(amp_indices, amp_perfect, color=color, marker='^', label=f'{amp}nm Parfait' if amp == AMPLITUDES_RMS_NM[0] else "")
        
        # Ajouter les légendes
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.xlabel("Index du tirage (0-39)")
        plt.ylabel("Erreur RMS (nm)")
        plt.title("Comportement de l'erreur pour chaque tirage")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "individual_draws_errors.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Version avec amplitude sur l'axe X
        plt.figure(figsize=(14, 8))
        
        for i, (amp, draw, zonal, modal, perfect) in enumerate(zip(amplitudes, draws, zonal_errors, modal_errors, perfect_errors)):
            offset = draw * 0.1  # Décaler légèrement pour éviter le chevauchement
            plt.scatter(amp + offset, zonal, color='blue', marker='o', alpha=0.6, label='Zonal' if i == 0 else "")
            plt.scatter(amp + offset, modal, color='red', marker='s', alpha=0.6, label='Modal' if i == 0 else "")
            plt.scatter(amp + offset, perfect, color='green', marker='^', alpha=0.6, label='Parfait' if i == 0 else "")
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.xlabel("Amplitude RMS des aberrations (nm)")
        plt.ylabel("Erreur RMS (nm)")
        plt.title("Comportement de l'erreur pour chaque tirage (par amplitude)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "individual_draws_errors_by_amplitude.png"), dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_error_histograms(self, all_results: Dict) -> None:
        """FR: Trace les histogrammes des erreurs."""
        results_list = all_results['results']
        
        # Extraire les erreurs
        zonal_errors = [r['zonal']['error_rms'] for r in results_list]
        modal_errors = [r['modal']['error_rms'] for r in results_list]
        perfect_errors = [r['perfect']['error_rms'] for r in results_list]
        
        # Créer le graphique
        plt.figure(figsize=(14, 8))
        
        plt.hist(zonal_errors, bins=20, alpha=0.5, label='Mode Zonal')
        plt.hist(modal_errors, bins=20, alpha=0.5, label='Mode Modal')
        plt.hist(perfect_errors, bins=20, alpha=0.5, label='Parfait')
        
        plt.xlabel("Erreur RMS (nm)")
        plt.ylabel("Fréquence")
        plt.title("Histogramme des erreurs RMS (tous tirages confondus)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "error_histogram.png"), dpi=150, bbox_inches='tight')
        plt.close()

    def _save_results(self, all_results: Dict) -> None:
        """FR: Sauvegarde les résultats complets."""
        
        # 1. Sauvegarder en JSON
        json_file = os.path.join(self.output_dir, "simulation_results.json")
        with open(json_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Résultats sauvegardés dans {json_file}")
        
        # 2. Sauvegarder les données brutes en NPY
        np.save(os.path.join(self.output_dir, "all_results.npy"), all_results['results'])
        
        # 3. Sauvegarder les statistiques
        stats_file = os.path.join(self.output_dir, "simulation_statistics.json")
        with open(stats_file, 'w') as f:
            json.dump(all_results['statistics'], f, indent=2)
        
        # 4. Sauvegarder les paramètres
        params = {
            'wavelength_nm': WAVELENGTH_NM,
            'beam_diameter_mm': BEAM_DIAMETER_MM,
            'microlens_side_um': MICROLENS_SIDE_UM,
            'focal_length_mm': FOCAL_LENGTH_MM,
            'thickness_mm': THICKNESS_MM,
            'material': MATERIAL_NAME,
            'surface_quality_pv_nm': SURFACE_QUALITY_PV_NM,
            'parallelism_arcsec': PARALLELISM_ARCSEC,
            'scratch_dig': SCRATCH_DIG,
            'camera_pixel_size_um': PIXEL_SIZE_UM,
            'camera_size_mm': CAMERA_SIZE_MM,
            'exposure_time_s': EXPOSURE_TIME_S,
            'amplitudes_nm': AMPLITUDES_RMS_NM,
            'num_draws_per_amplitude': NUM_DRAWS_PER_AMPLITUDE,
            'wyant_modes': [(n, m) for n, m in WYANT_MODES]
        }
        
        params_file = os.path.join(self.output_dir, "simulation_parameters.json")
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)
        
        logger.info(f"Paramètres sauvegardés dans {params_file}")


# =============================================================================
# FONCTION UTILITAIRE POUR CRÉER UNE SIMULATION
# =============================================================================

def create_simulation(name: str = "SH_Simulation",
                      output_dir: str = "simulation_output",
                      display: bool = True) -> ShackHartmannSimulation:
    """
    FR: Fabrique une simulation Shack-Hartmann complète.
    
    EN: Factory function to create a complete Shack-Hartmann simulation.
    
    Args:
        name (str): Nom de la simulation.
        output_dir (str): Répertoire de sortie.
        display (bool): Afficher les images.
    
    Returns:
        ShackHartmannSimulation: La simulation créée.
    """
    return ShackHartmannSimulation(
        name=name,
        output_dir=output_dir,
        display=display
    )


# =============================================================================
# EXÉCUTION PRINCIPALE
# =============================================================================

if __name__ == "__main__":
    # Vérifier que tous les modules sont disponibles
    missing_modules = []
    if not BEAM_AVAILABLE:
        missing_modules.append("Beam.py")
    if not OPTIQUES_AVAILABLE:
        missing_modules.append("Optiques.py")
    if not MICROSTRUCTURE_AVAILABLE:
        missing_modules.append("Microstructure.py")
    if not CAMERA_AVAILABLE:
        missing_modules.append("Camera.py")
    if not PROPAGATION_AVAILABLE:
        missing_modules.append("Propagation.py")
    if not SHACK_HARTMANN_AVAILABLE:
        missing_modules.append("Shack_Hartmann.py")
    if not SOUTHWELL_AVAILABLE:
        missing_modules.append("Southwell.py")
    
    if missing_modules:
        logger.error(f"Modules manquants: {', '.join(missing_modules)}")
        logger.error("Impossible d'exécuter la simulation.")
        exit(1)
    
    # Créer et exécuter la simulation
    logger.info("="*80)
    logger.info("DEBUT DE LA SIMULATION SHACK-HARTMANN COMPLÈTE")
    logger.info("="*80)
    
    simulation = create_simulation(
        name="SH_Simulation_Complete",
        output_dir="examples/simulation_output",
        display=True
    )
    
    # Exécuter la simulation complète
    results = simulation.run_full_simulation()
    
    logger.info("="*80)
    logger.info("SIMULATION TERMINÉE AVEC SUCCÈS")
    logger.info("="*80)
    logger.info(f"Résultats sauvegardés dans: examples/simulation_output/")
    logger.info("\nFichiers générés:")
    logger.info("  - simulation_results.json (résultats complets)")
    logger.info("  - simulation_statistics.json (statistiques par amplitude)")
    logger.info("  - simulation_parameters.json (paramètres de la simulation)")
    logger.info("  - all_results.npy (données brutes)")
    logger.info("  - error_rms_vs_amplitude.png (graphique)")
    logger.info("  - error_pv_vs_amplitude.png (graphique)")
    logger.info("  - zonal_vs_modal_comparison.png (graphique)")
    logger.info("  - individual_draws_errors.png (graphique)")
    logger.info("  - individual_draws_errors_by_amplitude.png (graphique)")
    logger.info("  - error_histogram.png (graphique)")
    logger.info("  - Dossiers amplitude_*/draw_* (images par tirage)")

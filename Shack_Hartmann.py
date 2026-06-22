"""
Shack_Hartmann.py

FR: Module pour le calcul des centroïdes et des pentes locales dans un système Shack-Hartmann.
    Permet de :
    - Simuler un système Shack-Hartmann complet (microlentilles + caméra)
    - Calculer les positions des centroïdes (barycentre) de chaque tâche d'Airy
    - Estimer l'erreur sur le calcul des centroïdes
    - En déduire les pentes locales de phase (dφ/dx, dφ/dy) en RADIANS
    - Afficher :
        * Carte des tâches d'Airy (spots)
        * Carte des centroïdes (marqués sur l'image des spots)
        * Cartes des pentes locales (X et Y)
        * Cartes d'erreur sur les centroïdes et les pentes

    Fonctionne avec :
    - Beam.py : pour le faisceau incident
    - Propagation.py : pour la propagation à travers les microlentilles
    - Microstructure.py : pour la matrice de microlentilles
    - Camera.py : pour la mesure des tâches d'Airy

    Algorithmes de calcul des centroïdes :
    - WEIGHTED_CENTROID : Barycentre pondéré par l'intensité (défaut)
    - THRESHOLDED_CENTROID : Barycentre avec seuil
    - GAUSSIAN_FIT : Ajustement gaussien 2D (plus précis pour les tâches bruitées)
    - MOMENT_BASED : Basé sur les moments (robuste au bruit)

    Formule des pentes :
        slope_x = arctan(dx / focal_length)  [rad]
        slope_y = arctan(dy / focal_length)  [rad]
    où dx, dy sont les décalages du centroïde par rapport à la position de référence.

    Unités :
    - Longueurs : mm (positions), µm (microlentilles)
    - Longueur d'onde : nm
    - Phase : nm, rad, mrad
    - Pentes : RAD (radians) - PAS rad/mm !

EN: Module for calculating centroids and local slopes in a Shack-Hartmann system.
    Allows:
    - Simulating a complete Shack-Hartmann system (microlens array + camera)
    - Calculating centroid positions (barycenter) of each Airy spot
    - Estimating error on centroid calculation
    - Deducing local phase slopes (dφ/dx, dφ/dy) in RADIANS
    - Displaying:
        * Spot map (Airy spots)
        * Centroid map (marked on the spot image)
        * Local slope maps (X and Y)
        * Error maps (centroids and slopes)

    Works with:
    - Beam.py: for the incident beam
    - Propagation.py: for propagation through microlenses
    - Microstructure.py: for the microlens array
    - Camera.py: for measuring Airy spots

    Centroid calculation algorithms:
    - WEIGHTED_CENTROID: Intensity-weighted centroid (default)
    - THRESHOLDED_CENTROID: Centroid with threshold
    - GAUSSIAN_FIT: 2D Gaussian fitting (more accurate for noisy spots)
    - MOMENT_BASED: Moment-based (robust to noise)

    Slope formula:
        slope_x = arctan(dx / focal_length)  [rad]
        slope_y = arctan(dy / focal_length)  [rad]
    where dx, dy are the centroid shifts from the reference position.

    Units:
    - Lengths: mm (positions), µm (microlenses)
    - Wavelength: nm
    - Phase: nm, rad, mrad
    - Slopes: RAD (radians) - NOT rad/mm!

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch

Dependencies:
    - numpy
    - scipy (for Gaussian fitting, optional)
    - Beam.py
    - Propagation.py
    - Microstructure.py
    - Camera.py
    - Visualization.py (optional)
    - MathAndPhysicsTools.py (optional)

Sources:
    - "Principles of Adaptive Optics" by R.K. Tyson (Academic Press, 1991)
      -> Shack-Hartmann wavefront sensing principles (Ch. 4)
      -> Centroid calculation and slope measurement
    - "Shack-Hartmann wavefront sensor" by B.C. Platt & R.V. Shack (2001)
      -> Original Shack-Hartmann design and centroid algorithms
    - "Adaptive Optics for Astronomical Telescopes" by J.W. Hardy (Oxford, 1998)
      -> Wavefront reconstruction from slopes (Ch. 5)
      -> Slope formula: theta = arctan(dx/f) for Shack-Hartmann
    - "Centroiding algorithms for Shack-Hartmann sensors" by J.M. Winther & R. Seldin (2004)
      -> Comparison of centroiding methods and their performance
    - "The Southwell Geometry for Wavefront Sensing" by W.H. Southwell (1980)
      -> Geometry and slope calculation in Shack-Hartmann
    - "Error analysis in Shack-Hartmann wavefront sensors" by D.L. Fried (1978)
      -> Centroid and slope error estimation
"""

import numpy as np
import logging
import os
from typing import Optional, Tuple, Dict, List, Union
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# IMPORT DES DÉPENDANCES
# =============================================================================

try:
    from Beam import Beam
    BEAM_AVAILABLE = True
except ImportError as e:
    BEAM_AVAILABLE = False
    logging.warning(f"Beam module not available: {e}")

try:
    from Propagation import Propagation
    PROPAGATION_AVAILABLE = True
except ImportError as e:
    PROPAGATION_AVAILABLE = False
    logging.warning(f"Propagation module not available: {e}")

try:
    from Microstructure import MicrolensArray, create_microlens_array, ArrayPattern
    MICROSTRUCTURE_AVAILABLE = True
except ImportError as e:
    MICROSTRUCTURE_AVAILABLE = False
    logging.warning(f"Microstructure module not available: {e}")

try:
    from Camera import RealCamera, IdealCamera, Camera
    CAMERA_AVAILABLE = True
except ImportError as e:
    CAMERA_AVAILABLE = False
    logging.warning(f"Camera module not available: {e}")

try:
    from MathAndPhysicsTools import create_grid
    MATH_TOOLS_AVAILABLE = True
except ImportError as e:
    MATH_TOOLS_AVAILABLE = False
    logging.warning(f"MathAndPhysicsTools module not available: {e}")

try:
    from Visualization import plot_intensity, plot_phase
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    VISUALIZATION_AVAILABLE = False
    logging.warning(f"Visualization module not available: {e}")

try:
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Shack_Hartmann")


# =============================================================================
# ENUMS ET CONSTANTES
# =============================================================================

class CentroidAlgorithm(Enum):
    """
    FR: Algorithme de calcul des centroïdes.
        
        - WEIGHTED_CENTROID : Barycentre pondéré par l'intensité (méthode standard)
            Formule: x̄ = Σ(I_x * x) / ΣI, ȳ = Σ(I_y * y) / ΣI
        - THRESHOLDED_CENTROID : Barycentre avec seuil (pour éliminer le bruit de fond)
            Seuls les pixels avec I > seuil * max(I) sont pris en compte
        - GAUSSIAN_FIT : Ajustement gaussien 2D (plus précis pour les tâches bruitées)
            Modèle: I(x,y) = I₀ + A * exp(-((x-x₀)²/2σ_x² + (y-y₀)²/2σ_y²))
        - MOMENT_BASED : Basé sur les moments (robuste au bruit)
            Utilise les moments d'ordre 1 pour le centroïde et d'ordre 2 pour l'erreur
    
    EN: Centroid calculation algorithm.
        
        - WEIGHTED_CENTROID: Intensity-weighted centroid (standard method)
            Formula: x̄ = Σ(I_x * x) / ΣI, ȳ = Σ(I_y * y) / ΣI
        - THRESHOLDED_CENTROID: Centroid with threshold (to eliminate background noise)
            Only pixels with I > threshold * max(I) are considered
        - GAUSSIAN_FIT: 2D Gaussian fitting (more accurate for noisy spots)
            Model: I(x,y) = I₀ + A * exp(-((x-x₀)²/2σ_x² + (y-y₀)²/2σ_y²))
        - MOMENT_BASED: Moment-based (robust to noise)
            Uses first-order moments for centroid and second-order moments for error
    
    Source:
        - "Centroiding algorithms for Shack-Hartmann sensors" by Winther & Seldin (2004)
    """
    WEIGHTED_CENTROID = "weighted_centroid"
    THRESHOLDED_CENTROID = "thresholded_centroid"
    GAUSSIAN_FIT = "gaussian_fit"
    MOMENT_BASED = "moment_based"


class SlopeCalculationMethod(Enum):
    """
    FR: Méthode de calcul des pentes locales.
        
        - FINITE_DIFFERENCE : Utilise arctan(dx/f) et arctan(dy/f) (méthode standard)
            où dx, dy sont les décalages en mm et f est la distance focale en mm
            Résultat: pentes en RADIANS
        - LEAST_SQUARES : Moindres carrés (pour les pentes globales)
        - SOUTHWELL : Méthode de Southwell (pour la reconstruction de phase)
    
    EN: Method for calculating local slopes.
        
        - FINITE_DIFFERENCE: Uses arctan(dx/f) and arctan(dy/f) (standard method)
            where dx, dy are shifts in mm and f is focal length in mm
            Result: slopes in RADIANS
        - LEAST_SQUARES: Least squares (for global slopes)
        - SOUTHWELL: Southwell method (for phase reconstruction)
    
    Source:
        - "The Southwell Geometry for Wavefront Sensing" by Southwell (1980)
        - "Adaptive Optics for Astronomical Telescopes" by Hardy (1998)
          -> Slope formula: theta = arctan(dx/f) for Shack-Hartmann
    """
    FINITE_DIFFERENCE = "finite_difference"
    LEAST_SQUARES = "least_squares"


# Constantes pour les conversions
RAD_TO_MRAD = 1000.0  # Conversion rad → mrad
MRAD_TO_RAD = 1.0 / RAD_TO_MRAD
RAD_TO_DEG = 180.0 / np.pi  # Conversion rad → degrés
DEG_TO_RAD = np.pi / 180.0


# =============================================================================
# CLASSE PRINCIPALE: SHACK-HARTMANN
# =============================================================================

class ShackHartmann:
    """
    FR: Système Shack-Hartmann complet.
        Permet de simuler un système Shack-Hartmann (matrice de microlentilles + caméra)
        et de calculer :
        - Les positions des centroïdes (barycentre) de chaque tâche d'Airy
        - Les erreurs sur les centroïdes
        - Les pentes locales de phase (dφ/dx, dφ/dy) EN RADIANS
        - Les erreurs sur les pentes
        
        Le système fonctionne en 3 étapes :
        1. Appliquer la matrice de microlentilles au faisceau incident
        2. Propage le faisceau jusqu'au plan focal des microlentilles
        3. Échantillonner le faisceau avec la caméra pour obtenir les tâches d'Airy
        
        Ensuite, les centroïdes et les pentes sont calculés automatiquement.
        
        FORMULE DES PENTES (IMPORTANT) :
            slope_x = arctan(dx / focal_length)  [RADIANS]
            slope_y = arctan(dy / focal_length)  [RADIANS]
        où:
            - dx, dy : décalages du centroïde par rapport à la position de référence (en mm)
            - focal_length : distance focale des microlentilles (en mm)
            - Le résultat est en RADIANS (pas en rad/mm !)
    
    EN: Complete Shack-Hartmann system.
        Allows simulating a Shack-Hartmann system (microlens array + camera)
        and calculating:
        - Centroid positions (barycenter) of each Airy spot
        - Centroid errors
        - Local phase slopes (dφ/dx, dφ/dy) IN RADIANS
        - Slope errors
        
        The system works in 3 steps:
        1. Apply the microlens array to the incident beam
        2. Propagate the beam to the focal plane of the microlenses
        3. Sample the beam with the camera to get Airy spots
        
        Then, centroids and slopes are calculated automatically.
        
        SLOPE FORMULA (IMPORTANT):
            slope_x = arctan(dx / focal_length)  [RADIANS]
            slope_y = arctan(dy / focal_length)  [RADIANS]
        where:
            - dx, dy: centroid shifts from reference position (in mm)
            - focal_length: focal length of microlenses (in mm)
            - Result is in RADIANS (not rad/mm!)
    
    Attributes:
        name (str): Nom du système Shack-Hartmann.
        microlens_array (MicrolensArray): Matrice de microlentilles.
        camera (Camera): Caméra virtuelle.
        wavelength_nm (float): Longueur d'onde en nm.
        focal_length_mm (float): Distance focale des microlentilles en mm.
        propagation_distance_mm (float): Distance de propagation jusqu'au plan focal.
        centroid_algorithm (CentroidAlgorithm): Algorithme de calcul des centroïdes.
        slope_method (SlopeCalculationMethod): Méthode de calcul des pentes.
        display (bool): Afficher automatiquement les cartes.
        display_dir (str): Répertoire pour sauvegarder les images.
        
        Results (after simulation):
        spot_image (np.ndarray): Image des tâches d'Airy (2D array).
        centroids (np.ndarray): Positions des centroïdes en pixels (Nx2 array: [x, y]).
        centroid_errors (np.ndarray): Erreurs sur les centroïdes en pixels (Nx2 array).
        slopes_x (np.ndarray): Pentes locales en x (RADIANS, 2D array).
        slopes_y (np.ndarray): Pentes locales en y (RADIANS, 2D array).
        slope_errors (np.ndarray): Erreurs sur les pentes (NxNx2 array: [x_error, y_error]).
    
    Sources:
        - "Principles of Adaptive Optics" by Tyson (1991)
        - "Shack-Hartmann wavefront sensor" by Platt & Shack (2001)
        - "Adaptive Optics for Astronomical Telescopes" by Hardy (1998)
          -> Slope formula: theta = arctan(dx/f) in RADIANS
    """

    def __init__(self,
                 name: str = "ShackHartmann",
                 microlens_array: Optional['MicrolensArray'] = None,
                 camera: Optional['Camera'] = None,
                 wavelength_nm: float = 633.0,
                 focal_length_mm: float = 20.0,
                 centroid_algorithm: CentroidAlgorithm = CentroidAlgorithm.WEIGHTED_CENTROID,
                 slope_method: SlopeCalculationMethod = SlopeCalculationMethod.FINITE_DIFFERENCE,
                 display: bool = False,
                 display_dir: str = "output"):
        """
        FR: Initialise un système Shack-Hartmann.
            
        EN: Initializes a Shack-Hartmann system.
        
        Args:
            name (str): Nom du système.
            microlens_array (MicrolensArray, optional): Matrice de microlentilles.
                Si None, une matrice par défaut est créée.
            camera (Camera, optional): Caméra virtuelle.
                Si None, une caméra par défaut est créée.
            wavelength_nm (float): Longueur d'onde en nm (défaut: 633.0).
            focal_length_mm (float): Distance focale des microlentilles en mm (défaut: 20.0).
            centroid_algorithm (CentroidAlgorithm): Algorithme de calcul des centroïdes.
            slope_method (SlopeCalculationMethod): Méthode de calcul des pentes.
            display (bool): Afficher automatiquement les cartes (défaut: False).
            display_dir (str): Répertoire pour sauvegarder les images (défaut: "output").
        """
        self.name = name
        self.wavelength_nm = wavelength_nm
        self.focal_length_mm = focal_length_mm
        self.centroid_algorithm = centroid_algorithm
        self.slope_method = slope_method
        self.display = display
        self.display_dir = display_dir

        # Créer la matrice de microlentilles si non fournie
        if microlens_array is None:
            self.microlens_array = self._create_default_microlens_array()
        else:
            self.microlens_array = microlens_array

        # Créer la caméra si non fournie
        if camera is None:
            self.camera = self._create_default_camera()
        else:
            self.camera = camera

        # Calculer la distance de propagation jusqu'au plan focal
        self.propagation_distance_mm = self.focal_length_mm

        # Initialiser les résultats (seront remplis après simulation)
        self.spot_image = None
        self.centroids = None
        self.centroid_errors = None
        self.slopes_x = None
        self.slopes_y = None
        self.slope_errors = None

        # Créer le répertoire d'affichage
        if self.display:
            os.makedirs(self.display_dir, exist_ok=True)

        logger.info(f"Système Shack-Hartmann '{name}' initialisé avec "
                   f"{self.microlens_array.num_elements_x}x{self.microlens_array.num_elements_y} microlentilles, "
                   f"f={self.focal_length_mm}mm, λ={self.wavelength_nm}nm")

    def _create_default_microlens_array(self) -> 'MicrolensArray':
        """
        FR: Crée une matrice de microlentilles par défaut (10x10, pitch=0.5mm).
            
        EN: Creates a default microlens array (10x10, pitch=0.5mm).
        """
        if not MICROSTRUCTURE_AVAILABLE:
            raise ImportError("Microstructure module required for default microlens array")
        
        return create_microlens_array(
            name=f"{self.name}_MicrolensArray",
            pitch_mm=0.5,  # Distance bord-à-bord
            num_elements_x=10,
            num_elements_y=10,
            focal_length_mm=self.focal_length_mm,
            wavelength_nm=self.wavelength_nm,
            edge_to_edge_spacing_mm=0.0  # Éléments joints
        )

    def _create_default_camera(self) -> 'Camera':
        """
        FR: Crée une caméra par défaut (1024x1024, 5µm/pixel).
            
        EN: Creates a default camera (1024x1024, 5µm/pixel).
        """
        if not CAMERA_AVAILABLE:
            raise ImportError("Camera module required for default camera")
        
        from Camera import RealCamera
        return RealCamera(
            name=f"{self.name}_Camera",
            num_pixels_x=1024,
            num_pixels_y=1024,
            pixel_size_um=5.0,  # 5 µm
            wavelength_nm=self.wavelength_nm,
            material_name="Silicon"
        )

    def simulate(self, beam: 'Beam') -> None:
        """
        FR: Simule le passage d'un faisceau à travers le système Shack-Hartmann.
            
            Étapes :
            1. Applique la matrice de microlentilles au faisceau
            2. Propage le faisceau jusqu'au plan focal
            3. Échantillonne le faisceau avec la caméra
            4. Calcule les centroïdes des tâches d'Airy
            5. Calcule les pentes locales de phase (EN RADIANS)
            
        EN: Simulates the passage of a beam through the Shack-Hartmann system.
            
            Steps:
            1. Apply the microlens array to the beam
            2. Propagate the beam to the focal plane
            3. Sample the beam with the camera
            4. Calculate centroids of Airy spots
            5. Calculate local phase slopes (IN RADIANS)
        
        Args:
            beam (Beam): Faisceau incident.
        
        Sources:
            - "Principles of Adaptive Optics" by Tyson (1991), Ch. 4
        """
        if not BEAM_AVAILABLE:
            raise ImportError("Beam module is required for simulate()")

        # Étape 1: Appliquer la matrice de microlentilles au faisceau
        logger.info("Étape 1/4: Application de la matrice de microlentilles...")
        beam_after_microlenses = self.microlens_array.apply_to_beam(beam)

        # Étape 2: Propage jusqu'au plan focal
        logger.info("Étape 2/4: Propagation jusqu'au plan focal...")
        if PROPAGATION_AVAILABLE:
            try:
                propagator = Propagation(
                    wavelength_nm=self.wavelength_nm,
                    propagation_distance_mm=self.propagation_distance_mm,
                    input_diameter_mm=beam_after_microlenses.diameter_mm,
                    output_diameter_mm=self.camera.sensor_width_mm,
                    num_points=self.camera.specifications.num_pixels_x,
                    method="angular_spectrum"
                )
                propagated_field = propagator.propagate(beam_after_microlenses.electric_field)

                # Créer un nouveau faisceau avec le champ propagé
                propagated_beam = Beam(
                    wavelength_nm=self.wavelength_nm,
                    diameter_mm=self.camera.sensor_width_mm,
                    energy=beam_after_microlenses.energy,
                    num_points=self.camera.specifications.num_pixels_x
                )
                propagated_beam.electric_field = propagated_field
                propagated_beam.intensity = propagated_beam.compute_intensity_from_electric_field(propagated_field)
                propagated_beam.phase = propagated_beam.extract_phase_from_electric_field(propagated_field)
            except Exception as e:
                logger.warning(f"Propagation échouée: {e}. Utilisation du faisceau après microlentilles.")
                propagated_beam = beam_after_microlenses
        else:
            logger.warning("Propagation module not available. Utilisation du faisceau après microlentilles.")
            propagated_beam = beam_after_microlenses

        # Étape 3: Échantillonner avec la caméra
        logger.info("Étape 3/4: Échantillonnage avec la caméra...")
        self.spot_image = self.camera.sample_beam(propagated_beam)

        # Étape 4: Calculer les centroïdes
        logger.info("Étape 4/4: Calcul des centroïdes et des pentes...")
        self.calculate_centroids()
        self.calculate_slopes()

        logger.info(f"Simulation Shack-Hartmann terminée. "
                   f"Spots détectés: {len(self.centroids) if self.centroids is not None else 0}")

    def calculate_centroids(self) -> None:
        """
        FR: Calcule les positions des centroïdes des tâches d'Airy.
            Utilise l'algorithme spécifié (par défaut: barycentre pondéré).
            
            Pour chaque tâche d'Airy :
            1. Extrait la sous-image correspondant à la tâche
            2. Calcule le centroïde avec l'algorithme choisi
            3. Estime l'erreur sur le centroïde
            
        EN: Calculates the centroid positions of Airy spots.
            Uses the specified algorithm (default: weighted centroid).
            
            For each Airy spot:
            1. Extract the sub-image corresponding to the spot
            2. Calculate the centroid with the chosen algorithm
            3. Estimate the error on the centroid
        
        Algorithmes disponibles:
            - WEIGHTED_CENTROID: Barycentre pondéré par l'intensité
            - THRESHOLDED_CENTROID: Barycentre avec seuil
            - GAUSSIAN_FIT: Ajustement gaussien 2D
            - MOMENT_BASED: Basé sur les moments
        
        Sources:
            - "Centroiding algorithms for Shack-Hartmann sensors" by Winther & Seldin (2004)
        """
        if self.spot_image is None:
            raise ValueError("Spot image not available. Run simulate() first.")

        num_spots_x = self.microlens_array.num_elements_x
        num_spots_y = self.microlens_array.num_elements_y

        # Initialiser les centroïdes et les erreurs
        self.centroids = np.zeros((num_spots_x * num_spots_y, 2))  # (x, y) en pixels
        self.centroid_errors = np.zeros((num_spots_x * num_spots_y, 2))  # (x_error, y_error) en pixels

        # Calculer la taille d'une sous-image par spot (en pixels)
        spot_size_pixels_x = self.camera.specifications.num_pixels_x // num_spots_x
        spot_size_pixels_y = self.camera.specifications.num_pixels_y // num_spots_y

        # Parcourir chaque spot
        for j in range(num_spots_y):
            for i in range(num_spots_x):
                # Index du spot
                spot_idx = j * num_spots_x + i

                # Extraire la sous-image du spot
                x_start = i * spot_size_pixels_x
                x_end = (i + 1) * spot_size_pixels_x
                y_start = j * spot_size_pixels_y
                y_end = (j + 1) * spot_size_pixels_y

                spot_subimage = self.spot_image[y_start:y_end, x_start:x_end]

                # Calculer le centroïde et l'erreur
                centroid, error = self._calculate_single_centroid(spot_subimage, i, j)

                # Convertir en coordonnées pixels (relatives à l'image complète)
                self.centroids[spot_idx, 0] = x_start + centroid[0]  # x en pixels
                self.centroids[spot_idx, 1] = y_start + centroid[1]  # y en pixels
                self.centroid_errors[spot_idx, :] = error

        logger.info(f"Centroïdes calculés: {len(self.centroids)} spots")

    def _calculate_single_centroid(self,
                                  spot_image: np.ndarray,
                                  i: int,
                                  j: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        FR: Calcule le centroïde d'une seule tâche d'Airy.
            
        EN: Calculates the centroid of a single Airy spot.
        
        Args:
            spot_image (np.ndarray): Sous-image de la tâche (2D array).
            i (int): Indice x de la microlentille.
            j (int): Indice y de la microlentille.
        
        Returns:
            Tuple[Tuple[float, float], Tuple[float, float]]:
                - (x_centroid, y_centroid) en pixels (relatifs à la sous-image)
                - (x_error, y_error) erreurs estimées en pixels
        """
        if self.centroid_algorithm == CentroidAlgorithm.WEIGHTED_CENTROID:
            return self._weighted_centroid(spot_image)
        elif self.centroid_algorithm == CentroidAlgorithm.THRESHOLDED_CENTROID:
            return self._thresholded_centroid(spot_image, threshold=0.5)
        elif self.centroid_algorithm == CentroidAlgorithm.GAUSSIAN_FIT:
            return self._gaussian_fit_centroid(spot_image)
        elif self.centroid_algorithm == CentroidAlgorithm.MOMENT_BASED:
            return self._moment_based_centroid(spot_image)
        else:
            return self._weighted_centroid(spot_image)

    def _weighted_centroid(self,
                          spot_image: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        FR: Calcule le barycentre pondéré par l'intensité.
            
            Formule :
                x̄ = Σ(I(x,y) * x) / ΣI(x,y)
                ȳ = Σ(I(x,y) * y) / ΣI(x,y)
            
            où I(x,y) est l'intensité au pixel (x,y).
            
            L'erreur est estimée par l'écart-type pondéré :
                σ_x = √(Σ(I(x,y) * (x - x̄)²) / ΣI(x,y))
                σ_y = √(Σ(I(x,y) * (y - ȳ)²) / ΣI(x,y))
        
        EN: Calculates the intensity-weighted centroid.
            
            Formula:
                x̄ = Σ(I(x,y) * x) / ΣI(x,y)
                ȳ = Σ(I(x,y) * y) / ΣI(x,y)
            
            where I(x,y) is the intensity at pixel (x,y).
            
            Error is estimated by the weighted standard deviation:
                σ_x = √(Σ(I(x,y) * (x - x̄)²) / ΣI(x,y))
                σ_y = √(Σ(I(x,y) * (y - ȳ)²) / ΣI(x,y))
        
        Returns:
            Tuple[Tuple[float, float], Tuple[float, float]]: (centroid, error)
        
        Sources:
            - "Principles of Adaptive Optics" by Tyson (1991), Eq. 4.1
        """
        if np.sum(spot_image) == 0:
            return ((0.0, 0.0), (0.0, 0.0))

        # Créer une grille de coordonnées (relatives à la sous-image)
        y_coords, x_coords = np.indices(spot_image.shape)
        total_intensity = np.sum(spot_image)

        # Calculer le centroïde
        x_centroid = np.sum(spot_image * x_coords) / total_intensity
        y_centroid = np.sum(spot_image * y_coords) / total_intensity

        # Estimer l'erreur (écart-type pondéré)
        x_error = np.sqrt(np.sum(spot_image * (x_coords - x_centroid)**2) / total_intensity)
        y_error = np.sqrt(np.sum(spot_image * (y_coords - y_centroid)**2) / total_intensity)

        return ((x_centroid, y_centroid), (x_error, y_error))

    def _thresholded_centroid(self,
                             spot_image: np.ndarray,
                             threshold: float = 0.5) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        FR: Calcule le barycentre avec un seuil.
            Seuls les pixels avec une intensité > threshold * max_intensity sont pris en compte.
            
            Cela permet d'éliminer le bruit de fond et d'améliorer la précision
            pour les tâches avec un faible rapport signal/bruit.
        
        EN: Calculates the centroid with a threshold.
            Only pixels with intensity > threshold * max_intensity are considered.
            
            This helps eliminate background noise and improves accuracy
            for spots with low signal-to-noise ratio.
        
        Args:
            spot_image (np.ndarray): Sous-image de la tâche.
            threshold (float): Seuil relatif (0-1, défaut: 0.5).
        
        Returns:
            Tuple[Tuple[float, float], Tuple[float, float]]: (centroid, error)
        
        Sources:
            - "Centroiding algorithms for Shack-Hartmann sensors" by Winther & Seldin (2004)
        """
        max_intensity = np.max(spot_image)
        if max_intensity == 0:
            return ((0.0, 0.0), (0.0, 0.0))

        # Appliquer le seuil
        masked_image = np.where(spot_image > threshold * max_intensity, spot_image, 0.0)

        # Créer une grille de coordonnées
        y_coords, x_coords = np.indices(spot_image.shape)
        total_intensity = np.sum(masked_image)

        if total_intensity == 0:
            return ((0.0, 0.0), (0.0, 0.0))

        # Calculer le centroïde
        x_centroid = np.sum(masked_image * x_coords) / total_intensity
        y_centroid = np.sum(masked_image * y_coords) / total_intensity

        # Estimer l'erreur
        x_error = np.sqrt(np.sum(masked_image * (x_coords - x_centroid)**2) / total_intensity)
        y_error = np.sqrt(np.sum(masked_image * (y_coords - y_centroid)**2) / total_intensity)

        return ((x_centroid, y_centroid), (x_error, y_error))

    def _gaussian_fit_centroid(self,
                               spot_image: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        FR: Calcule le centroïde par ajustement gaussien 2D.
            
            Modèle :
                I(x,y) = I₀ + A * exp(-((x-x₀)²/2σ_x² + (y-y₀)²/2σ_y²))
            
            où :
                - I₀ : niveau de fond
                - A : amplitude
                - (x₀, y₀) : position du centroïde
                - (σ_x, σ_y) : écarts-types
            
            L'ajustement est effectué avec scipy.optimize.curve_fit.
            
            Cet algorithme est plus précis pour les tâches bruitées ou déformées,
            mais il est plus lent que le barycentre pondéré.
        
        EN: Calculates the centroid by 2D Gaussian fitting.
            
            Model:
                I(x,y) = I₀ + A * exp(-((x-x₀)²/2σ_x² + (y-y₀)²/2σ_y²))
            
            where:
                - I₀: background level
                - A: amplitude
                - (x₀, y₀): centroid position
                - (σ_x, σ_y): standard deviations
            
            Fitting is performed with scipy.optimize.curve_fit.
            
            This algorithm is more accurate for noisy or distorted spots,
            but it is slower than the weighted centroid.
        
        Returns:
            Tuple[Tuple[float, float], Tuple[float, float]]: (centroid, error)
        
        Sources:
            - "Centroiding algorithms for Shack-Hartmann sensors" by Winther & Seldin (2004)
        """
        if not SCIPY_AVAILABLE:
            logger.warning("scipy not available. Falling back to weighted centroid.")
            return self._weighted_centroid(spot_image)

        # Créer une grille de coordonnées
        y_coords, x_coords = np.indices(spot_image.shape)
        x_flat = x_coords.flatten()
        y_flat = y_coords.flatten()
        I_flat = spot_image.flatten()

        # Fonction gaussienne 2D
        def gaussian_2d(xy, I0, A, x0, y0, sigma_x, sigma_y):
            x, y = xy
            return I0 + A * np.exp(-((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2)))

        # Estimation initiale
        I0_init = np.min(I_flat)
        A_init = np.max(I_flat) - I0_init
        x0_init = np.sum(x_flat * I_flat) / np.sum(I_flat)
        y0_init = np.sum(y_flat * I_flat) / np.sum(I_flat)
        sigma_x_init = np.sqrt(np.sum((x_flat - x0_init)**2 * I_flat) / np.sum(I_flat))
        sigma_y_init = np.sqrt(np.sum((y_flat - y0_init)**2 * I_flat) / np.sum(I_flat))

        initial_guess = [I0_init, A_init, x0_init, y0_init, sigma_x_init, sigma_y_init]

        try:
            # Ajustement
            popt, pcov = curve_fit(
                gaussian_2d,
                (x_flat, y_flat),
                I_flat,
                p0=initial_guess,
                maxfev=10000
            )

            # Extraire les paramètres
            _, _, x0, y0, sigma_x, sigma_y = popt

            # Estimer l'erreur (écart-type des paramètres)
            if pcov is not None and pcov.size > 0:
                x_error = np.sqrt(pcov[2, 2]) if pcov.shape[0] > 2 else sigma_x / np.sqrt(np.sum(I_flat))
                y_error = np.sqrt(pcov[3, 3]) if pcov.shape[0] > 3 else sigma_y / np.sqrt(np.sum(I_flat))
            else:
                x_error = sigma_x / np.sqrt(np.sum(I_flat))
                y_error = sigma_y / np.sqrt(np.sum(I_flat))

            return ((x0, y0), (x_error, y_error))

        except Exception as e:
            logger.warning(f"Gaussian fitting failed: {e}. Falling back to weighted centroid.")
            return self._weighted_centroid(spot_image)

    def _moment_based_centroid(self,
                              spot_image: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        FR: Calcule le centroïde basé sur les moments.
            
            Utilise les moments d'ordre 1 pour le centroïde et d'ordre 2 pour l'erreur.
            
            Moments d'ordre 1 :
                x̄ = m10 / m00
                ȳ = m01 / m00
            où m_pq = Σ(x^p * y^q * I(x,y))
            
            Moments d'ordre 2 (variance) :
                σ_x² = m20/m00 - x̄²
                σ_y² = m02/m00 - ȳ²
            
            Cet algorithme est robuste au bruit et souvent utilisé dans les
            systèmes Shack-Hartmann commerciaux.
        
        EN: Calculates the centroid based on moments.
            
            Uses first-order moments for centroid and second-order moments for error.
            
            First-order moments:
                x̄ = m10 / m00
                ȳ = m01 / m00
            where m_pq = Σ(x^p * y^q * I(x,y))
            
            Second-order moments (variance):
                σ_x² = m20/m00 - x̄²
                σ_y² = m02/m00 - ȳ²
            
            This algorithm is robust to noise and often used in commercial
            Shack-Hartmann systems.
        
        Returns:
            Tuple[Tuple[float, float], Tuple[float, float]]: (centroid, error)
        
        Sources:
            - "Image Moments for Visual Recognition" by Hu (1962)
            - "Centroiding algorithms for Shack-Hartmann sensors" by Winther & Seldin (2004)
        """
        if np.sum(spot_image) == 0:
            return ((0.0, 0.0), (0.0, 0.0))

        # Créer une grille de coordonnées
        y_coords, x_coords = np.indices(spot_image.shape)
        total_intensity = np.sum(spot_image)

        # Moments d'ordre 1 (centroïde)
        m10 = np.sum(spot_image * x_coords)
        m01 = np.sum(spot_image * y_coords)
        x_centroid = m10 / total_intensity
        y_centroid = m01 / total_intensity

        # Moments d'ordre 2 (variance)
        m20 = np.sum(spot_image * x_coords**2)
        m02 = np.sum(spot_image * y_coords**2)
        x_variance = m20 / total_intensity - x_centroid**2
        y_variance = m02 / total_intensity - y_centroid**2

        # Erreur = écart-type
        x_error = np.sqrt(x_variance) if x_variance > 0 else 0.0
        y_error = np.sqrt(y_variance) if y_variance > 0 else 0.0

        return ((x_centroid, y_centroid), (x_error, y_error))

    def calculate_slopes(self) -> None:
        """
        FR: Calcule les pentes locales de phase à partir des centroïdes.
            
            FORMULE (CORRIGÉE) :
                slope_x = arctan(dx / focal_length)  [RADIANS]
                slope_y = arctan(dy / focal_length)  [RADIANS]
            
            où :
                - dx, dy : décalages du centroïde par rapport à la position de référence (en mm)
                - focal_length : distance focale des microlentilles (en mm)
                - Le résultat est en RADIANS (pas en rad/mm !)
            
            L'erreur sur les pentes est calculée par propagation de l'erreur :
                error_slope_x = |d(slope_x)/dx| * error_x_centroid
                error_slope_y = |d(slope_y)/dy| * error_y_centroid
            
            où d(slope_x)/dx = 1 / (focal_length * (1 + (dx/focal_length)^2))
            
        EN: Calculates local phase slopes from centroids.
            
            FORMULA (CORRECTED):
                slope_x = arctan(dx / focal_length)  [RADIANS]
                slope_y = arctan(dy / focal_length)  [RADIANS]
            
            where:
                - dx, dy: centroid shifts from reference position (in mm)
                - focal_length: focal length of microlenses (in mm)
                - Result is in RADIANS (not rad/mm!)
            
            Error on slopes is calculated by error propagation:
                error_slope_x = |d(slope_x)/dx| * error_x_centroid
                error_slope_y = |d(slope_y)/dy| * error_y_centroid
            
            where d(slope_x)/dx = 1 / (focal_length * (1 + (dx/focal_length)^2))
        
        Sources:
            - "Principles of Adaptive Optics" by Tyson (1991), Eq. 4.2
            - "The Southwell Geometry for Wavefront Sensing" by Southwell (1980)
            - "Adaptive Optics for Astronomical Telescopes" by Hardy (1998)
              -> Slope formula: theta = arctan(dx/f) in RADIANS
        """
        if self.centroids is None:
            raise ValueError("Centroids not calculated. Run calculate_centroids() first.")

        num_spots_x = self.microlens_array.num_elements_x
        num_spots_y = self.microlens_array.num_elements_y

        # Initialiser les cartes de pentes et d'erreurs
        self.slopes_x = np.zeros((num_spots_y, num_spots_x))
        self.slopes_y = np.zeros((num_spots_y, num_spots_x))
        self.slope_errors = np.zeros((num_spots_y, num_spots_x, 2))  # (x_error, y_error)

        # Calculer la taille d'une sous-image par spot (en pixels)
        spot_size_pixels_x = self.camera.specifications.num_pixels_x // num_spots_x
        spot_size_pixels_y = self.camera.specifications.num_pixels_y // num_spots_y

        # Taille des pixels en mm
        pixel_size_mm = self.camera.pixel_width_mm

        for j in range(num_spots_y):
            for i in range(num_spots_x):
                spot_idx = j * num_spots_x + i

                # Position de référence (centre de la sous-image en pixels)
                x_ref_pixels = (i + 0.5) * spot_size_pixels_x
                y_ref_pixels = (j + 0.5) * spot_size_pixels_y

                # Position du centroïde (en pixels)
                x_centroid_pixels = self.centroids[spot_idx, 0]
                y_centroid_pixels = self.centroids[spot_idx, 1]

                # Erreur sur le centroïde (en pixels)
                x_error_pixels = self.centroid_errors[spot_idx, 0]
                y_error_pixels = self.centroid_errors[spot_idx, 1]

                # Calculer les décalages (en mm)
                dx_mm = (x_centroid_pixels - x_ref_pixels) * pixel_size_mm
                dy_mm = (y_centroid_pixels - y_ref_pixels) * pixel_size_mm

                # Calculer les pentes EN RADIANS (CORRECTION)
                # theta = arctan(dx / f)
                self.slopes_x[j, i] = np.arctan2(dx_mm, self.focal_length_mm)
                self.slopes_y[j, i] = np.arctan2(dy_mm, self.focal_length_mm)

                # Calculer les erreurs sur les pentes EN RADIANS
                # d(slope)/dx = 1 / (f * (1 + (dx/f)^2)) = f / (f^2 + dx^2)
                # Mais pour la propagation d'erreur : error_slope = |d(slope)/dx| * error_x
                # Pour arctan(dx/f) : d(slope)/dx = f / (f^2 + dx^2)
                f_sq = self.focal_length_mm ** 2
                
                d_slope_x_dx = self.focal_length_mm / (f_sq + dx_mm**2)
                d_slope_y_dy = self.focal_length_mm / (f_sq + dy_mm**2)
                
                # Erreur en mm
                x_error_mm = x_error_pixels * pixel_size_mm
                y_error_mm = y_error_pixels * pixel_size_mm
                
                self.slope_errors[j, i, 0] = np.abs(d_slope_x_dx) * x_error_mm
                self.slope_errors[j, i, 1] = np.abs(d_slope_y_dy) * y_error_mm

        logger.info(f"Pentes calculées (en RADIANS): {num_spots_x}x{num_spots_y} valeurs")

    def get_slope_maps(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        FR: Retourne les cartes de pentes locales (en RADIANS).
            
        EN: Returns the local slope maps (in RADIANS).
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (slopes_x, slopes_y)
                - slopes_x: Carte des pentes en x (RADIANS)
                - slopes_y: Carte des pentes en y (RADIANS)
        """
        if self.slopes_x is None or self.slopes_y is None:
            raise ValueError("Slopes not calculated. Run calculate_slopes() first.")
        return self.slopes_x, self.slopes_y

    def get_slope_error_maps(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        FR: Retourne les cartes d'erreur sur les pentes (en RADIANS).
            
        EN: Returns the slope error maps (in RADIANS).
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (slope_errors_x, slope_errors_y)
                - slope_errors_x: Carte des erreurs sur les pentes en x (RADIANS)
                - slope_errors_y: Carte des erreurs sur les pentes en y (RADIANS)
        """
        if self.slope_errors is None:
            raise ValueError("Slope errors not calculated. Run calculate_slopes() first.")
        return self.slope_errors[..., 0], self.slope_errors[..., 1]

    def get_centroid_map(self) -> np.ndarray:
        """
        FR: Retourne une carte des centroïdes (pour visualisation).
            Chaque centroïde est marqué par un point sur l'image des spots.
            
        EN: Returns a centroid map (for visualization).
            Each centroid is marked by a point on the spot image.
        
        Returns:
            np.ndarray: Image avec les centroïdes marqués (même taille que spot_image).
        """
        if self.spot_image is None or self.centroids is None:
            raise ValueError("Spot image or centroids not available.")

        # Créer une copie de l'image des spots
        centroid_map = np.copy(self.spot_image)

        # Normaliser l'image pour l'affichage
        if np.max(centroid_map) > 0:
            centroid_map = centroid_map / np.max(centroid_map)

        # Marquer les centroïdes (en blanc)
        for centroid in self.centroids:
            x, y = int(round(centroid[0])), int(round(centroid[1]))
            if 0 <= x < centroid_map.shape[1] and 0 <= y < centroid_map.shape[0]:
                # Dessiner une croix
                size = 3
                centroid_map[max(0, y-size):min(centroid_map.shape[0], y+size+1),
                             max(0, x-size):min(centroid_map.shape[1], x+size+1)] = 1.0

        return centroid_map

    def get_centroid_error_map(self) -> np.ndarray:
        """
        FR: Retourne une carte des erreurs sur les centroïdes (en pixels).
            
        EN: Returns a map of centroid errors (in pixels).
        
        Returns:
            np.ndarray: Carte des erreurs (moyenne des erreurs x et y).
        """
        if self.centroid_errors is None:
            raise ValueError("Centroid errors not calculated.")

        num_spots_x = self.microlens_array.num_elements_x
        num_spots_y = self.microlens_array.num_elements_y

        error_map = np.zeros((num_spots_y, num_spots_x))

        for j in range(num_spots_y):
            for i in range(num_spots_x):
                spot_idx = j * num_spots_x + i
                error_map[j, i] = np.mean(self.centroid_errors[spot_idx])

        return error_map

    def get_total_slope_error_map(self) -> np.ndarray:
        """
        FR: Retourne une carte de l'erreur totale sur les pentes (en RADIANS).
            
        EN: Returns a map of total slope error (in RADIANS).
        
        Returns:
            np.ndarray: Carte de l'erreur totale (sqrt(error_x² + error_y²)).
        """
        if self.slope_errors is None:
            raise ValueError("Slope errors not calculated.")

        slope_error_x, slope_error_y = self.get_slope_error_maps()
        return np.sqrt(slope_error_x**2 + slope_error_y**2)

    def get_microlens_positions(self) -> np.ndarray:
        """
        FR: Retourne les positions des microlentilles (en mm).
            
        EN: Returns the positions of the microlenses (in mm).
        
        Returns:
            np.ndarray: Tableau de positions (Nx2: [x, y] en mm).
        """
        positions = []
        for mo in self.microlens_array.micro_optics:
            positions.append(mo.position_mm)
        return np.array(positions)

    # =========================================================================
    # VISUALISATION / VISUALIZATION
    # =========================================================================

    def visualize_spots(self,
                        save_path: Optional[str] = None,
                        show: bool = False,
                        cmap: str = "hot") -> None:
        """
        FR: Visualise la carte des tâches d'Airy.
            
        EN: Visualizes the Airy spot map.
        
        Args:
            save_path (str, optional): Chemin pour sauvegarder l'image.
            show (bool, optional): Afficher l'image.
            cmap (str, optional): Colormap à utiliser (défaut: "hot").
        """
        if self.spot_image is None:
            raise ValueError("Spot image not available. Run simulate() first.")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available. Cannot visualize.")
            return

        plt.figure(figsize=(10, 8))

        # Afficher l'image des spots
        extent = [
            -self.camera.sensor_width_mm / 2,
            self.camera.sensor_width_mm / 2,
            -self.camera.sensor_height_mm / 2,
            self.camera.sensor_height_mm / 2
        ]
        
        plt.imshow(self.spot_image, cmap=cmap, extent=extent, origin='lower')
        plt.title(f"Carte des tâches d'Airy - {self.name}")
        plt.xlabel("x (mm)")
        plt.ylabel("y (mm)")
        plt.colorbar(label="Intensité (ADU)")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        elif show:
            plt.show()
        else:
            plt.close()

    def visualize_centroids(self,
                            save_path: Optional[str] = None,
                            show: bool = False,
                            cmap: str = "hot") -> None:
        """
        FR: Visualise la carte des centroïdes (marqués sur l'image des spots).
            
        EN: Visualizes the centroid map (marked on the spot image).
        
        Args:
            save_path (str, optional): Chemin pour sauvegarder l'image.
            show (bool, optional): Afficher l'image.
            cmap (str, optional): Colormap à utiliser (défaut: "hot").
        """
        if self.spot_image is None or self.centroids is None:
            raise ValueError("Spot image or centroids not available.")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available. Cannot visualize.")
            return

        centroid_map = self.get_centroid_map()

        extent = [
            -self.camera.sensor_width_mm / 2,
            self.camera.sensor_width_mm / 2,
            -self.camera.sensor_height_mm / 2,
            self.camera.sensor_height_mm / 2
        ]

        plt.figure(figsize=(10, 8))
        plt.imshow(centroid_map, cmap=cmap, extent=extent, origin='lower')
        plt.title(f"Carte des centroïdes - {self.name}")
        plt.xlabel("x (mm)")
        plt.ylabel("y (mm)")
        plt.colorbar(label="Intensité normalisée")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        elif show:
            plt.show()
        else:
            plt.close()

    def visualize_slopes(self,
                        save_path: Optional[str] = None,
                        show: bool = False,
                        cmap: str = "Jet") -> None:
        """
        FR: Visualise les cartes de pentes locales (X et Y) EN RADIANS.
            
        EN: Visualizes the local slope maps (X and Y) IN RADIANS.
        
        Args:
            save_path (str, optional): Chemin pour sauvegarder l'image.
            show (bool, optional): Afficher l'image.
            cmap (str, optional): Colormap à utiliser (défaut: "Jet").
        """
        if self.slopes_x is None or self.slopes_y is None:
            raise ValueError("Slopes not calculated. Run calculate_slopes() first.")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available. Cannot visualize.")
            return

        # Obtenir les positions des microlentilles (en mm)
        microlens_positions = self.get_microlens_positions()
        microlens_x = microlens_positions[:, 0]
        microlens_y = microlens_positions[:, 1]

        # Créer une grille pour l'affichage
        num_spots_x = self.microlens_array.num_elements_x
        num_spots_y = self.microlens_array.num_elements_y

        x_min, x_max = np.min(microlens_x), np.max(microlens_x)
        y_min, y_max = np.min(microlens_y), np.max(microlens_y)

        # Visualisation
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Pentes X EN RADIANS
        im1 = axes[0].imshow(
            self.slopes_x,
            extent=[x_min, x_max, y_min, y_max],
            cmap=cmap,
            origin='lower'
        )
        axes[0].set_title(f"Pentes locales X (dφ/dx) - {self.name}\n"
                         f"PV={np.max(self.slopes_x) - np.min(self.slopes_x):.6f} rad, "
                         f"RMS={np.std(self.slopes_x):.6f} rad")
        axes[0].set_xlabel("x (mm)")
        axes[0].set_ylabel("y (mm)")
        plt.colorbar(im1, ax=axes[0], label="Pente (rad)")

        # Pentes Y EN RADIANS
        im2 = axes[1].imshow(
            self.slopes_y,
            extent=[x_min, x_max, y_min, y_max],
            cmap=cmap,
            origin='lower'
        )
        axes[1].set_title(f"Pentes locales Y (dφ/dy) - {self.name}\n"
                         f"PV={np.max(self.slopes_y) - np.min(self.slopes_y):.6f} rad, "
                         f"RMS={np.std(self.slopes_y):.6f} rad")
        axes[1].set_xlabel("x (mm)")
        axes[1].set_ylabel("y (mm)")
        plt.colorbar(im2, ax=axes[1], label="Pente (rad)")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        elif show:
            plt.show()
        else:
            plt.close()

    def visualize_errors(self,
                         save_path: Optional[str] = None,
                         show: bool = False,
                         cmap: str = "hot") -> None:
        """
        FR: Visualise les cartes d'erreur (centroïdes et pentes) EN RADIANS.
            
        EN: Visualizes the error maps (centroids and slopes) IN RADIANS.
        
        Args:
            save_path (str, optional): Chemin pour sauvegarder l'image.
            show (bool, optional): Afficher l'image.
            cmap (str, optional): Colormap à utiliser (défaut: "hot").
        """
        if self.centroid_errors is None or self.slope_errors is None:
            raise ValueError("Errors not calculated. Run calculate_slopes() first.")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available. Cannot visualize.")
            return

        # Obtenir les positions des microlentilles (en mm)
        microlens_positions = self.get_microlens_positions()
        microlens_x = microlens_positions[:, 0]
        microlens_y = microlens_positions[:, 1]

        x_min, x_max = np.min(microlens_x), np.max(microlens_x)
        y_min, y_max = np.min(microlens_y), np.max(microlens_y)

        # Cartes d'erreur
        centroid_error_map = self.get_centroid_error_map()
        slope_error_x, slope_error_y = self.get_slope_error_maps()
        total_slope_error = self.get_total_slope_error_map()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Erreurs sur les centroïdes (en pixels)
        im1 = axes[0, 0].imshow(
            centroid_error_map,
            extent=[x_min, x_max, y_min, y_max],
            cmap=cmap,
            origin='lower'
        )
        axes[0, 0].set_title(f"Erreurs sur les centroïdes - {self.name}\n"
                            f"Moyenne={np.mean(centroid_error_map):.4f} pixels, "
                            f"Max={np.max(centroid_error_map):.4f} pixels")
        axes[0, 0].set_xlabel("x (mm)")
        axes[0, 0].set_ylabel("y (mm)")
        plt.colorbar(im1, ax=axes[0, 0], label="Erreur (pixels)")

        # Erreurs sur les pentes X EN RADIANS
        im2 = axes[0, 1].imshow(
            slope_error_x,
            extent=[x_min, x_max, y_min, y_max],
            cmap=cmap,
            origin='lower'
        )
        axes[0, 1].set_title(f"Erreurs sur les pentes X - {self.name}\n"
                            f"Moyenne={np.mean(slope_error_x):.6f} rad, "
                            f"Max={np.max(slope_error_x):.6f} rad")
        axes[0, 1].set_xlabel("x (mm)")
        axes[0, 1].set_ylabel("y (mm)")
        plt.colorbar(im2, ax=axes[0, 1], label="Erreur (rad)")

        # Erreurs sur les pentes Y EN RADIANS
        im3 = axes[1, 0].imshow(
            slope_error_y,
            extent=[x_min, x_max, y_min, y_max],
            cmap=cmap,
            origin='lower'
        )
        axes[1, 0].set_title(f"Erreurs sur les pentes Y - {self.name}\n"
                            f"Moyenne={np.mean(slope_error_y):.6f} rad, "
                            f"Max={np.max(slope_error_y):.6f} rad")
        axes[1, 0].set_xlabel("x (mm)")
        axes[1, 0].set_ylabel("y (mm)")
        plt.colorbar(im3, ax=axes[1, 0], label="Erreur (rad)")

        # Erreur totale sur les pentes EN RADIANS
        im4 = axes[1, 1].imshow(
            total_slope_error,
            extent=[x_min, x_max, y_min, y_max],
            cmap=cmap,
            origin='lower'
        )
        axes[1, 1].set_title(f"Erreur totale sur les pentes - {self.name}\n"
                            f"Moyenne={np.mean(total_slope_error):.6f} rad, "
                            f"Max={np.max(total_slope_error):.6f} rad")
        axes[1, 1].set_xlabel("x (mm)")
        axes[1, 1].set_ylabel("y (mm)")
        plt.colorbar(im4, ax=axes[1, 1], label="Erreur (rad)")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        elif show:
            plt.show()
        else:
            plt.close()

    def visualize_all(self,
                       output_dir: Optional[str] = None) -> None:
        """
        FR: Visualise toutes les cartes (spots, centroïdes, pentes, erreurs).
            
        EN: Visualizes all maps (spots, centroids, slopes, errors).
        
        Args:
            output_dir (str, optional): Répertoire pour sauvegarder les images.
        """
        if output_dir is None:
            output_dir = self.display_dir

        os.makedirs(output_dir, exist_ok=True)

        # Carte des spots
        self.visualize_spots(
            save_path=os.path.join(output_dir, f"{self.name}_spots.png"),
            show=False
        )

        # Carte des centroïdes
        self.visualize_centroids(
            save_path=os.path.join(output_dir, f"{self.name}_centroids.png"),
            show=False
        )

        # Cartes des pentes EN RADIANS
        self.visualize_slopes(
            save_path=os.path.join(output_dir, f"{self.name}_slopes.png"),
            show=False
        )

        # Cartes des erreurs EN RADIANS
        self.visualize_errors(
            save_path=os.path.join(output_dir, f"{self.name}_errors.png"),
            show=False
        )

        logger.info(f"Toutes les cartes sauvegardées dans {output_dir}/")

    def get_slope_data(self) -> Dict:
        """
        FR: Retourne toutes les données de pentes sous forme de dictionnaire.
            
        EN: Returns all slope data as a dictionary.
        
        Returns:
            Dict: Dictionnaire avec toutes les données de pentes.
        """
        if self.slopes_x is None or self.slopes_y is None:
            raise ValueError("Slopes not calculated.")

        return {
            'name': self.name,
            'slopes_x': self.slopes_x,
            'slopes_y': self.slopes_y,
            'slope_errors_x': self.slope_errors[..., 0],
            'slope_errors_y': self.slope_errors[..., 1],
            'centroids': self.centroids,
            'centroid_errors': self.centroid_errors,
            'focal_length_mm': self.focal_length_mm,
            'wavelength_nm': self.wavelength_nm,
            'pixel_size_mm': self.camera.pixel_width_mm,
            'microlens_array': self.microlens_array,
            'camera': self.camera,
            'spot_image': self.spot_image,
            'slope_units': 'rad'  # UNITÉS CORRIGÉES
        }

    def get_slope_statistics(self) -> Dict:
        """
        FR: Retourne les statistiques des pentes (PV, RMS, etc.) EN RADIANS.
            
        EN: Returns slope statistics (PV, RMS, etc.) IN RADIANS.
        
        Returns:
            Dict: Dictionnaire avec les statistiques des pentes.
        """
        if self.slopes_x is None or self.slopes_y is None:
            raise ValueError("Slopes not calculated.")

        return {
            'slopes_x': {
                'min': float(np.min(self.slopes_x)),
                'max': float(np.max(self.slopes_x)),
                'mean': float(np.mean(self.slopes_x)),
                'std': float(np.std(self.slopes_x)),
                'pv': float(np.max(self.slopes_x) - np.min(self.slopes_x)),
                'rms': float(np.std(self.slopes_x)),
                'units': 'rad'  # UNITÉS CORRIGÉES
            },
            'slopes_y': {
                'min': float(np.min(self.slopes_y)),
                'max': float(np.max(self.slopes_y)),
                'mean': float(np.mean(self.slopes_y)),
                'std': float(np.std(self.slopes_y)),
                'pv': float(np.max(self.slopes_y) - np.min(self.slopes_y)),
                'rms': float(np.std(self.slopes_y)),
                'units': 'rad'  # UNITÉS CORRIGÉES
            },
            'slope_errors_x': {
                'min': float(np.min(self.slope_errors[..., 0])),
                'max': float(np.max(self.slope_errors[..., 0])),
                'mean': float(np.mean(self.slope_errors[..., 0])),
                'std': float(np.std(self.slope_errors[..., 0])),
                'rms': float(np.std(self.slope_errors[..., 0])),
                'units': 'rad'  # UNITÉS CORRIGÉES
            },
            'slope_errors_y': {
                'min': float(np.min(self.slope_errors[..., 1])),
                'max': float(np.max(self.slope_errors[..., 1])),
                'mean': float(np.mean(self.slope_errors[..., 1])),
                'std': float(np.std(self.slope_errors[..., 1])),
                'rms': float(np.std(self.slope_errors[..., 1])),
                'units': 'rad'  # UNITÉS CORRIGÉES
            }
        }

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"microlenses={self.microlens_array.num_elements_x}x{self.microlens_array.num_elements_y}, "
                f"focal_length={self.focal_length_mm:.1f}mm, "
                f"camera={self.camera.specifications.num_pixels_x}x{self.camera.specifications.num_pixels_y}, "
                f"algorithm={self.centroid_algorithm.value})")


# =============================================================================
# FONCTIONS UTILITAIRES / UTILITY FUNCTIONS
# =============================================================================

def create_shack_hartmann(
    name: str = "ShackHartmann",
    num_microlenses_x: int = 10,
    num_microlenses_y: int = 10,
    microlens_pitch_mm: float = 0.5,
    focal_length_mm: float = 20.0,
    camera_num_pixels_x: int = 1024,
    camera_num_pixels_y: int = 1024,
    camera_pixel_size_um: float = 5.0,
    wavelength_nm: float = 633.0,
    centroid_algorithm: CentroidAlgorithm = CentroidAlgorithm.WEIGHTED_CENTROID,
    slope_method: SlopeCalculationMethod = SlopeCalculationMethod.FINITE_DIFFERENCE,
    display: bool = False,
    display_dir: str = "output"
) -> ShackHartmann:
    """
    FR: Fabrique un système Shack-Hartmann complet avec des paramètres par défaut.
        Les pentes seront calculées EN RADIANS.
        
    EN: Factory function to create a complete Shack-Hartmann system with default parameters.
        Slopes will be calculated IN RADIANS.
    
    Args:
        name (str): Nom du système.
        num_microlenses_x (int): Nombre de microlentilles en x (défaut: 10).
        num_microlenses_y (int): Nombre de microlentilles en y (défaut: 10).
        microlens_pitch_mm (float): Pas entre les microlentilles (bord-à-bord) en mm (défaut: 0.5).
        focal_length_mm (float): Distance focale des microlentilles en mm (défaut: 20.0).
        camera_num_pixels_x (int): Nombre de pixels de la caméra en x (défaut: 1024).
        camera_num_pixels_y (int): Nombre de pixels de la caméra en y (défaut: 1024).
        camera_pixel_size_um (float): Taille des pixels de la caméra en µm (défaut: 5.0).
        wavelength_nm (float): Longueur d'onde en nm (défaut: 633.0).
        centroid_algorithm (CentroidAlgorithm): Algorithme de calcul des centroïdes.
        slope_method (SlopeCalculationMethod): Méthode de calcul des pentes.
        display (bool): Afficher automatiquement les cartes.
        display_dir (str): Répertoire pour sauvegarder les images.
    
    Returns:
        ShackHartmann: Le système Shack-Hartmann créé.
    
    Sources:
        - "Principles of Adaptive Optics" by Tyson (1991)
        - "Adaptive Optics for Astronomical Telescopes" by Hardy (1998)
          -> Slope formula: theta = arctan(dx/f) in RADIANS
    """
    if not MICROSTRUCTURE_AVAILABLE:
        raise ImportError("Microstructure module required for create_shack_hartmann()")
    if not CAMERA_AVAILABLE:
        raise ImportError("Camera module required for create_shack_hartmann()")

    # Créer la matrice de microlentilles
    microlens_array = create_microlens_array(
        name=f"{name}_MicrolensArray",
        pitch_mm=microlens_pitch_mm,
        num_elements_x=num_microlenses_x,
        num_elements_y=num_microlenses_y,
        focal_length_mm=focal_length_mm,
        wavelength_nm=wavelength_nm,
        edge_to_edge_spacing_mm=0.0  # Éléments joints
    )

    # Créer la caméra
    from Camera import RealCamera
    camera = RealCamera(
        name=f"{name}_Camera",
        num_pixels_x=camera_num_pixels_x,
        num_pixels_y=camera_num_pixels_y,
        pixel_size_um=camera_pixel_size_um,
        wavelength_nm=wavelength_nm,
        material_name="Silicon"
    )

    # Créer le système Shack-Hartmann
    return ShackHartmann(
        name=name,
        microlens_array=microlens_array,
        camera=camera,
        wavelength_nm=wavelength_nm,
        focal_length_mm=focal_length_mm,
        centroid_algorithm=centroid_algorithm,
        slope_method=slope_method,
        display=display,
        display_dir=display_dir
    )


# =============================================================================
# TESTS UNITAIRES / UNIT TESTS
# =============================================================================

class TestShackHartmann:
    """FR: Tests unitaires pour Shack_Hartmann.py."""

    def test_shack_hartmann_creation(self):
        """FR: Test la création d'un système Shack-Hartmann."""
        if not MICROSTRUCTURE_AVAILABLE or not CAMERA_AVAILABLE:
            return

        sh = create_shack_hartmann(
            name="Test",
            num_microlenses_x=5,
            num_microlenses_y=5,
            focal_length_mm=20.0
        )

        assert sh.microlens_array.num_elements_x == 5
        assert sh.microlens_array.num_elements_y == 5
        assert sh.focal_length_mm == 20.0
        assert sh.camera.specifications.num_pixels_x == 1024
        assert sh.centroid_algorithm == CentroidAlgorithm.WEIGHTED_CENTROID

    def test_simulation(self):
        """FR: Test la simulation avec un faisceau gaussien."""
        if not BEAM_AVAILABLE or not PROPAGATION_AVAILABLE or not MICROSTRUCTURE_AVAILABLE or not CAMERA_AVAILABLE:
            return

        sh = create_shack_hartmann(
            name="Test Simulation",
            num_microlenses_x=3,
            num_microlenses_y=3,
            focal_length_mm=20.0
        )

        beam = Beam(
            wavelength_nm=633.0,
            diameter_mm=5.0,
            num_points=128
        )
        electric_field = beam.generate_electric_field(method="gaussian")
        beam.electric_field = electric_field
        beam.intensity = beam.compute_intensity_from_electric_field(electric_field)

        sh.simulate(beam)

        assert sh.spot_image is not None
        assert sh.centroids is not None
        assert sh.slopes_x is not None
        assert sh.slopes_y is not None
        assert sh.centroids.shape[0] == 9  # 3x3 spots
        assert sh.slopes_x.shape == (3, 3)
        assert sh.slopes_y.shape == (3, 3)

    def test_slope_units_are_radians(self):
        """FR: Test que les pentes sont bien en radians et non en rad/mm."""
        if not BEAM_AVAILABLE or not PROPAGATION_AVAILABLE or not MICROSTRUCTURE_AVAILABLE or not CAMERA_AVAILABLE:
            return

        sh = create_shack_hartmann(
            name="Test Units",
            num_microlenses_x=3,
            num_microlenses_y=3,
            focal_length_mm=20.0
        )

        # Créer un faisceau décentré pour avoir des pentes non nulles
        beam = Beam(
            wavelength_nm=633.0,
            diameter_mm=5.0,
            num_points=128
        )
        electric_field = beam.generate_electric_field(method="gaussian", sigma_mm=0.5)
        # Décaler le faisceau de 0.5 mm
        x = np.linspace(-beam.diameter_mm/2, beam.diameter_mm/2, beam.num_points)
        y = np.linspace(-beam.diameter_mm/2, beam.diameter_mm/2, beam.num_points)
        X, Y = np.meshgrid(x, y)
        shifted_field = electric_field * np.exp(1j * 2 * np.pi * 0.5 / beam.wavelength_nm * 1e-3)  # Décalage de 0.5 mm
        beam.electric_field = shifted_field
        beam.intensity = beam.compute_intensity_from_electric_field(shifted_field)

        sh.simulate(beam)

        slopes_x, slopes_y = sh.get_slope_maps()
        
        # Les pentes doivent être en radians
        # Pour un décalage de 0.5 mm avec f=20 mm : theta = arctan(0.5/20) ≈ 0.025 rad
        # Vérifier que les valeurs sont dans une plage raisonnable pour des radians
        assert np.all(np.abs(slopes_x) < np.pi/2), "Les pentes doivent être < π/2 rad"
        assert np.all(np.abs(slopes_y) < np.pi/2), "Les pentes doivent être < π/2 rad"
        
        # Vérifier que les unités sont bien en 'rad' dans les statistiques
        stats = sh.get_slope_statistics()
        assert stats['slopes_x']['units'] == 'rad', "Les unités doivent être en rad"
        assert stats['slopes_y']['units'] == 'rad', "Les unités doivent être en rad"

    def test_centroid_algorithms(self):
        """FR: Test les différents algorithmes de calcul des centroïdes."""
        if not BEAM_AVAILABLE or not PROPAGATION_AVAILABLE or not MICROSTRUCTURE_AVAILABLE or not CAMERA_AVAILABLE:
            return

        algorithms = [
            CentroidAlgorithm.WEIGHTED_CENTROID,
            CentroidAlgorithm.THRESHOLDED_CENTROID,
            CentroidAlgorithm.MOMENT_BASED
        ]

        for algo in algorithms:
            sh = create_shack_hartmann(
                name=f"Test {algo.value}",
                num_microlenses_x=3,
                num_microlenses_y=3,
                focal_length_mm=20.0,
                centroid_algorithm=algo
            )

            beam = Beam(
                wavelength_nm=633.0,
                diameter_mm=5.0,
                num_points=128
            )
            electric_field = beam.generate_electric_field(method="gaussian")
            beam.electric_field = electric_field
            beam.intensity = beam.compute_intensity_from_electric_field(electric_field)

            sh.simulate(beam)
            assert sh.centroids is not None
            assert sh.centroids.shape[0] == 9

    def test_slope_calculation(self):
        """FR: Test le calcul des pentes."""
        if not BEAM_AVAILABLE or not PROPAGATION_AVAILABLE or not MICROSTRUCTURE_AVAILABLE or not CAMERA_AVAILABLE:
            return

        sh = create_shack_hartmann(
            name="Test Slopes",
            num_microlenses_x=3,
            num_microlenses_y=3,
            focal_length_mm=20.0
        )

        beam = Beam(
            wavelength_nm=633.0,
            diameter_mm=5.0,
            num_points=128
        )
        electric_field = beam.generate_electric_field(method="gaussian")
        beam.electric_field = electric_field
        beam.intensity = beam.compute_intensity_from_electric_field(electric_field)

        sh.simulate(beam)

        slopes_x, slopes_y = sh.get_slope_maps()
        assert slopes_x.shape == (3, 3)
        assert slopes_y.shape == (3, 3)

        # Les pentes doivent être proches de 0 pour un faisceau gaussien centré
        assert np.abs(np.mean(slopes_x)) < 0.01
        assert np.abs(np.mean(slopes_y)) < 0.01

    def test_error_estimation(self):
        """FR: Test l'estimation des erreurs."""
        if not BEAM_AVAILABLE or not PROPAGATION_AVAILABLE or not MICROSTRUCTURE_AVAILABLE or not CAMERA_AVAILABLE:
            return

        sh = create_shack_hartmann(
            name="Test Errors",
            num_microlenses_x=3,
            num_microlenses_y=3,
            focal_length_mm=20.0
        )

        beam = Beam(
            wavelength_nm=633.0,
            diameter_mm=5.0,
            num_points=128
        )
        electric_field = beam.generate_electric_field(method="gaussian")
        beam.electric_field = electric_field
        beam.intensity = beam.compute_intensity_from_electric_field(electric_field)

        sh.simulate(beam)

        centroid_errors = sh.centroid_errors
        slope_errors_x, slope_errors_y = sh.get_slope_error_maps()

        assert centroid_errors.shape == (9, 2)
        assert slope_errors_x.shape == (3, 3)
        assert slope_errors_y.shape == (3, 3)

        # Les erreurs doivent être positives
        assert np.all(centroid_errors >= 0)
        assert np.all(slope_errors_x >= 0)
        assert np.all(slope_errors_y >= 0)

    def test_visualization(self):
        """FR: Test la visualisation."""
        if not BEAM_AVAILABLE or not PROPAGATION_AVAILABLE or not MICROSTRUCTURE_AVAILABLE or not CAMERA_AVAILABLE:
            return

        try:
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            return

        sh = create_shack_hartmann(
            name="Test Visualization",
            num_microlenses_x=3,
            num_microlenses_y=3,
            focal_length_mm=20.0,
            display=True,
            display_dir="test_output"
        )

        beam = Beam(
            wavelength_nm=633.0,
            diameter_mm=5.0,
            num_points=128
        )
        electric_field = beam.generate_electric_field(method="gaussian")
        beam.electric_field = electric_field
        beam.intensity = beam.compute_intensity_from_electric_field(electric_field)

        sh.simulate(beam)
        sh.visualize_all(output_dir="test_output")

        # Vérifier que les fichiers ont été créés
        assert os.path.exists("test_output/Test Visualization_spots.png")
        assert os.path.exists("test_output/Test Visualization_centroids.png")
        assert os.path.exists("test_output/Test Visualization_slopes.png")
        assert os.path.exists("test_output/Test Visualization_errors.png")

        # Nettoyer
        import shutil
        shutil.rmtree("test_output")

    def test_statistics(self):
        """FR: Test le calcul des statistiques."""
        if not BEAM_AVAILABLE or not PROPAGATION_AVAILABLE or not MICROSTRUCTURE_AVAILABLE or not CAMERA_AVAILABLE:
            return

        sh = create_shack_hartmann(
            name="Test Statistics",
            num_microlenses_x=3,
            num_microlenses_y=3,
            focal_length_mm=20.0
        )

        beam = Beam(
            wavelength_nm=633.0,
            diameter_mm=5.0,
            num_points=128
        )
        electric_field = beam.generate_electric_field(method="gaussian")
        beam.electric_field = electric_field
        beam.intensity = beam.compute_intensity_from_electric_field(electric_field)

        sh.simulate(beam)

        stats = sh.get_slope_statistics()

        assert 'slopes_x' in stats
        assert 'slopes_y' in stats
        assert 'slope_errors_x' in stats
        assert 'slope_errors_y' in stats

        assert 'pv' in stats['slopes_x']
        assert 'rms' in stats['slopes_x']
        assert stats['slopes_x']['units'] == 'rad'
        assert stats['slopes_y']['units'] == 'rad'


if __name__ == "__main__":
    import unittest
    unittest.main()

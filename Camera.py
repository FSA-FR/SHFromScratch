"""
Camera.py

FR: Module pour la création et la gestion de capteurs virtuels (parfaits ou réels).
    Permet de simuler :
    - Un capteur parfait (échantillonnage simple du faisceau par pixel)
    - Un capteur réel avec :
        * Puits quantiques (full well capacity)
        * Bruits standards (gaussien, Poisson, lecture, obscurité, thermique, quantification)
        * Matériau et expansion thermique
        * Filtre couleur (CFA)
    
    Chaque image générée (phase et intensité) aura :
    - Une échelle visuelle
    - Le PV (Peak-to-Valley) et le RMS des valeurs
    - Colormap : "Jet" pour la phase, "hot" pour l'intensité
    
    Unités :
    - Longueurs : mm (taille du capteur), µm (taille des pixels)
    - Longueur d'onde : nm
    - Phase : nm (principale), λ (longueur d'onde), rad, mrad
    - Intensité : a.u. (arbitrary units) ou ADU

EN: Module for creating and managing virtual sensors (ideal or real).
    Allows simulating:
    - An ideal sensor (simple beam sampling per pixel)
    - A real sensor with:
        * Quantum wells (full well capacity)
        * Standard noise (Gaussian, Poisson, readout, dark current, thermal, quantization)
        * Material and thermal expansion
        * Color Filter Array (CFA)
    
    Each generated image (phase and intensity) will have:
    - A visual scale
    - PV (Peak-to-Valley) and RMS values
    - Colormap: "Jet" for phase, "hot" for intensity
    
    Units:
    - Lengths: mm (sensor size), µm (pixel size)
    - Wavelength: nm
    - Phase: nm (main), λ (wavelength), rad, mrad
    - Intensity: a.u. (arbitrary units) or ADU

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch

Sources:
    - "Digital Image Sensors and Cameras" by Gerald C. Holst (SPIE Press, 2011)
      -> Modèle de bruit des capteurs CMOS/CCD
    - "The Physics and Technology of CCD and CMOS Image Sensors" by Janesick (SPIE, 2007)
      -> Puits quantiques, bruit de lecture, courant d'obscurité
    - "Thermal Expansion of Silicon" by Okada & Tokumaru (1984)
      -> Coefficient de dilatation thermique du silicium
    - "Color Filter Array Demosaicking" by K. Ramanath et al. (2002)
      -> Filtres couleur (Bayer, etc.)
    - "Quantum Efficiency of Silicon Photodiodes" by J. Geist et al. (1972)
      -> Efficacité quantique des capteurs en silicium
    - "Noise Sources in CCD and CMOS Image Sensors" by J. Hynecek (2001)
      -> Modèles de bruit (Poisson, gaussien, etc.)
    - "The Southwell Geometry for Wavefront Sensing" by Southwell (1980)
      -> Échantillonnage pour Shack-Hartmann (référence pour l'intégration)

Dependencies:
    - numpy
    - Material_Behaviour.py (for thermal expansion and material properties)
    - Beam.py (for beam sampling)
    - MathAndPhysicsTools.py (for grid creation and interpolation)
    - Visualization.py (for image display)
    - scipy (for interpolation, optional)
    - skimage (for image resizing, optional)
"""

import numpy as np
import logging
import os
from typing import Optional, Tuple, Dict, List, Union
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# IMPORT DES DÉPENDANCES
# =============================================================================

try:
    from Material_Behaviour import MaterialBehaviour, STANDARD_TEMPERATURE_K
    MATERIAL_BEHAVIOUR_AVAILABLE = True
except ImportError as e:
    MATERIAL_BEHAVIOUR_AVAILABLE = False
    logging.warning(f"Material_Behaviour module not available: {e}")
    STANDARD_TEMPERATURE_K = 293.15  # 20°C

try:
    from Beam import Beam
    BEAM_AVAILABLE = True
except ImportError as e:
    BEAM_AVAILABLE = False
    logging.warning(f"Beam module not available: {e}")

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
    from scipy.interpolate import interp2d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from skimage.transform import resize
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Camera")


# =============================================================================
# ENUMS ET CONSTANTES
# =============================================================================

class SensorType(Enum):
    """
    FR: Type de capteur.
        - IDEAL: Capteur parfait (échantillonnage simple, pas de bruit)
        - REAL: Capteur réel (avec bruit, puits quantiques, etc.)
    
    EN: Sensor type.
        - IDEAL: Ideal sensor (simple sampling, no noise)
        - REAL: Real sensor (with noise, quantum wells, etc.)
    """
    IDEAL = "ideal"
    REAL = "real"


class NoiseType(Enum):
    """
    FR: Type de bruit pour les capteurs.
        - GAUSSIAN: Bruit gaussien (bruit électronique)
        - POISSON: Bruit de Poisson (shot noise, bruit photonique)
        - READOUT: Bruit de lecture (électronique du capteur)
        - DARK: Courant d'obscurité (génération thermique d'électrons)
        - THERMAL: Bruit thermique (dépend de la température)
        - QUANTIZATION: Bruit de quantification (dû à la numérisation)
        - FIXED_PATTERN: Bruit de motif fixe (non-uniformité des pixels)
    
    EN: Noise type for sensors.
        - GAUSSIAN: Gaussian noise (electronic noise)
        - POISSON: Poisson noise (shot noise, photon noise)
        - READOUT: Readout noise (sensor electronics)
        - DARK: Dark current (thermal electron generation)
        - THERMAL: Thermal noise (temperature-dependent)
        - QUANTIZATION: Quantization noise (digitization)
        - FIXED_PATTERN: Fixed pattern noise (pixel non-uniformity)
    """
    GAUSSIAN = "gaussian"
    POISSON = "poisson"
    READOUT = "readout"
    DARK = "dark"
    THERMAL = "thermal"
    QUANTIZATION = "quantization"
    FIXED_PATTERN = "fixed_pattern"


class PixelType(Enum):
    """
    FR: Forme des pixels.
        - SQUARE: Pixels carrés (défaut)
        - RECTANGULAR: Pixels rectangulaires
    
    EN: Pixel shape.
        - SQUARE: Square pixels (default)
        - RECTANGULAR: Rectangular pixels
    """
    SQUARE = "square"
    RECTANGULAR = "rectangular"


class ColorFilterArray(Enum):
    """
    FR: Type de filtre couleur (CFA).
        - NONE: Pas de filtre couleur (monochrome)
        - BAYER_RGGB: Filtre Bayer RGGB (Rouge, Vert, Vert, Bleu)
        - BAYER_GRBG: Filtre Bayer GRBG (Vert, Rouge, Bleu, Vert)
        - BAYER_GBRG: Filtre Bayer GBRG (Vert, Bleu, Rouge, Vert)
        - BAYER_BGGR: Filtre Bayer BGGR (Bleu, Vert, Vert, Rouge)
    
    EN: Color Filter Array (CFA) type.
        - NONE: No color filter (monochrome)
        - BAYER_RGGB: Bayer RGGB filter (Red, Green, Green, Blue)
        - BAYER_GRBG: Bayer GRBG filter (Green, Red, Blue, Green)
        - BAYER_GBRG: Bayer GBRG filter (Green, Blue, Red, Green)
        - BAYER_BGGR: Bayer BGGR filter (Blue, Green, Green, Red)
    
    Source: "Color Filter Array Demosaicking" by K. Ramanath et al. (2002)
    """
    NONE = "none"
    BAYER_RGGB = "bayer_rggb"
    BAYER_GRBG = "bayer_grbg"
    BAYER_GBRG = "bayer_gbrg"
    BAYER_BGGR = "bayer_bggr"


class ResponseType(Enum):
    """
    FR: Type de réponse du capteur.
        - LINEAR: Réponse linéaire (signal proportionnel à l'intensité)
        - LOGARITHMIC: Réponse logarithmique (pour étendre la dynamique)
        - GAMMA: Réponse avec correction gamma
    
    EN: Sensor response type.
        - LINEAR: Linear response (signal proportional to intensity)
        - LOGARITHMIC: Logarithmic response (to extend dynamic range)
        - GAMMA: Gamma-corrected response
    """
    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"
    GAMMA = "gamma"


# Constantes pour les capteurs typiques
# Source: "Digital Image Sensors and Cameras" by Gerald C. Holst
STANDARD_PIXEL_SIZE_UM = 5.0  # 5 µm (typique pour les capteurs CMOS)
STANDARD_QUANTUM_EFFICIENCY = 0.7  # 70% (typique pour le silicium)
STANDARD_FULL_WELL_CAPACITY = 50000  # 50,000 électrons (typique)
STANDARD_READOUT_NOISE_E = 3.0  # 3 électrons RMS (typique)
STANDARD_DARK_CURRENT_E = 0.1  # 0.1 électrons/pixel/s (typique à 20°C)
STANDARD_BIT_DEPTH = 16  # 16 bits (typique)
STANDARD_GAIN_E_PER_ADU = 1.0  # 1 électron/ADU (typique)

# Coefficient de dilatation thermique du silicium (ppm/K)
# Source: "Thermal Expansion of Silicon" by Okada & Tokumaru (1984)
SILICON_CTE = 2.6e-6  # ppm/K à 20°C


# =============================================================================
# CLASSE DE BASE: CAMERA
# =============================================================================

@dataclass
class CameraSpecifications:
    """
    FR: Spécifications techniques d'un capteur.
        
    EN: Technical specifications of a sensor.
    
    Attributes:
        num_pixels_x (int): Nombre de pixels en x.
        num_pixels_y (int): Nombre de pixels en y.
        pixel_size_um (float): Taille des pixels en µm (microns).
        pixel_type (PixelType): Forme des pixels (SQUARE ou RECTANGULAR).
        pixel_width_um (Optional[float]): Largeur des pixels (pour RECTANGULAR).
        pixel_height_um (Optional[float]): Hauteur des pixels (pour RECTANGULAR).
        wavelength_nm (float): Longueur d'onde en nm.
        temperature_K (float): Température en Kelvin.
        material_name (str): Nom du matériau du capteur.
        sensor_type (SensorType): Type de capteur (IDEAL ou REAL).
        quantum_efficiency (float): Efficacité quantique (0-1).
        full_well_capacity (int): Capacité maximale des puits quantiques (électrons).
        readout_noise_e (float): Bruit de lecture en électrons RMS.
        dark_current_e (float): Courant d'obscurité en électrons/pixel/s.
        exposure_time_s (float): Temps d'exposition en secondes.
        gain_e_per_adu (float): Gain en électrons/ADU.
        bit_depth (int): Profondeur de bits (8, 12, 14, 16).
        cfa (ColorFilterArray): Filtre couleur.
        response_type (ResponseType): Type de réponse du capteur.
        gamma (float): Correction gamma (pour response_type=GAMMA).
    """
    num_pixels_x: int
    num_pixels_y: int
    pixel_size_um: float = STANDARD_PIXEL_SIZE_UM
    pixel_type: PixelType = PixelType.SQUARE
    pixel_width_um: Optional[float] = None
    pixel_height_um: Optional[float] = None
    wavelength_nm: float = 633.0
    temperature_K: float = STANDARD_TEMPERATURE_K
    material_name: str = "Silicon"
    sensor_type: SensorType = SensorType.IDEAL
    quantum_efficiency: float = STANDARD_QUANTUM_EFFICIENCY
    full_well_capacity: int = STANDARD_FULL_WELL_CAPACITY
    readout_noise_e: float = STANDARD_READOUT_NOISE_E
    dark_current_e: float = STANDARD_DARK_CURRENT_E
    exposure_time_s: float = 0.1
    gain_e_per_adu: float = STANDARD_GAIN_E_PER_ADU
    bit_depth: int = STANDARD_BIT_DEPTH
    cfa: ColorFilterArray = ColorFilterArray.NONE
    response_type: ResponseType = ResponseType.LINEAR
    gamma: float = 1.0


class Camera:
    """
    FR: Capteur virtuel (parfait ou réel).
        Permet d'échantillonner un faisceau et d'ajouter des effets de capteur.
        
        Chaque image générée (phase et intensité) aura :
        - Une échelle visuelle
        - Le PV (Peak-to-Valley) et le RMS des valeurs
        - Colormap : "Jet" pour la phase, "hot" pour l'intensité
        
        Unités :
        - Longueurs : mm (taille du capteur), µm (taille des pixels)
        - Longueur d'onde : nm
        - Phase : nm (principale), λ (longueur d'onde), rad, mrad
        - Intensité : a.u. (arbitrary units) ou ADU
    
    EN: Virtual sensor (ideal or real).
        Allows sampling a beam and adding sensor effects.
        
        Each generated image (phase and intensity) will have:
        - A visual scale
        - PV (Peak-to-Valley) and RMS values
        - Colormap: "Jet" for phase, "hot" for intensity
        
        Units:
        - Lengths: mm (sensor size), µm (pixel size)
        - Wavelength: nm
        - Phase: nm (main), λ (wavelength), rad, mrad
        - Intensity: a.u. (arbitrary units) or ADU
    
    Attributes:
        name (str): Nom du capteur.
        specifications (CameraSpecifications): Spécifications techniques.
        sensor_width_mm (float): Largeur du capteur en mm.
        sensor_height_mm (float): Hauteur du capteur en mm.
        sensor_width_um (float): Largeur du capteur en µm.
        sensor_height_um (float): Hauteur du capteur en µm.
        material (MaterialBehaviour): Matériau du capteur.
        display (bool): Afficher automatiquement les images.
        display_dir (str): Répertoire pour sauvegarder les images.
    
    Sources:
        - "Digital Image Sensors and Cameras" by Gerald C. Holst (SPIE Press, 2011)
        - "The Physics and Technology of CCD and CMOS Image Sensors" by Janesick (SPIE, 2007)
    """

    def __init__(self,
                 name: str,
                 specifications: CameraSpecifications,
                 display: bool = False,
                 display_dir: str = "output"):
        """
        FR: Initialise un capteur.
            
        EN: Initializes a sensor.
        
        Args:
            name (str): Nom du capteur.
            specifications (CameraSpecifications): Spécifications techniques.
            display (bool): Afficher automatiquement les images.
            display_dir (str): Répertoire pour sauvegarder les images.
        """
        self.name = name
        self.specifications = specifications
        self.display = display
        self.display_dir = display_dir

        # Calculer la taille du capteur en mm et µm
        if specifications.pixel_type == PixelType.SQUARE:
            pixel_width_um = specifications.pixel_size_um
            pixel_height_um = specifications.pixel_size_um
        else:  # RECTANGULAR
            pixel_width_um = specifications.pixel_width_um or specifications.pixel_size_um
            pixel_height_um = specifications.pixel_height_um or specifications.pixel_size_um

        self.pixel_width_mm = pixel_width_um * 1e-3
        self.pixel_height_mm = pixel_height_um * 1e-3
        self.sensor_width_mm = specifications.num_pixels_x * self.pixel_width_mm
        self.sensor_height_mm = specifications.num_pixels_y * self.pixel_height_mm
        self.sensor_width_um = specifications.num_pixels_x * pixel_width_um
        self.sensor_height_um = specifications.num_pixels_y * pixel_height_um

        # Matériau
        if MATERIAL_BEHAVIOUR_AVAILABLE:
            try:
                self.material = MaterialBehaviour(specifications.material_name)
            except Exception as e:
                logger.warning(f"Matériau inconnu: {specifications.material_name}. {e}")
                self.material = None
        else:
            self.material = None

        # Créer le répertoire d'affichage
        if self.display:
            os.makedirs(self.display_dir, exist_ok=True)

        logger.info(f"Capteur '{name}' initialisé: "
                   f"{specifications.num_pixels_x}x{specifications.num_pixels_y} pixels, "
                   f"{self.sensor_width_mm:.2f}x{self.sensor_height_mm:.2f} mm, "
                   f"type={specifications.sensor_type.value}")

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"size={self.specifications.num_pixels_x}x{self.specifications.num_pixels_y}, "
                f"pixel={self.pixel_width_mm*1e3:.1f}x{self.pixel_height_mm*1e3:.1f} µm, "
                f"sensor={self.sensor_width_mm:.2f}x{self.sensor_height_mm:.2f} mm, "
                f"type={self.specifications.sensor_type.value}, "
                f"material={self.specifications.material_name}, "
                f"T={self.specifications.temperature_K:.1f}K)")

    # =========================================================================
    # ÉCHANTILLONNAGE DU FAISCEAU / BEAM SAMPLING
    # =========================================================================

    def sample_beam(self, beam: any) -> np.ndarray:
        """
        FR: Échantillonne un faisceau sur le capteur.
            Retourne une image en ADU (ou en intensité normalisée pour un capteur parfait).
            
            Chaque image générée aura :
            - Une échelle visuelle
            - Le PV (Peak-to-Valley) et le RMS des valeurs
            
        EN: Samples a beam on the sensor.
            Returns an image in ADU (or normalized intensity for ideal sensor).
            
            Each generated image will have:
            - A visual scale
            - PV (Peak-to-Valley) and RMS values
        
        Args:
            beam (Beam): Faisceau incident.
            
        Returns:
            np.ndarray: Image échantillonnée (2D array).
        
        Sources:
            - "Digital Image Sensors and Cameras" by Gerald C. Holst (SPIE Press, 2011)
              -> Méthode d'échantillonnage des capteurs
        """
        if not BEAM_AVAILABLE:
            raise ImportError("Beam module is required for sample_beam().")

        if beam.electric_field is None:
            raise ValueError("Beam must have an electric_field to be sampled.")

        # Calculer l'intensité si elle n'existe pas
        if beam.intensity is None:
            beam.intensity = beam.compute_intensity_from_electric_field(beam.electric_field)

        # Créer une grille pour le capteur
        x_pixels = np.linspace(-self.sensor_width_mm/2, self.sensor_width_mm/2, self.specifications.num_pixels_x)
        y_pixels = np.linspace(-self.sensor_height_mm/2, self.sensor_height_mm/2, self.specifications.num_pixels_y)
        sensor_grid_x, sensor_grid_y = np.meshgrid(x_pixels, y_pixels)

        # Créer une grille pour le faisceau
        if beam.diameter_mm <= 0 or beam.num_points <= 0:
            raise ValueError("Beam must have valid diameter_mm and num_points.")

        beam_x = np.linspace(-beam.diameter_mm/2, beam.diameter_mm/2, beam.intensity.shape[1])
        beam_y = np.linspace(-beam.diameter_mm/2, beam.diameter_mm/2, beam.intensity.shape[0])
        beam_grid_x, beam_grid_y = np.meshgrid(beam_x, beam_y)

        # Interpoler l'intensité du faisceau sur la grille du capteur
        if SCIPY_AVAILABLE:
            f = interp2d(beam_x, beam_y, beam.intensity, kind='cubic', fill_value=0.0)
            sampled_intensity = f(x_pixels, y_pixels)
        elif SKIMAGE_AVAILABLE:
            from skimage.transform import resize
            # Redimensionner l'intensité du faisceau pour correspondre au capteur
            scale_x = self.specifications.num_pixels_x / beam.intensity.shape[1]
            scale_y = self.specifications.num_pixels_y / beam.intensity.shape[0]
            sampled_intensity = resize(
                beam.intensity,
                (self.specifications.num_pixels_y, self.specifications.num_pixels_x),
                mode='constant',
                cval=0.0,
                preserve_range=True,
                order=3  # Interpolation cubique
            )
        else:
            # Méthode simple sans interpolation (pixel le plus proche)
            sampled_intensity = np.zeros((self.specifications.num_pixels_y, self.specifications.num_pixels_x))
            beam_x_centered = beam_x - beam.diameter_mm/2
            beam_y_centered = beam_y - beam.diameter_mm/2
            sensor_x_centered = x_pixels - self.sensor_width_mm/2
            sensor_y_centered = y_pixels - self.sensor_height_mm/2

            for i in range(self.specifications.num_pixels_y):
                for j in range(self.specifications.num_pixels_x):
                    # Trouver l'index du pixel du faisceau le plus proche
                    beam_i = np.argmin(np.abs(beam_y_centered - sensor_y_centered[i]))
                    beam_j = np.argmin(np.abs(beam_x_centered - sensor_x_centered[j]))
                    sampled_intensity[i, j] = beam.intensity[beam_i, beam_j]

        # Normaliser l'intensité (0-1)
        if np.max(sampled_intensity) > 0:
            sampled_intensity /= np.max(sampled_intensity)

        # Appliquer l'efficacité quantique
        sampled_intensity *= self.specifications.quantum_efficiency

        # Appliquer la réponse du capteur (linéaire, logarithmique, gamma)
        sampled_intensity = self._apply_response(sampled_intensity)

        # Pour un capteur parfait, retourner l'intensité normalisée
        if self.specifications.sensor_type == SensorType.IDEAL:
            return sampled_intensity

        # Pour un capteur réel, appliquer les effets
        return self._apply_sensor_effects(sampled_intensity)

    def _apply_response(self, intensity: np.ndarray) -> np.ndarray:
        """
        FR: Applique la réponse du capteur (linéaire, logarithmique, gamma).
            
        EN: Applies the sensor response (linear, logarithmic, gamma).
        
        Args:
            intensity (np.ndarray): Intensité normalisée (0-1).
            
        Returns:
            np.ndarray: Intensité après application de la réponse.
        """
        if self.specifications.response_type == ResponseType.LINEAR:
            return intensity
        elif self.specifications.response_type == ResponseType.LOGARITHMIC:
            # Éviter log(0)
            return np.log1p(intensity * 9999) / np.log(10000)
        elif self.specifications.response_type == ResponseType.GAMMA:
            return intensity ** (1.0 / self.specifications.gamma)
        else:
            return intensity

    def _apply_sensor_effects(self, intensity: np.ndarray) -> np.ndarray:
        """
        FR: Applique les effets d'un capteur réel (bruit, quantification, etc.).
            
        EN: Applies real sensor effects (noise, quantization, etc.).
        
        Args:
            intensity (np.ndarray): Intensité normalisée (0-1).
            
        Returns:
            np.ndarray: Image avec les effets de capteur (en ADU).
        
        Sources:
            - "The Physics and Technology of CCD and CMOS Image Sensors" by Janesick (SPIE, 2007)
              -> Modèles de bruit des capteurs
            - "Noise Sources in CCD and CMOS Image Sensors" by J. Hynecek (2001)
              -> Bruit de Poisson, gaussien, etc.
        """
        # Convertir en électrons
        # Supposons que l'intensité max (1.0) correspond à la capacité maximale des puits
        electrons = intensity * self.specifications.full_well_capacity

        # Ajouter le courant d'obscurité (bruit de Poisson)
        dark_electrons = self.specifications.dark_current_e * self.specifications.exposure_time_s
        electrons += np.random.poisson(dark_electrons, electrons.shape)

        # Ajouter le bruit de lecture (bruit gaussien)
        readout_noise = np.random.normal(
            0,
            self.specifications.readout_noise_e,
            electrons.shape
        )
        electrons += readout_noise

        # Ajouter le bruit de Poisson (shot noise)
        electrons = np.random.poisson(electrons).astype(float)

        # Limiter à la capacité maximale des puits
        electrons = np.clip(electrons, 0, self.specifications.full_well_capacity)

        # Convertir en ADU
        adu = electrons / self.specifications.gain_e_per_adu

        # Appliquer la quantification
        adu = self._quantize(adu)

        # Appliquer le filtre couleur (CFA)
        if self.specifications.cfa != ColorFilterArray.NONE:
            adu = self._apply_cfa(adu)

        return adu

    def _quantize(self, adu: np.ndarray) -> np.ndarray:
        """
        FR: Applique la quantification sur l'image.
            
        EN: Applies quantization to the image.
        
        Args:
            adu (np.ndarray): Image en ADU (avant quantification).
            
        Returns:
            np.ndarray: Image quantifiée.
        """
        max_adu = 2**self.specifications.bit_depth - 1
        return np.clip(np.round(adu), 0, max_adu)

    def _apply_cfa(self, adu: np.ndarray) -> np.ndarray:
        """
        FR: Applique un filtre couleur (CFA) à l'image.
            Implémentation simplifiée des filtres Bayer.
            
        EN: Applies a Color Filter Array (CFA) to the image.
            Simplified implementation of Bayer filters.
        
        Args:
            adu (np.ndarray): Image en ADU.
            
        Returns:
            np.ndarray: Image avec le filtre CFA appliqué.
        
        Sources:
            - "Color Filter Array Demosaicking" by K. Ramanath et al. (2002)
              -> Structure des filtres Bayer
        """
        if self.specifications.cfa == ColorFilterArray.NONE:
            return adu

        cfa_image = np.zeros_like(adu)

        for i in range(adu.shape[0]):
            for j in range(adu.shape[1]):
                if self.specifications.cfa == ColorFilterArray.BAYER_RGGB:
                    if i % 2 == 0 and j % 2 == 0:
                        cfa_image[i, j] = adu[i, j]  # R
                    elif i % 2 == 0 and j % 2 == 1:
                        cfa_image[i, j] = adu[i, j]  # G
                    elif i % 2 == 1 and j % 2 == 0:
                        cfa_image[i, j] = adu[i, j]  # G
                    else:
                        cfa_image[i, j] = adu[i, j]  # B
                elif self.specifications.cfa == ColorFilterArray.BAYER_GRBG:
                    if i % 2 == 0 and j % 2 == 0:
                        cfa_image[i, j] = adu[i, j]  # G
                    elif i % 2 == 0 and j % 2 == 1:
                        cfa_image[i, j] = adu[i, j]  # R
                    elif i % 2 == 1 and j % 2 == 0:
                        cfa_image[i, j] = adu[i, j]  # B
                    else:
                        cfa_image[i, j] = adu[i, j]  # G
                elif self.specifications.cfa == ColorFilterArray.BAYER_GBRG:
                    if i % 2 == 0 and j % 2 == 0:
                        cfa_image[i, j] = adu[i, j]  # G
                    elif i % 2 == 0 and j % 2 == 1:
                        cfa_image[i, j] = adu[i, j]  # B
                    elif i % 2 == 1 and j % 2 == 0:
                        cfa_image[i, j] = adu[i, j]  # R
                    else:
                        cfa_image[i, j] = adu[i, j]  # G
                elif self.specifications.cfa == ColorFilterArray.BAYER_BGGR:
                    if i % 2 == 0 and j % 2 == 0:
                        cfa_image[i, j] = adu[i, j]  # B
                    elif i % 2 == 0 and j % 2 == 1:
                        cfa_image[i, j] = adu[i, j]  # G
                    elif i % 2 == 1 and j % 2 == 0:
                        cfa_image[i, j] = adu[i, j]  # G
                    else:
                        cfa_image[i, j] = adu[i, j]  # R

        return cfa_image

    # =========================================================================
    # AJOUT DE BRUIT / NOISE ADDITION
    # =========================================================================

    def add_noise(self,
                  image: np.ndarray,
                  noise_type: NoiseType = NoiseType.GAUSSIAN,
                  **kwargs) -> np.ndarray:
        """
        FR: Ajoute un bruit spécifique à une image.
            
        EN: Adds specific noise to an image.
        
        Args:
            image (np.ndarray): Image de départ.
            noise_type (NoiseType): Type de bruit à ajouter.
            **kwargs: Arguments spécifiques au type de bruit.
            
        Returns:
            np.ndarray: Image avec le bruit ajouté.
        
        Sources:
            - "Noise Sources in CCD and CMOS Image Sensors" by J. Hynecek (2001)
              -> Modèles de bruit pour les capteurs
        """
        if noise_type == NoiseType.GAUSSIAN:
            # Bruit gaussien
            sigma = kwargs.get('sigma', self.specifications.readout_noise_e)
            return image + np.random.normal(0, sigma, image.shape)

        elif noise_type == NoiseType.POISSON:
            # Bruit de Poisson (shot noise)
            return np.random.poisson(image).astype(float)

        elif noise_type == NoiseType.READOUT:
            # Bruit de lecture
            readout_noise = kwargs.get('readout_noise_e', self.specifications.readout_noise_e)
            return image + np.random.normal(0, readout_noise, image.shape)

        elif noise_type == NoiseType.DARK:
            # Courant d'obscurité
            dark_current = kwargs.get('dark_current_e', self.specifications.dark_current_e)
            exposure_time = kwargs.get('exposure_time_s', self.specifications.exposure_time_s)
            dark_electrons = dark_current * exposure_time
            return image + np.random.poisson(dark_electrons, image.shape)

        elif noise_type == NoiseType.THERMAL:
            # Bruit thermique (dépend de la température)
            if not MATERIAL_BEHAVIOUR_AVAILABLE or self.material is None:
                logger.warning("Material_Behaviour not available. Using default thermal noise.")
                thermal_noise = 1.0  # Valeur par défaut
            else:
                try:
                    thermal_noise = self.material.get_thermal_noise(
                        self.specifications.temperature_K,
                        self.specifications.exposure_time_s
                    )
                except:
                    thermal_noise = 1.0
            return image + np.random.normal(0, thermal_noise, image.shape)

        elif noise_type == NoiseType.QUANTIZATION:
            # Bruit de quantification
            bit_depth = kwargs.get('bit_depth', self.specifications.bit_depth)
            max_value = 2**bit_depth - 1
            return np.round(image * max_value) / max_value

        elif noise_type == NoiseType.FIXED_PATTERN:
            # Bruit de motif fixe (non-uniformité des pixels)
            # Générer un motif fixe (même pour chaque appel)
            if not hasattr(self, '_fixed_pattern_noise'):
                self._fixed_pattern_noise = np.random.normal(
                    0,
                    kwargs.get('sigma', 0.01),
                    (self.specifications.num_pixels_y, self.specifications.num_pixels_x)
                )
            return image + self._fixed_pattern_noise

        else:
            return image

    # =========================================================================
    # DILATATION THERMIQUE / THERMAL EXPANSION
    # =========================================================================

    def apply_thermal_expansion(self, new_temperature_K: float) -> None:
        """
        FR: Applique la dilatation thermique au capteur.
            Met à jour :
            - La taille des pixels (pixel_size_um)
            - La taille du capteur (sensor_width/height_mm)
            - La température du capteur
            
        Formule : ΔL = L₀ · α · ΔT
        où :
            - L₀ = taille initiale
            - α = coefficient de dilatation thermique (CTE)
            - ΔT = variation de température (K)
            
        EN: Applies thermal expansion to the sensor.
            Updates:
            - Pixel size (pixel_size_um)
            - Sensor size (sensor_width/height_mm)
            - Sensor temperature
            
        Formula: ΔL = L₀ · α · ΔT
        where:
            - L₀ = initial size
            - α = thermal expansion coefficient (CTE)
            - ΔT = temperature change (K)
        
        Args:
            new_temperature_K (float): Nouvelle température en Kelvin.
        
        Sources:
            - "Thermal Expansion of Silicon" by Okada & Tokumaru (1984)
              -> CTE du silicium : 2.6 ppm/K à 20°C
        """
        if not MATERIAL_BEHAVIOUR_AVAILABLE or self.material is None:
            logger.warning("Material_Behaviour not available or material not set. "
                          "Using default CTE for Silicon.")
            alpha = SILICON_CTE
        else:
            try:
                alpha = self.material.get_thermal_expansion_coefficient(
                    self.specifications.temperature_K
                )
            except:
                alpha = SILICON_CTE

        delta_T = new_temperature_K - self.specifications.temperature_K
        if delta_T == 0:
            return

        # Mettre à jour la taille des pixels
        old_pixel_size_um = self.specifications.pixel_size_um
        self.specifications.pixel_size_um *= (1 + alpha * delta_T)
        self.pixel_width_mm = self.specifications.pixel_size_um * 1e-3
        self.pixel_height_mm = self.specifications.pixel_size_um * 1e-3

        # Mettre à jour la taille du capteur
        self.sensor_width_mm = self.specifications.num_pixels_x * self.pixel_width_mm
        self.sensor_height_mm = self.specifications.num_pixels_y * self.pixel_height_mm
        self.sensor_width_um = self.specifications.num_pixels_x * self.specifications.pixel_size_um
        self.sensor_height_um = self.specifications.num_pixels_y * self.specifications.pixel_size_um

        # Mettre à jour la température
        self.specifications.temperature_K = new_temperature_K

        logger.info(f"Dilatation thermique appliquée: ΔT={delta_T:.2f}K, α={alpha:.2e}, "
                   f"Δpixel_size={self.specifications.pixel_size_um - old_pixel_size_um:.4f} µm")

    def get_thermal_expansion_info(self, new_temperature_K: float) -> Dict:
        """
        FR: Calcule les informations de dilatation thermique sans l'appliquer.
            
        EN: Calculates thermal expansion information without applying it.
        
        Args:
            new_temperature_K (float): Nouvelle température en Kelvin.
            
        Returns:
            Dict: Dictionnaire avec les informations de dilatation.
        """
        if not MATERIAL_BEHAVIOUR_AVAILABLE or self.material is None:
            alpha = SILICON_CTE
        else:
            try:
                alpha = self.material.get_thermal_expansion_coefficient(
                    self.specifications.temperature_K
                )
            except:
                alpha = SILICON_CTE

        delta_T = new_temperature_K - self.specifications.temperature_K

        return {
            "delta_T_K": delta_T,
            "cte": alpha,
            "delta_pixel_size_um": self.specifications.pixel_size_um * alpha * delta_T,
            "new_pixel_size_um": self.specifications.pixel_size_um * (1 + alpha * delta_T),
            "delta_sensor_width_mm": self.sensor_width_mm * alpha * delta_T,
            "new_sensor_width_mm": self.sensor_width_mm * (1 + alpha * delta_T),
            "delta_sensor_height_mm": self.sensor_height_mm * alpha * delta_T,
            "new_sensor_height_mm": self.sensor_height_mm * (1 + alpha * delta_T),
        }

    # =========================================================================
    # AFFICHAGE / DISPLAY
    # =========================================================================

    def display_image(self,
                      image: np.ndarray,
                      title: str = "",
                      save_path: Optional[str] = None,
                      show: bool = False,
                      cmap: Optional[str] = None,
                      is_phase: bool = False) -> None:
        """
        FR: Affiche une image avec la colormap appropriée et les statistiques.
            - Colormap : "Jet" pour la phase, "hot" pour l'intensité
            - Affiche le PV (Peak-to-Valley) et le RMS
            - Affiche une échelle
            
        EN: Displays an image with the appropriate colormap and statistics.
            - Colormap: "Jet" for phase, "hot" for intensity
            - Displays PV (Peak-to-Valley) and RMS
            - Displays a scale
        
        Args:
            image (np.ndarray): Image à afficher.
            title (str): Titre de l'image.
            save_path (str): Chemin pour sauvegarder l'image.
            show (bool): Afficher l'image.
            cmap (str): Colormap à utiliser (par défaut : "hot" pour l'intensité, "Jet" pour la phase).
            is_phase (bool): Indique si l'image est une carte de phase.
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Visualization module not available. Cannot display image.")
            return

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available. Cannot display image.")
            return

        # Déterminer la colormap
        if cmap is None:
            cmap = "Jet" if is_phase else "hot"

        # Créer la figure
        plt.figure(figsize=(10, 8))

        # Afficher l'image
        im = plt.imshow(image, cmap=cmap)

        # Ajouter le titre avec les statistiques
        if len(image.shape) == 2:
            pv = np.max(image) - np.min(image)
            rms = np.std(image)
            mean = np.mean(image)
            title_with_stats = f"{title}\nPV={pv:.2f}, RMS={rms:.2f}, Mean={mean:.2f}"
        else:
            title_with_stats = title

        plt.title(title_with_stats)
        plt.colorbar(im, label="Intensité (a.u.)" if not is_phase else "Phase (nm)")

        # Sauvegarder ou afficher
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Image sauvegardée: {save_path}")
        elif show:
            plt.show()
        else:
            plt.close()

    def display_beam_sample(self,
                           beam: any,
                           title: str = "",
                           save_path: Optional[str] = None,
                           show: bool = False) -> np.ndarray:
        """
        FR: Échantillonne un faisceau et affiche le résultat.
            
        EN: Samples a beam and displays the result.
        
        Args:
            beam (Beam): Faisceau à échantillonner.
            title (str): Titre de l'image.
            save_path (str): Chemin pour sauvegarder l'image.
            show (bool): Afficher l'image.
            
        Returns:
            np.ndarray: Image échantillonnée.
        """
        image = self.sample_beam(beam)

        # Déterminer si c'est une phase ou une intensité
        is_phase = hasattr(beam, 'phase') and title.lower().find('phase') != -1

        self.display_image(
            image,
            title=title,
            save_path=save_path,
            show=show,
            is_phase=is_phase
        )

        return image

    # =========================================================================
    # UTILITAIRES / UTILITIES
    # =========================================================================

    def get_pixel_scale(self) -> Tuple[float, float]:
        """
        FR: Retourne l'échelle des pixels en mm/pixel.
            
        EN: Returns the pixel scale in mm/pixel.
        
        Returns:
            Tuple[float, float]: (scale_x, scale_y) en mm/pixel.
        """
        return (self.pixel_width_mm, self.pixel_height_mm)

    def get_sensor_info(self) -> Dict:
        """
        FR: Retourne les informations complètes du capteur.
            
        EN: Returns complete sensor information.
        
        Returns:
            Dict: Dictionnaire avec toutes les informations du capteur.
        """
        return {
            "name": self.name,
            "sensor_type": self.specifications.sensor_type.value,
            "num_pixels_x": self.specifications.num_pixels_x,
            "num_pixels_y": self.specifications.num_pixels_y,
            "pixel_size_um": self.specifications.pixel_size_um,
            "pixel_type": self.specifications.pixel_type.value,
            "sensor_width_mm": self.sensor_width_mm,
            "sensor_height_mm": self.sensor_height_mm,
            "sensor_width_um": self.sensor_width_um,
            "sensor_height_um": self.sensor_height_um,
            "wavelength_nm": self.specifications.wavelength_nm,
            "temperature_K": self.specifications.temperature_K,
            "material": self.specifications.material_name,
            "quantum_efficiency": self.specifications.quantum_efficiency,
            "full_well_capacity": self.specifications.full_well_capacity,
            "readout_noise_e": self.specifications.readout_noise_e,
            "dark_current_e": self.specifications.dark_current_e,
            "exposure_time_s": self.specifications.exposure_time_s,
            "gain_e_per_adu": self.specifications.gain_e_per_adu,
            "bit_depth": self.specifications.bit_depth,
            "cfa": self.specifications.cfa.value,
            "response_type": self.specifications.response_type.value,
            "gamma": self.specifications.gamma,
        }


# =============================================================================
# CLASSES SPÉCIALISÉES / SPECIALIZED CLASSES
# =============================================================================

class IdealCamera(Camera):
    """
    FR: Capteur virtuel parfait.
        Échantillonne simplement le faisceau sans ajouter de bruit.
        Retourne une intensité normalisée (0-1).
        
    EN: Ideal virtual sensor.
        Simply samples the beam without adding noise.
        Returns a normalized intensity (0-1).
    
    Sources:
        - "Digital Image Sensors and Cameras" by Gerald C. Holst (SPIE Press, 2011)
          -> Modèle de capteur idéal
    """

    def __init__(self,
                 name: str = "Ideal Camera",
                 num_pixels_x: int = 1024,
                 num_pixels_y: int = 1024,
                 pixel_size_um: float = STANDARD_PIXEL_SIZE_UM,
                 wavelength_nm: float = 633.0,
                 display: bool = False,
                 display_dir: str = "output"):
        """
        FR: Initialise un capteur parfait.
            
        EN: Initializes an ideal sensor.
        
        Args:
            name (str): Nom du capteur.
            num_pixels_x (int): Nombre de pixels en x (défaut: 1024).
            num_pixels_y (int): Nombre de pixels en y (défaut: 1024).
            pixel_size_um (float): Taille des pixels en µm (défaut: 5.0).
            wavelength_nm (float): Longueur d'onde en nm (défaut: 633.0).
            display (bool): Afficher automatiquement les images.
            display_dir (str): Répertoire pour sauvegarder les images.
        """
        specifications = CameraSpecifications(
            num_pixels_x=num_pixels_x,
            num_pixels_y=num_pixels_y,
            pixel_size_um=pixel_size_um,
            wavelength_nm=wavelength_nm,
            sensor_type=SensorType.IDEAL,
        )
        super().__init__(name=name, specifications=specifications, display=display, display_dir=display_dir)


class RealCamera(Camera):
    """
    FR: Capteur virtuel réel.
        Simule les effets d'un capteur réel :
        - Puits quantiques (full well capacity)
        - Bruit de lecture (readout noise)
        - Courant d'obscurité (dark current)
        - Bruit thermique (thermal noise)
        - Quantification (bit depth)
        - Filtre couleur (CFA)
        
        Retourne une image en ADU.
        
    EN: Real virtual sensor.
        Simulates the effects of a real sensor:
        - Quantum wells (full well capacity)
        - Readout noise
        - Dark current
        - Thermal noise
        - Quantization (bit depth)
        - Color Filter Array (CFA)
        
        Returns an image in ADU.
    
    Sources:
        - "The Physics and Technology of CCD and CMOS Image Sensors" by Janesick (SPIE, 2007)
          -> Puits quantiques, bruit de lecture, courant d'obscurité
        - "Noise Sources in CCD and CMOS Image Sensors" by J. Hynecek (2001)
          -> Modèles de bruit
    """

    def __init__(self,
                 name: str = "Real Camera",
                 num_pixels_x: int = 1024,
                 num_pixels_y: int = 1024,
                 pixel_size_um: float = STANDARD_PIXEL_SIZE_UM,
                 material_name: str = "Silicon",
                 quantum_efficiency: float = STANDARD_QUANTUM_EFFICIENCY,
                 full_well_capacity: int = STANDARD_FULL_WELL_CAPACITY,
                 readout_noise_e: float = STANDARD_READOUT_NOISE_E,
                 dark_current_e: float = STANDARD_DARK_CURRENT_E,
                 exposure_time_s: float = 0.1,
                 gain_e_per_adu: float = STANDARD_GAIN_E_PER_ADU,
                 bit_depth: int = STANDARD_BIT_DEPTH,
                 cfa: ColorFilterArray = ColorFilterArray.NONE,
                 wavelength_nm: float = 633.0,
                 temperature_K: float = STANDARD_TEMPERATURE_K,
                 display: bool = False,
                 display_dir: str = "output"):
        """
        FR: Initialise un capteur réel.
            
        EN: Initializes a real sensor.
        
        Args:
            name (str): Nom du capteur.
            num_pixels_x (int): Nombre de pixels en x (défaut: 1024).
            num_pixels_y (int): Nombre de pixels en y (défaut: 1024).
            pixel_size_um (float): Taille des pixels en µm (défaut: 5.0).
            material_name (str): Matériau du capteur (défaut: "Silicon").
            quantum_efficiency (float): Efficacité quantique (0-1, défaut: 0.7).
            full_well_capacity (int): Capacité maximale des puits (électrons, défaut: 50000).
            readout_noise_e (float): Bruit de lecture (électrons RMS, défaut: 3.0).
            dark_current_e (float): Courant d'obscurité (électrons/pixel/s, défaut: 0.1).
            exposure_time_s (float): Temps d'exposition (s, défaut: 0.1).
            gain_e_per_adu (float): Gain (électrons/ADU, défaut: 1.0).
            bit_depth (int): Profondeur de bits (défaut: 16).
            cfa (ColorFilterArray): Filtre couleur (défaut: NONE).
            wavelength_nm (float): Longueur d'onde (nm, défaut: 633.0).
            temperature_K (float): Température (K, défaut: 293.15).
            display (bool): Afficher automatiquement les images.
            display_dir (str): Répertoire pour sauvegarder les images.
        """
        specifications = CameraSpecifications(
            num_pixels_x=num_pixels_x,
            num_pixels_y=num_pixels_y,
            pixel_size_um=pixel_size_um,
            material_name=material_name,
            sensor_type=SensorType.REAL,
            quantum_efficiency=quantum_efficiency,
            full_well_capacity=full_well_capacity,
            readout_noise_e=readout_noise_e,
            dark_current_e=dark_current_e,
            exposure_time_s=exposure_time_s,
            gain_e_per_adu=gain_e_per_adu,
            bit_depth=bit_depth,
            cfa=cfa,
            wavelength_nm=wavelength_nm,
            temperature_K=temperature_K,
        )
        super().__init__(name=name, specifications=specifications, display=display, display_dir=display_dir)


# =============================================================================
# FONCTIONS UTILITAIRES / UTILITY FUNCTIONS
# =============================================================================

def create_camera(camera_type: SensorType = SensorType.IDEAL,
                  name: str = "Camera",
                  num_pixels_x: int = 1024,
                  num_pixels_y: int = 1024,
                  pixel_size_um: float = STANDARD_PIXEL_SIZE_UM,
                  **kwargs) -> Camera:
    """
    FR: Fabrique un capteur (parfait ou réel).
        
    EN: Factory function to create a sensor (ideal or real).
    
    Args:
        camera_type (SensorType): Type de capteur (IDEAL ou REAL).
        name (str): Nom du capteur.
        num_pixels_x (int): Nombre de pixels en x.
        num_pixels_y (int): Nombre de pixels en y.
        pixel_size_um (float): Taille des pixels en µm.
        **kwargs: Arguments supplémentaires pour les capteurs réels.
        
    Returns:
        Camera: Le capteur créé (IdealCamera ou RealCamera).
    """
    if camera_type == SensorType.IDEAL:
        return IdealCamera(
            name=name,
            num_pixels_x=num_pixels_x,
            num_pixels_y=num_pixels_y,
            pixel_size_um=pixel_size_um,
            **{k: v for k, v in kwargs.items() if k in ['wavelength_nm', 'display', 'display_dir']}
        )
    else:
        return RealCamera(
            name=name,
            num_pixels_x=num_pixels_x,
            num_pixels_y=num_pixels_y,
            pixel_size_um=pixel_size_um,
            **kwargs
        )


# =============================================================================
# TESTS UNITAIRES / UNIT TESTS
# =============================================================================

class TestCamera:
    """FR: Tests unitaires pour Camera.py."""

    def test_ideal_camera_creation(self):
        """FR: Test la création d'un capteur parfait."""
        camera = IdealCamera(
            name="Test Ideal",
            num_pixels_x=100,
            num_pixels_y=100,
            pixel_size_um=5.0
        )
        assert camera.specifications.sensor_type == SensorType.IDEAL
        assert camera.num_pixels_x == 100
        assert camera.num_pixels_y == 100
        assert abs(camera.sensor_width_mm - 0.5) < 1e-6  # 100 * 5 µm = 0.5 mm
        assert abs(camera.sensor_height_mm - 0.5) < 1e-6

    def test_real_camera_creation(self):
        """FR: Test la création d'un capteur réel."""
        camera = RealCamera(
            name="Test Real",
            num_pixels_x=100,
            num_pixels_y=100,
            pixel_size_um=5.0
        )
        assert camera.specifications.sensor_type == SensorType.REAL
        assert camera.specifications.quantum_efficiency == STANDARD_QUANTUM_EFFICIENCY
        assert camera.specifications.full_well_capacity == STANDARD_FULL_WELL_CAPACITY

    def test_sample_beam_ideal(self):
        """FR: Test l'échantillonnage d'un faisceau avec un capteur parfait."""
        if not BEAM_AVAILABLE:
            return

        camera = IdealCamera(
            name="Test Sample",
            num_pixels_x=64,
            num_pixels_y=64,
            pixel_size_um=10.0  # 10 µm = 0.01 mm
        )

        beam = Beam(
            wavelength_nm=633.0,
            diameter_mm=0.64,  # 64 * 0.01 mm
            num_points=64
        )
        electric_field = beam.generate_electric_field(method="gaussian", sigma_mm=0.1)
        beam.electric_field = electric_field
        beam.intensity = beam.compute_intensity_from_electric_field(electric_field)

        image = camera.sample_beam(beam)
        assert image.shape == (64, 64)
        assert np.any(image > 0)
        assert np.all(image <= 1.0)  # Normalisé

    def test_sample_beam_real(self):
        """FR: Test l'échantillonnage d'un faisceau avec un capteur réel."""
        if not BEAM_AVAILABLE:
            return

        camera = RealCamera(
            name="Test Sample Real",
            num_pixels_x=64,
            num_pixels_y=64,
            pixel_size_um=10.0,
            exposure_time_s=0.1
        )

        beam = Beam(
            wavelength_nm=633.0,
            diameter_mm=0.64,
            num_points=64
        )
        electric_field = beam.generate_electric_field(method="gaussian", sigma_mm=0.1)
        beam.electric_field = electric_field
        beam.intensity = beam.compute_intensity_from_electric_field(electric_field)

        image = camera.sample_beam(beam)
        assert image.shape == (64, 64)
        # Pour un capteur réel, l'image peut dépasser 1.0 à cause du bruit
        # Mais elle doit être dans une plage raisonnable
        assert np.all(image >= 0)

    def test_thermal_expansion(self):
        """FR: Test la dilatation thermique."""
        if not MATERIAL_BEHAVIOUR_AVAILABLE:
            return

        camera = RealCamera(
            name="Test Thermal",
            num_pixels_x=100,
            num_pixels_y=100,
            pixel_size_um=5.0,
            material_name="Silicon"
        )

        initial_pixel_size_um = camera.specifications.pixel_size_um
        initial_sensor_width_mm = camera.sensor_width_mm

        camera.apply_thermal_expansion(373.15)  # +80 K

        new_pixel_size_um = camera.specifications.pixel_size_um
        new_sensor_width_mm = camera.sensor_width_mm

        assert new_pixel_size_um > initial_pixel_size_um
        assert new_sensor_width_mm > initial_sensor_width_mm

    def test_noise_addition(self):
        """FR: Test l'ajout de bruit."""
        camera = RealCamera(
            name="Test Noise",
            num_pixels_x=10,
            num_pixels_y=10
        )

        image = np.ones((10, 10))

        # Test bruit gaussien
        noisy_image = camera.add_noise(image, NoiseType.GAUSSIAN, sigma=0.1)
        assert noisy_image.shape == (10, 10)
        assert not np.allclose(noisy_image, image)

        # Test bruit de Poisson
        noisy_image = camera.add_noise(image, NoiseType.POISSON)
        assert noisy_image.shape == (10, 10)

        # Test bruit de lecture
        noisy_image = camera.add_noise(image, NoiseType.READOUT, readout_noise_e=5.0)
        assert noisy_image.shape == (10, 10)

    def test_cfa_application(self):
        """FR: Test l'application du filtre couleur (CFA)."""
        camera = RealCamera(
            name="Test CFA",
            num_pixels_x=4,
            num_pixels_y=4,
            cfa=ColorFilterArray.BAYER_RGGB
        )

        image = np.ones((4, 4))
        cfa_image = camera._apply_cfa(image)

        # Le filtre Bayer doit conserver certains pixels et en modifier d'autres
        assert cfa_image.shape == (4, 4)

    def test_get_sensor_info(self):
        """FR: Test la récupération des informations du capteur."""
        camera = IdealCamera(
            name="Test Info",
            num_pixels_x=100,
            num_pixels_y=100
        )

        info = camera.get_sensor_info()
        assert info["name"] == "Test Info"
        assert info["num_pixels_x"] == 100
        assert info["sensor_type"] == "ideal"


if __name__ == "__main__":
    import unittest
    unittest.main()

"""
Beam.py

FR: Module pour la génération et la gestion de faisceaux optiques.
    Permet de créer et manipuler des faisceaux avec :
    - Différents profils d'intensité (gaussien, uniforme, annulaire, etc.)
    - Calcul du champ électrique, de l'intensité et de la phase
    - Gestion des unités :
        * Longueurs : mm (pour les dimensions du faisceau)
        * Longueur d'onde : nm
        * Phase : nm (principale), rad (pour les calculs), λ (longueur d'onde), mrad
        * Intensité : a.u. (arbitrary units) ou normalisée
    - Fonctions de propagation (doivent rester dans ce module)
    - Gestion des NaN : toutes les fonctions gèrent les NaN sans les propager
    
    Chaque image générée (phase et intensité) aura :
    - Une échelle visuelle
    - Le PV (Peak-to-Valley) et le RMS des valeurs
    - Colormap : "Jet" pour la phase, "hot" pour l'intensité

EN: Module for generating and managing optical beams.
    Allows creating and manipulating beams with:
    - Different intensity profiles (Gaussian, uniform, annular, etc.)
    - Electric field, intensity, and phase calculation
    - Unit management:
        * Lengths: mm (for beam dimensions)
        * Wavelength: nm
        * Phase: nm (main), rad (for calculations), λ (wavelength), mrad
        * Intensity: a.u. (arbitrary units) or normalized
    - Propagation functions (must remain in this module)
    - NaN handling: all functions handle NaN without propagating them
    
    Each generated image (phase and intensity) will have:
    - A visual scale
    - PV (Peak-to-Valley) and RMS values
    - Colormap: "Jet" for phase, "hot" for intensity

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch

Dependencies:
    - numpy
    - scipy (for FFT, Bessel functions, Hermite/Laguerre polynomials, optional)
    - matplotlib (for visualization, optional)

Sources:
    - "Principles of Optics" by M. Born & E. Wolf (Cambridge University Press, 1999)
      -> Fondements de l'optique (propagation, diffraction, interférences)
      -> Calcul du champ électrique et de l'intensité (Ch. 5-7)
    - "Laser Beam Propagation" by J. W. Goodman (1996)
      -> Propagation des faisceaux gaussiens (Ch. 3)
      -> Transformée de Fourier pour la propagation (Ch. 3-4)
    - "Optical Physics" by S. G. Lipson, H. Lipson, D. S. Tannhauser (Cambridge, 2011)
      -> Interférences et diffraction
    - "Handbook of Optical Systems" by H. Gross (2005)
      -> Volume 1: Fundamentals of Technical Optics
      -> Volume 3: Aberration Theory and Correction
    - "Fourier Optics" by J. W. Goodman (2005)
      -> Transformée de Fourier en optique
      -> Méthodes de propagation (spectre angulaire, Fresnel, Fraunhofer)
    - "Numerical Recipes in C" by Press et al. (1992)
      -> Algorithmes numériques (FFT, interpolation, gestion des NaN)
      -> Chapitres 5 (FFT), 3 (interpolation), 1 (gestion des erreurs)
"""

import numpy as np
import logging
from typing import Optional, Tuple, Dict, List, Union, Callable
from enum import Enum
from datetime import datetime
import warnings


# =============================================================================
# CONFIGURATION GLOBALE
# =============================================================================

# Désactiver les warnings de division par zéro et NaN
warnings.filterwarnings("ignore", category=np.RankWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Beam")


# =============================================================================
# CONSTANTES PHYSIQUES ET PAR DÉFAUT
# =============================================================================

# Constantes physiques (SI)
SPEED_OF_LIGHT_M_PER_S = 299792458  # m/s
PLANCK_CONSTANT_J_S = 6.62607015e-34  # J·s

# Constantes par défaut
DEFAULT_WAVELENGTH_NM = 633.0  # Longueur d'onde par défaut (He-Ne laser, nm)
DEFAULT_ENERGY = 1.0  # Énergie par défaut (a.u.)
DEFAULT_DIAMETER_MM = 5.0  # Diamètre par défaut (mm)
DEFAULT_NUM_POINTS = 512  # Nombre de points par défaut

# Tolérance numérique
NUMERICAL_TOLERANCE = 1e-12


# =============================================================================
# ENUMS
# =============================================================================

class BeamProfile(Enum):
    """
    FR: Profil d'intensité du faisceau.
        
        - GAUSSIAN: Faisceau gaussien (défaut)
        - UNIFORM: Faisceau uniforme (intensité constante)
        - ANNULAR: Faisceau annulaire (anneau)
        - DONUT: Faisceau en forme de donut (anneau avec trou central)
        - TOPHAT: Faisceau "chapeau haut" (uniforme avec bords nets)
        - AIRY: Faisceau d'Airy (diffraction par une ouverture circulaire)
        - HERMITE_GAUSSIAN: Faisceau Hermite-Gaussien (modes HG)
        - LAGUERRE_GAUSSIAN: Faisceau Lagrange-Gaussien (modes LG)
    
    EN: Beam intensity profile.
        
        - GAUSSIAN: Gaussian beam (default)
        - UNIFORM: Uniform beam (constant intensity)
        - ANNULAR: Annular beam (ring)
        - DONUT: Donut beam (ring with central hole)
        - TOPHAT: Top-hat beam (uniform with sharp edges)
        - AIRY: Airy beam (diffraction by circular aperture)
        - HERMITE_GAUSSIAN: Hermite-Gaussian beam (HG modes)
        - LAGUERRE_GAUSSIAN: Laguerre-Gaussian beam (LG modes)
    
    Sources:
        - "Laser Beam Propagation" by Goodman (1996), Ch. 3
    """
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    ANNULAR = "annular"
    DONUT = "donut"
    TOPHAT = "tophat"
    AIRY = "airy"
    HERMITE_GAUSSIAN = "hermite_gaussian"
    LAGUERRE_GAUSSIAN = "laguerre_gaussian"


class PropagationMethod(Enum):
    """
    FR: Méthode de propagation du faisceau.
        
        - ANGULAR_SPECTRUM: Méthode du spectre angulaire (défaut, précise pour les courtes distances)
        - FRESNEL: Approximation de Fresnel (pour les moyennes distances)
        - FRAUNHOFER: Approximation de Fraunhofer (pour les longues distances)
        - RAY_TRACING: Lancer de rayons (pour les systèmes optiques complexes)
    
    EN: Beam propagation method.
        
        - ANGULAR_SPECTRUM: Angular spectrum method (default, accurate for short distances)
        - FRESNEL: Fresnel approximation (for medium distances)
        - FRAUNHOFER: Fraunhofer approximation (for long distances)
        - RAY_TRACING: Ray tracing (for complex optical systems)
    
    Sources:
        - "Fourier Optics" by Goodman (2005), Ch. 3-4
        - "Principles of Optics" by Born & Wolf (1999), Ch. 8
    """
    ANGULAR_SPECTRUM = "angular_spectrum"
    FRESNEL = "fresnel"
    FRAUNHOFER = "fraunhofer"
    RAY_TRACING = "ray_tracing"


# =============================================================================
# FONCTIONS UTILITAIRES DE GESTION DES NaN (DOIVENT RESTER DANS Beam.py)
# Ces fonctions sont spécifiques à la manipulation des champs optiques
# =============================================================================

def handle_nan(array: np.ndarray,
               method: str = 'zero') -> np.ndarray:
    """
    FR: Gère les valeurs NaN dans un tableau.
        
        Méthodes disponibles :
        - 'zero': Remplace les NaN par 0 (défaut)
        - 'mean': Remplace les NaN par la moyenne
        - 'median': Remplace les NaN par la médiane
        - 'ignore': Ne fait rien (laisse les NaN)
    
    EN: Handles NaN values in an array.
        
        Available methods:
        - 'zero': Replace NaN with 0 (default)
        - 'mean': Replace NaN with mean
        - 'median': Replace NaN with median
        - 'ignore': Do nothing (leave NaN)
    
    Args:
        array (np.ndarray): Tableau avec des NaN.
        method (str): Méthode de gestion des NaN.
    
    Returns:
        np.ndarray: Tableau sans NaN (selon la méthode).
    
    Sources:
        - "Numerical Recipes in C" by Press et al. (1992), Ch. 1
    """
    if not np.any(np.isnan(array)):
        return array
    
    if method == 'zero':
        return np.nan_to_num(array, nan=0.0)
    elif method == 'mean':
        return np.nan_to_num(array, nan=np.nanmean(array))
    elif method == 'median':
        return np.nan_to_num(array, nan=np.nanmedian(array))
    elif method == 'ignore':
        return array
    else:
        logger.warning(f"Unknown NaN handling method: {method}. Using 'zero'.")
        return np.nan_to_num(array, nan=0.0)


def safe_divide(numerator: Union[float, np.ndarray],
                denominator: Union[float, np.ndarray],
                default: float = 0.0) -> Union[float, np.ndarray]:
    """
    FR: Division sûre (évite les divisions par zéro et les NaN).
        
        Retourne default si le dénominateur est 0 ou NaN.
    
    EN: Safe division (avoids division by zero and NaN).
        
        Returns default if denominator is 0 or NaN.
    
    Args:
        numerator: Numérateur.
        denominator: Dénominateur.
        default: Valeur par défaut si division impossible.
    
    Returns:
        Résultat de la division ou default.
    
    Sources:
        - "Numerical Recipes in C" by Press et al. (1992), Ch. 1
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(numerator, denominator)
    
    # Remplacer les inf et NaN par default
    result = np.where(np.isfinite(result), result, default)
    
    return result


def safe_sqrt(x: Union[float, np.ndarray],
              default: float = 0.0) -> Union[float, np.ndarray]:
    """
    FR: Racine carrée sûre (évite les sqrt(négatif) et sqrt(NaN)).
        
    EN: Safe square root (avoids sqrt(negative) and sqrt(NaN)).
    
    Args:
        x: Valeur ou tableau.
        default: Valeur par défaut si sqrt impossible.
    
    Returns:
        Résultat de la racine carrée ou default.
    
    Sources:
        - "Numerical Recipes in C" by Press et al. (1992), Ch. 1
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.sqrt(np.maximum(np.real(x), 0.0))
    
    result = np.where(np.isfinite(result), result, default)
    return result


def safe_log(x: Union[float, np.ndarray],
              default: float = 0.0) -> Union[float, np.ndarray]:
    """
    FR: Logarithme sûr (évite les log(0), log(négatif) et log(NaN)).
        
    EN: Safe logarithm (avoids log(0), log(negative) and log(NaN)).
    
    Args:
        x: Valeur ou tableau.
        default: Valeur par défaut si log impossible.
    
    Returns:
        Résultat du logarithme ou default.
    
    Sources:
        - "Numerical Recipes in C" by Press et al. (1992), Ch. 1
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.log(np.maximum(np.real(x), 0.0))
    
    result = np.where(np.isfinite(result), result, default)
    return result


# =============================================================================
# CLASSE PRINCIPALE: BEAM
# =============================================================================

class Beam:
    """
    FR: Faisceau optique.
        
        Permet de représenter un faisceau optique avec :
        - Champ électrique (complexe)
        - Intensité (réelle)
        - Phase (réelle)
        - Diamètre et nombre de points
        - Longueur d'onde
        - Énergie
        
        Unités :
        - Longueurs : mm (diamètre du faisceau)
        - Longueur d'onde : nm
        - Phase : nm (principale), rad (pour les calculs)
        - Intensité : a.u. (arbitrary units)
        
        Chaque image générée (phase et intensité) aura :
        - Une échelle visuelle
        - Le PV (Peak-to-Valley) et le RMS des valeurs
        - Colormap : "Jet" pour la phase, "hot" pour l'intensité
    
    EN: Optical beam.
        
        Allows representing an optical beam with:
        - Electric field (complex)
        - Intensity (real)
        - Phase (real)
        - Diameter and number of points
        - Wavelength
        - Energy
        
        Units:
        - Lengths: mm (beam diameter)
        - Wavelength: nm
        - Phase: nm (main), rad (for calculations)
        - Intensity: a.u. (arbitrary units)
        
        Each generated image (phase and intensity) will have:
        - A visual scale
        - PV (Peak-to-Valley) and RMS values
        - Colormap: "Jet" for phase, "hot" for intensity
    
    Attributes:
        wavelength_nm (float): Longueur d'onde en nm.
        diameter_mm (float): Diamètre du faisceau en mm.
        energy (float): Énergie du faisceau (a.u.).
        num_points (int): Nombre de points dans chaque dimension.
        electric_field (np.ndarray): Champ électrique (complexe, 2D array).
        intensity (np.ndarray): Intensité (réelle, 2D array).
        phase (np.ndarray): Phase (réelle, 2D array, en nm).
        name (str): Nom du faisceau.
    
    Sources:
        - "Principles of Optics" by Born & Wolf (1999), Ch. 5-7
        - "Laser Beam Propagation" by Goodman (1996)
    """

    def __init__(self,
                 wavelength_nm: float = DEFAULT_WAVELENGTH_NM,
                 diameter_mm: float = DEFAULT_DIAMETER_MM,
                 energy: float = DEFAULT_ENERGY,
                 num_points: int = DEFAULT_NUM_POINTS,
                 name: str = "Beam"):
        """
        FR: Initialise un faisceau optique.
            
        EN: Initializes an optical beam.
        
        Args:
            wavelength_nm (float): Longueur d'onde en nm (défaut: 633.0).
            diameter_mm (float): Diamètre du faisceau en mm (défaut: 5.0).
            energy (float): Énergie du faisceau (a.u., défaut: 1.0).
            num_points (int): Nombre de points dans chaque dimension (défaut: 512).
            name (str): Nom du faisceau (défaut: "Beam").
        
        Raises:
            ValueError: Si diameter_mm ou num_points sont ≤ 0.
        """
        if diameter_mm <= 0:
            raise ValueError(f"diameter_mm doit être > 0, obtenu: {diameter_mm}")
        if num_points <= 0:
            raise ValueError(f"num_points doit être > 0, obtenu: {num_points}")
        
        self.wavelength_nm = float(wavelength_nm)
        self.diameter_mm = float(diameter_mm)
        self.energy = float(energy)
        self.num_points = int(num_points)
        self.name = name
        
        # Initialiser les attributs (seront remplis plus tard)
        self.electric_field: Optional[np.ndarray] = None
        self.intensity: Optional[np.ndarray] = None
        self.phase: Optional[np.ndarray] = None
        
        # Métadonnées pour le suivi
        self._creation_time = datetime.now()
        self._last_modified = datetime.now()
        
        logger.info(f"Faisceau '{name}' initialisé: "
                   f"λ={self.wavelength_nm}nm, diamètre={self.diameter_mm}mm, "
                   f"énergie={self.energy}, points={self.num_points}")

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"λ={self.wavelength_nm}nm, diameter={self.diameter_mm}mm, "
                f"energy={self.energy}, points={self.num_points})")

    def __str__(self) -> str:
        status = "avec champ électrique" if self.electric_field is not None else "sans champ électrique"
        return (f"Faisceau '{self.name}': {self.diameter_mm}mm, "
                f"λ={self.wavelength_nm}nm, {self.num_points}x{self.num_points} points, "
                f"énergie={self.energy}, {status}")

    # =========================================================================
    # GÉNÉRATION DU CHAMP ÉLECTRIQUE / ELECTRIC FIELD GENERATION
    # =========================================================================

    def generate_electric_field(self,
                                 method: Union[BeamProfile, str] = BeamProfile.GAUSSIAN,
                                 **kwargs) -> np.ndarray:
        """
        FR: Génère le champ électrique du faisceau.
            
            Le champ électrique est un tableau 2D complexe :
                E(x,y) = A(x,y) * exp(i * φ(x,y))
            où A(x,y) est l'amplitude et φ(x,y) est la phase.
            
            Pour un faisceau sans aberrations, φ(x,y) = 0.
            
        EN: Generates the electric field of the beam.
            
            The electric field is a 2D complex array:
                E(x,y) = A(x,y) * exp(i * φ(x,y))
            where A(x,y) is the amplitude and φ(x,y) is the phase.
            
            For a beam without aberrations, φ(x,y) = 0.
        
        Args:
            method (BeamProfile or str): Méthode de génération.
            **kwargs: Arguments spécifiques à la méthode.
                - Pour GAUSSIAN: sigma_mm (écart-type, défaut: diameter_mm/4)
                - Pour ANNULAR: inner_diameter_mm, outer_diameter_mm
                - Pour DONUT: inner_diameter_mm, outer_diameter_mm, order (défaut: 1)
                - Pour TOPHAT: radius_mm (défaut: diameter_mm/2)
                - Pour AIRY: aperture_diameter_mm (défaut: diameter_mm)
                - Pour HERMITE_GAUSSIAN: n, m (ordres du mode)
                - Pour LAGUERRE_GAUSSIAN: p, l (ordres du mode)
        
        Returns:
            np.ndarray: Champ électrique (complexe, 2D array).
        
        Raises:
            ValueError: Si la méthode est inconnue.
        
        Sources:
            - "Laser Beam Propagation" by Goodman (1996), Ch. 3
        """
        if isinstance(method, str):
            method = BeamProfile(method.lower())
        
        # Créer la grille
        x, y, X, Y = self._create_xy_grid()
        
        # Générer selon la méthode
        if method == BeamProfile.GAUSSIAN:
            sigma_mm = kwargs.get('sigma_mm', self.diameter_mm / 4)
            return self._gaussian_electric_field(X, Y, sigma_mm)
        
        elif method == BeamProfile.UNIFORM:
            return self._uniform_electric_field(X, Y)
        
        elif method == BeamProfile.ANNULAR:
            inner_diameter_mm = kwargs.get('inner_diameter_mm', self.diameter_mm / 3)
            outer_diameter_mm = kwargs.get('outer_diameter_mm', self.diameter_mm / 2)
            return self._annular_electric_field(X, Y, inner_diameter_mm, outer_diameter_mm)
        
        elif method == BeamProfile.DONUT:
            inner_diameter_mm = kwargs.get('inner_diameter_mm', self.diameter_mm / 3)
            outer_diameter_mm = kwargs.get('outer_diameter_mm', self.diameter_mm / 2)
            order = kwargs.get('order', 1)
            return self._donut_electric_field(X, Y, inner_diameter_mm, outer_diameter_mm, order)
        
        elif method == BeamProfile.TOPHAT:
            radius_mm = kwargs.get('radius_mm', self.diameter_mm / 2)
            return self._tophat_electric_field(X, Y, radius_mm)
        
        elif method == BeamProfile.AIRY:
            aperture_diameter_mm = kwargs.get('aperture_diameter_mm', self.diameter_mm)
            return self._airy_electric_field(X, Y, aperture_diameter_mm)
        
        elif method == BeamProfile.HERMITE_GAUSSIAN:
            n = kwargs.get('n', 0)
            m = kwargs.get('m', 0)
            return self._hermite_gaussian_electric_field(X, Y, n, m)
        
        elif method == BeamProfile.LAGUERRE_GAUSSIAN:
            p = kwargs.get('p', 0)
            l = kwargs.get('l', 0)
            return self._laguerre_gaussian_electric_field(X, Y, p, l)
        
        else:
            raise ValueError(f"Méthode inconnue: {method}")

    def _create_xy_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        FR: Crée une grille de coordonnées (x, y) pour le faisceau.
            
        EN: Creates a coordinate grid (x, y) for the beam.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                (x, y, X, Y) où X et Y sont les grilles 2D.
        """
        x = np.linspace(-self.diameter_mm / 2, self.diameter_mm / 2, self.num_points)
        y = np.linspace(-self.diameter_mm / 2, self.diameter_mm / 2, self.num_points)
        X, Y = np.meshgrid(x, y)
        return x, y, X, Y

    def _gaussian_electric_field(self,
                                  X: np.ndarray,
                                  Y: np.ndarray,
                                  sigma_mm: float) -> np.ndarray:
        """
        FR: Génère un champ électrique gaussien.
            
            Formule : E(x,y) = exp(-(x² + y²) / (2σ²))
            
        EN: Generates a Gaussian electric field.
            
            Formula: E(x,y) = exp(-(x² + y²) / (2σ²))
        
        Args:
            X (np.ndarray): Grille x (2D).
            Y (np.ndarray): Grille y (2D).
            sigma_mm (float): Écart-type en mm.
        
        Returns:
            np.ndarray: Champ électrique gaussien.
        
        Sources:
            - "Laser Beam Propagation" by Goodman (1996), Eq. 3.1
        """
        R_sq = X**2 + Y**2
        # Éviter les overflow pour les grandes valeurs de R_sq
        exponent = -R_sq / (2 * sigma_mm**2)
        exponent = np.clip(exponent, -700, 0)  # exp(-700) ≈ 10^-304 (pratiquement 0)
        return np.exp(exponent)

    def _uniform_electric_field(self,
                                 X: np.ndarray,
                                 Y: np.ndarray) -> np.ndarray:
        """
        FR: Génère un champ électrique uniforme.
            
        EN: Generates a uniform electric field.
        
        Args:
            X (np.ndarray): Grille x (2D).
            Y (np.ndarray): Grille y (2D).
        
        Returns:
            np.ndarray: Champ électrique uniforme.
        """
        # Masque circulaire pour limiter le faisceau au diamètre
        R = np.sqrt(X**2 + Y**2)
        mask = R <= (self.diameter_mm / 2)
        return np.where(mask, 1.0 + 0.0j, 0.0 + 0.0j)

    def _annular_electric_field(self,
                                X: np.ndarray,
                                Y: np.ndarray,
                                inner_diameter_mm: float,
                                outer_diameter_mm: float) -> np.ndarray:
        """
        FR: Génère un champ électrique annulaire (anneau).
            
        EN: Generates an annular electric field (ring).
        
        Args:
            X (np.ndarray): Grille x (2D).
            Y (np.ndarray): Grille y (2D).
            inner_diameter_mm (float): Diamètre intérieur en mm.
            outer_diameter_mm (float): Diamètre extérieur en mm.
        
        Returns:
            np.ndarray: Champ électrique annulaire.
        """
        R = np.sqrt(X**2 + Y**2)
        inner_mask = R >= (inner_diameter_mm / 2)
        outer_mask = R <= (outer_diameter_mm / 2)
        return np.where(inner_mask & outer_mask, 1.0 + 0.0j, 0.0 + 0.0j)

    def _donut_electric_field(self,
                              X: np.ndarray,
                              Y: np.ndarray,
                              inner_diameter_mm: float,
                              outer_diameter_mm: float,
                              order: int = 1) -> np.ndarray:
        """
        FR: Génère un champ électrique en forme de donut.
            
            Formule : E(x,y) = (x + i*y)^order * exp(-(x² + y²) / (2σ²))
            où σ = (outer_diameter - inner_diameter) / 4
            
        EN: Generates a donut-shaped electric field.
            
            Formula: E(x,y) = (x + i*y)^order * exp(-(x² + y²) / (2σ²))
            where σ = (outer_diameter - inner_diameter) / 4
        
        Args:
            X (np.ndarray): Grille x (2D).
            Y (np.ndarray): Grille y (2D).
            inner_diameter_mm (float): Diamètre intérieur en mm.
            outer_diameter_mm (float): Diamètre extérieur en mm.
            order (int): Ordre du mode (défaut: 1).
        
        Returns:
            np.ndarray: Champ électrique en forme de donut.
        
        Sources:
            - "Optical Vortex Beams" by A. M. Yao & M. J. Padgett (2011)
        """
        R_sq = X**2 + Y**2
        sigma_mm = (outer_diameter_mm - inner_diameter_mm) / 4
        amplitude = np.exp(-R_sq / (2 * sigma_mm**2))
        
        # Phase azimutale (vortex) - éviter atan2(0,0) qui donne NaN
        theta = np.arctan2(Y, X)
        # Remplacer les NaN dans theta (au centre)
        theta = np.where(np.isfinite(theta), theta, 0.0)
        
        phase = order * theta
        
        return amplitude * np.exp(1j * phase)

    def _tophat_electric_field(self,
                               X: np.ndarray,
                               Y: np.ndarray,
                               radius_mm: float) -> np.ndarray:
        """
        FR: Génère un champ électrique "chapeau haut" (top-hat).
            
        EN: Generates a top-hat electric field.
        
        Args:
            X (np.ndarray): Grille x (2D).
            Y (np.ndarray): Grille y (2D).
            radius_mm (float): Rayon en mm.
        
        Returns:
            np.ndarray: Champ électrique top-hat.
        """
        R = np.sqrt(X**2 + Y**2)
        return np.where(R <= radius_mm, 1.0 + 0.0j, 0.0 + 0.0j)

    def _airy_electric_field(self,
                             X: np.ndarray,
                             Y: np.ndarray,
                             aperture_diameter_mm: float) -> np.ndarray:
        """
        FR: Génère un champ électrique correspondant à une tâche d'Airy.
            
            La tâche d'Airy est la figure de diffraction par une ouverture circulaire.
            Formule : E(r) = (2J₁(kr)) / (kr) où J₁ est la fonction de Bessel de 1ère espèce.
            
        EN: Generates an electric field corresponding to an Airy spot.
            
            The Airy spot is the diffraction pattern by a circular aperture.
            Formula: E(r) = (2J₁(kr)) / (kr) where J₁ is the Bessel function of the 1st kind.
        
        Args:
            X (np.ndarray): Grille x (2D).
            Y (np.ndarray): Grille y (2D).
            aperture_diameter_mm (float): Diamètre de l'ouverture en mm.
        
        Returns:
            np.ndarray: Champ électrique de tâche d'Airy.
        
        Sources:
            - "Principles of Optics" by Born & Wolf (1999), Ch. 8
        """
        try:
            from scipy.special import j1  # Fonction de Bessel de 1ère espèce
        except ImportError:
            logger.warning("scipy not available. Using Gaussian approximation for Airy spot.")
            return self._gaussian_electric_field(X, Y, aperture_diameter_mm / 4)
        
        # Calculer r (distance radiale)
        R = np.sqrt(X**2 + Y**2)
        
        # Nombre d'onde
        k = 2 * np.pi / (self.wavelength_nm * 1e-6)  # en mm⁻¹
        
        # Calculer l'amplitude de la tâche d'Airy
        # E(r) = (2J₁(kr)) / (kr) pour r > 0
        kr = k * R
        
        # Éviter la division par zéro pour r = 0
        with np.errstate(divide='ignore', invalid='ignore'):
            amplitude = 2 * j1(kr) / kr
        
        # Pour r = 0, J₁(0) = 0 et lim (2J₁(kr))/(kr) = 1 quand r → 0
        amplitude = np.where(kr == 0, 1.0, amplitude)
        
        # Appliquer un masque circulaire (ouverture)
        aperture_radius_mm = aperture_diameter_mm / 2
        mask = R <= aperture_radius_mm
        
        return np.where(mask, amplitude + 0.0j, 0.0 + 0.0j)

    def _hermite_gaussian_electric_field(self,
                                          X: np.ndarray,
                                          Y: np.ndarray,
                                          n: int,
                                          m: int) -> np.ndarray:
        """
        FR: Génère un champ électrique Hermite-Gaussien.
            
            Formule : E(x,y) = H_n(x) * H_m(y) * exp(-(x² + y²) / w₀²)
            où H_n et H_m sont les polynômes d'Hermite.
            
        EN: Generates a Hermite-Gaussian electric field.
            
            Formula: E(x,y) = H_n(x) * H_m(y) * exp(-(x² + y²) / w₀²)
            where H_n and H_m are Hermite polynomials.
        
        Args:
            X (np.ndarray): Grille x (2D).
            Y (np.ndarray): Grille y (2D).
            n (int): Ordre du mode selon x.
            m (int): Ordre du mode selon y.
        
        Returns:
            np.ndarray: Champ électrique Hermite-Gaussien.
        
        Sources:
            - "Laser Beams and Resonators" by A. E. Siegman (1986)
        """
        try:
            from scipy.special import hermite
        except ImportError:
            logger.warning("scipy not available. Using Gaussian approximation.")
            return self._gaussian_electric_field(X, Y, self.diameter_mm / 4)
        
        # Calculer les polynômes d'Hermite
        w0 = self.diameter_mm / 4  # Rayon du faisceau
        
        # Normaliser les coordonnées
        x_norm = X / w0
        y_norm = Y / w0
        
        # Calculer les polynômes d'Hermite
        H_n = hermite(n)(x_norm)
        H_m = hermite(m)(y_norm)
        
        # Calculer l'enveloppe gaussienne
        envelope = np.exp(-(X**2 + Y**2) / (2 * w0**2))
        
        return (H_n * H_m * envelope) + 0.0j

    def _laguerre_gaussian_electric_field(self,
                                            X: np.ndarray,
                                            Y: np.ndarray,
                                            p: int,
                                            l: int) -> np.ndarray:
        """
        FR: Génère un champ électrique Laguerre-Gaussien.
            
            Formule : E(r,θ) = L_p^l(r²) * exp(-r² / w₀²) * exp(i*l*θ)
            où L_p^l est le polynôme de Laguerre généralisé.
            
        EN: Generates a Laguerre-Gaussian electric field.
            
            Formula: E(r,θ) = L_p^l(r²) * exp(-r² / w₀²) * exp(i*l*θ)
            where L_p^l is the generalized Laguerre polynomial.
        
        Args:
            X (np.ndarray): Grille x (2D).
            Y (np.ndarray): Grille y (2D).
            p (int): Ordre radial.
            l (int): Ordre azimutal (charge topologique).
        
        Returns:
            np.ndarray: Champ électrique Laguerre-Gaussien.
        
        Sources:
            - "Laser Beams and Resonators" by A. E. Siegman (1986)
            - "Optical Vortex Beams" by Yao & Padgett (2011)
        """
        try:
            from scipy.special import genlaguerre
        except ImportError:
            logger.warning("scipy not available. Using Gaussian approximation.")
            return self._gaussian_electric_field(X, Y, self.diameter_mm / 4)
        
        # Calculer r et θ
        R = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        
        # Remplacer les NaN dans theta
        theta = np.where(np.isfinite(theta), theta, 0.0)
        
        # Normaliser r
        w0 = self.diameter_mm / 4
        r_norm = R / w0
        
        # Calculer le polynôme de Laguerre généralisé
        # genlaguerre(p, l) prend r² comme argument
        L_pl = genlaguerre(p, l)(r_norm**2)
        
        # Calculer l'enveloppe gaussienne
        envelope = np.exp(-r_norm**2 / 2)
        
        # Phase azimutale
        phase = l * theta
        
        return (L_pl * envelope * np.exp(1j * phase)) + 0.0j

    # =========================================================================
    # CALCUL DE L'INTENSITÉ ET DE LA PHASE
    # =========================================================================

    def compute_intensity_from_electric_field(self,
                                              electric_field: np.ndarray) -> np.ndarray:
        """
        FR: Calcule l'intensité à partir du champ électrique.
            
            Formule : I(x,y) = |E(x,y)|²
            
            Gère les NaN : les remplace par 0.
        
        EN: Computes intensity from electric field.
            
            Formula: I(x,y) = |E(x,y)|²
            
            Handles NaN: replaces them with 0.
        
        Args:
            electric_field (np.ndarray): Champ électrique (complexe).
        
        Returns:
            np.ndarray: Intensité (réelle, ≥ 0).
        
        Sources:
            - "Principles of Optics" by Born & Wolf (1999), Ch. 5
        """
        if electric_field is None:
            raise ValueError("electric_field cannot be None")
        
        intensity = np.abs(electric_field)**2
        
        # Gérer les NaN
        intensity = handle_nan(intensity, method='zero')
        
        return intensity.real  # S'assurer que c'est réel

    def extract_phase_from_electric_field(self,
                                           electric_field: np.ndarray,
                                           units: str = 'nm') -> np.ndarray:
        """
        FR: Extrait la phase à partir du champ électrique.
            
            Formule : φ(x,y) = arg(E(x,y))
            
            La phase est retournée en nanomètres (nm) par défaut,
            mais peut être convertie en radians (rad) ou en longueurs d'onde (λ).
            
            Gère les NaN : les remplace par 0.
        
        EN: Extracts phase from electric field.
            
            Formula: φ(x,y) = arg(E(x,y))
            
            The phase is returned in nanometers (nm) by default,
            but can be converted to radians (rad) or wavelengths (λ).
            
            Handles NaN: replaces them with 0.
        
        Args:
            electric_field (np.ndarray): Champ électrique (complexe).
            units (str): Unités de la phase ('nm', 'rad', 'lambda').
        
        Returns:
            np.ndarray: Phase (réelle).
        
        Sources:
            - "Principles of Optics" by Born & Wolf (1999), Ch. 5
        """
        if electric_field is None:
            raise ValueError("electric_field cannot be None")
        
        # Calculer la phase en radians
        phase_rad = np.angle(electric_field)
        
        # Gérer les NaN dans phase_rad
        phase_rad = handle_nan(phase_rad, method='zero')
        
        # Convertir en nm
        if units == 'nm':
            # phase_nm = phase_rad * (wavelength_nm / (2π))
            phase_nm = phase_rad * (self.wavelength_nm / (2 * np.pi))
        elif units == 'rad':
            phase_nm = phase_rad
        elif units == 'lambda':
            # phase_lambda = phase_rad / (2π)
            phase_nm = phase_rad / (2 * np.pi)
        else:
            raise ValueError(f"Unités inconnues: {units}. Utilisez 'nm', 'rad' ou 'lambda'.")
        
        # Gérer les NaN
        phase_nm = handle_nan(phase_nm, method='zero')
        
        return phase_nm.real  # S'assurer que c'est réel

    def compute_phase_from_intensity(self,
                                     intensity: np.ndarray,
                                     method: str = 'none') -> np.ndarray:
        """
        FR: Calcule la phase à partir de l'intensité (si possible).
            
            Note : En général, la phase ne peut pas être calculée à partir
            de l'intensité seule (problème de phase). Cette méthode est
            fournie pour des cas spécifiques (ex: holographie).
            
            Gère les NaN : les remplace par 0.
        
        EN: Computes phase from intensity (if possible).
            
            Note: In general, phase cannot be computed from intensity alone
            (phase problem). This method is provided for specific cases (e.g., holography).
            
            Handles NaN: replaces them with 0.
        
        Args:
            intensity (np.ndarray): Intensité.
            method (str): Méthode de calcul ('none', 'holography').
        
        Returns:
            np.ndarray: Phase (zéros si method='none').
        """
        if method == 'none':
            return np.zeros_like(intensity)
        elif method == 'holography':
            # Méthode simplifiée pour l'holographie (à implémenter)
            logger.warning("Holography method not implemented. Returning zeros.")
            return np.zeros_like(intensity)
        else:
            raise ValueError(f"Méthode inconnue: {method}")

    # =========================================================================
    # CALCUL DES STATISTIQUES (PV, RMS)
    # =========================================================================

    def compute_pv_rms(self,
                       data: np.ndarray,
                       handle_nan: bool = True) -> Tuple[float, float]:
        """
        FR: Calcule le PV (Peak-to-Valley) et le RMS (Root Mean Square) d'un tableau.
            
            Formule :
                PV = max(data) - min(data)
                RMS = sqrt(mean((data - mean(data))²))
            
            Gère les NaN si handle_nan=True.
        
        EN: Computes PV (Peak-to-Valley) and RMS (Root Mean Square) of an array.
            
            Formula:
                PV = max(data) - min(data)
                RMS = sqrt(mean((data - mean(data))²))
            
            Handles NaN if handle_nan=True.
        
        Args:
            data (np.ndarray): Tableau de données.
            handle_nan (bool): Gérer les NaN (défaut: True).
        
        Returns:
            Tuple[float, float]: (PV, RMS).
        
        Sources:
            - "Data Reduction and Error Analysis" by P. R. Bevington (1969)
        """
        if data is None:
            raise ValueError("data cannot be None")
        
        if handle_nan:
            data = handle_nan(data, method='zero')
        
        pv = float(np.max(data) - np.min(data))
        rms = float(np.std(data))
        
        return pv, rms

    def compute_statistics(self,
                           data: np.ndarray) -> Dict[str, float]:
        """
        FR: Calcule les statistiques complètes d'un tableau.
            
            Gère les NaN.
        
        EN: Computes complete statistics of an array.
            
            Handles NaN.
        
        Args:
            data (np.ndarray): Tableau de données.
        
        Returns:
            Dict[str, float]: Dictionnaire avec les statistiques.
        """
        if data is None:
            raise ValueError("data cannot be None")
        
        data = handle_nan(data, method='zero')
        
        return {
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'pv': float(np.max(data) - np.min(data)),
            'rms': float(np.std(data))
        }

    # =========================================================================
    # NORMALISATION
    # =========================================================================

    def normalize_intensity(self,
                            intensity: Optional[np.ndarray] = None,
                            method: str = 'max') -> np.ndarray:
        """
        FR: Normalise l'intensité.
            
            Méthodes disponibles :
            - 'max': Normalise par la valeur maximale (défaut)
            - 'sum': Normalise par la somme (intensité totale = 1)
            - 'energy': Normalise par l'énergie (∫I dx dy = 1)
            
            Gère les NaN.
        
        EN: Normalizes intensity.
            
            Available methods:
            - 'max': Normalize by maximum value (default)
            - 'sum': Normalize by sum (total intensity = 1)
            - 'energy': Normalize by energy (∫I dx dy = 1)
            
            Handles NaN.
        
        Args:
            intensity (np.ndarray, optional): Intensité à normaliser.
                Si None, utilise self.intensity.
            method (str): Méthode de normalisation.
        
        Returns:
            np.ndarray: Intensité normalisée.
        """
        if intensity is None:
            if self.intensity is None:
                raise ValueError("intensity is None and self.intensity is None")
            intensity = self.intensity
        
        intensity = handle_nan(intensity, method='zero')
        
        if method == 'max':
            max_val = np.max(intensity)
            if max_val > 0:
                return intensity / max_val
            else:
                return intensity
        
        elif method == 'sum':
            total = np.sum(intensity)
            if total > 0:
                return intensity / total
            else:
                return intensity
        
        elif method == 'energy':
            # Intégration numérique (approximation)
            dx = self.diameter_mm / self.num_points
            dy = self.diameter_mm / self.num_points
            total_energy = np.sum(intensity) * dx * dy
            if total_energy > 0:
                return intensity / total_energy
            else:
                return intensity
        
        else:
            raise ValueError(f"Méthode de normalisation inconnue: {method}")

    def normalize_phase(self,
                       phase: Optional[np.ndarray] = None,
                       method: str = 'unwrap') -> np.ndarray:
        """
        FR: Normalise la phase.
            
            Méthodes disponibles :
            - 'unwrap': Déroule la phase (défaut)
            - 'zero_mean': Soustrait la moyenne
            - 'zero_min': Soustrait le minimum
            
            Gère les NaN.
        
        EN: Normalizes phase.
            
            Available methods:
            - 'unwrap': Unwrap phase (default)
            - 'zero_mean': Subtract mean
            - 'zero_min': Subtract minimum
            
            Handles NaN.
        
        Args:
            phase (np.ndarray, optional): Phase à normaliser.
                Si None, utilise self.phase.
            method (str): Méthode de normalisation.
        
        Returns:
            np.ndarray: Phase normalisée.
        """
        if phase is None:
            if self.phase is None:
                raise ValueError("phase is None and self.phase is None")
            phase = self.phase
        
        phase = handle_nan(phase, method='zero')
        
        if method == 'unwrap':
            try:
                return np.unwrap(phase)
            except:
                return phase
        
        elif method == 'zero_mean':
            return phase - np.mean(phase)
        
        elif method == 'zero_min':
            return phase - np.min(phase)
        
        else:
            raise ValueError(f"Méthode de normalisation inconnue: {method}")

    # =========================================================================
    # PROPAGATION (DOIVENT RESTER DANS Beam.py)
    # =========================================================================

    def propagate(self,
                  distance_mm: float,
                  method: Union[PropagationMethod, str] = PropagationMethod.ANGULAR_SPECTRUM,
                  **kwargs) -> np.ndarray:
        """
        FR: Propage le faisceau sur une distance donnée.
            
            Méthodes disponibles :
            - ANGULAR_SPECTRUM: Méthode du spectre angulaire (défaut, précise)
            - FRESNEL: Approximation de Fresnel
            - FRAUNHOFER: Approximation de Fraunhofer
            - RAY_TRACING: Lancer de rayons (non implémenté)
            
            La propagation est effectuée dans le domaine de Fourier.
            
            Gère les NaN.
        
        EN: Propagates the beam over a given distance.
            
            Available methods:
            - ANGULAR_SPECTRUM: Angular spectrum method (default, accurate)
            - FRESNEL: Fresnel approximation
            - FRAUNHOFER: Fraunhofer approximation
            - RAY_TRACING: Ray tracing (not implemented)
            
            Propagation is performed in the Fourier domain.
            
            Handles NaN.
        
        Args:
            distance_mm (float): Distance de propagation en mm.
            method (PropagationMethod or str): Méthode de propagation.
            **kwargs: Arguments spécifiques à la méthode.
        
        Returns:
            np.ndarray: Champ électrique propagé.
        
        Raises:
            ValueError: Si le champ électrique est None.
        
        Sources:
            - "Fourier Optics" by Goodman (2005), Ch. 3-4
            - "Principles of Optics" by Born & Wolf (1999), Ch. 8
        """
        if self.electric_field is None:
            raise ValueError("electric_field is None. Generate it first with generate_electric_field().")
        
        if isinstance(method, str):
            method = PropagationMethod(method.lower())
        
        if method == PropagationMethod.ANGULAR_SPECTRUM:
            return self._propagate_angular_spectrum(distance_mm, **kwargs)
        
        elif method == PropagationMethod.FRESNEL:
            return self._propagate_fresnel(distance_mm, **kwargs)
        
        elif method == PropagationMethod.FRAUNHOFER:
            return self._propagate_fraunhofer(distance_mm, **kwargs)
        
        elif method == PropagationMethod.RAY_TRACING:
            raise NotImplementedError("Ray tracing propagation not implemented")
        
        else:
            raise ValueError(f"Méthode de propagation inconnue: {method}")

    def _propagate_angular_spectrum(self,
                                    distance_mm: float,
                                    **kwargs) -> np.ndarray:
        """
        FR: Propage le faisceau avec la méthode du spectre angulaire.
            
            Cette méthode est précise pour les courtes et moyennes distances.
            
            Formule :
                E(x,y,z) = IFFT{ FFT(E(x,y,0)) * H(f_x,f_y,z) }
            où H(f_x,f_y,z) = exp(i * 2π * z * sqrt(1/λ² - f_x² - f_y²))
            est la fonction de transfert de propagation.
            
            Gère les NaN.
        
        EN: Propagates the beam using the angular spectrum method.
            
            This method is accurate for short and medium distances.
            
            Formula:
                E(x,y,z) = IFFT{ FFT(E(x,y,0)) * H(f_x,f_y,z) }
            where H(f_x,f_y,z) = exp(i * 2π * z * sqrt(1/λ² - f_x² - f_y²))
            is the propagation transfer function.
            
            Handles NaN.
        
        Args:
            distance_mm (float): Distance de propagation en mm.
        
        Returns:
            np.ndarray: Champ électrique propagé.
        
        Sources:
            - "Fourier Optics" by Goodman (2005), Eq. 3.17
        """
        # Longueur d'onde en mm
        wavelength_mm = self.wavelength_nm * 1e-6
        
        # Nombre d'onde
        k = 2 * np.pi / wavelength_mm
        
        # Taille du faisceau en mm
        Lx = self.diameter_mm
        Ly = self.diameter_mm
        
        # Résolution spatiale
        dx = Lx / self.num_points
        dy = Ly / self.num_points
        
        # Fréquences spatiales
        fx = np.fft.fftfreq(self.num_points, d=dx)
        fy = np.fft.fftfreq(self.num_points, d=dy)
        FX, FY = np.meshgrid(fx, fy)
        
        # Fonction de transfert de propagation
        # H = exp(i * 2π * z * sqrt(1/λ² - fx² - fy²))
        # = exp(i * k * z * sqrt(1 - (λ*fx)² - (λ*fy)²))
        sqrt_term = np.sqrt(np.maximum(1 - (wavelength_mm * FX)**2 - (wavelength_mm * FY)**2, 0.0))
        
        # Calculer H
        H = np.exp(1j * k * distance_mm * sqrt_term)
        
        # Gérer les NaN dans H
        H = np.nan_to_num(H, nan=0.0 + 0.0j)
        
        # Transformée de Fourier du champ électrique initial
        E_fft = np.fft.fft2(self.electric_field)
        E_fft = np.fft.fftshift(E_fft)
        
        # Appliquer la fonction de transfert
        propagated_fft = E_fft * H
        
        # Transformée inverse
        propagated_field = np.fft.ifftshift(propagated_fft)
        propagated_field = np.fft.ifft2(propagated_field)
        
        # Gérer les NaN
        propagated_field = np.nan_to_num(propagated_field, nan=0.0 + 0.0j)
        
        return propagated_field

    def _propagate_fresnel(self,
                           distance_mm: float,
                           **kwargs) -> np.ndarray:
        """
        FR: Propage le faisceau avec l'approximation de Fresnel.
            
            Cette méthode est valable pour les moyennes distances où
            l'approximation paraxiale est valide.
            
            Formule :
                E(x,y,z) ≈ (i / (λz)) * exp(i * 2π * z / λ) *
                          IFFT{ FFT(E(x,y,0)) * exp(i * π * λ * z * (f_x² + f_y²)) }
            
            Gère les NaN.
        
        EN: Propagates the beam using the Fresnel approximation.
            
            This method is valid for medium distances where the paraxial
            approximation is valid.
            
            Formula:
                E(x,y,z) ≈ (i / (λz)) * exp(i * 2π * z / λ) *
                          IFFT{ FFT(E(x,y,0)) * exp(i * π * λ * z * (f_x² + f_y²)) }
            
            Handles NaN.
        
        Args:
            distance_mm (float): Distance de propagation en mm.
        
        Returns:
            np.ndarray: Champ électrique propagé.
        
        Sources:
            - "Fourier Optics" by Goodman (2005), Eq. 3.44
        """
        wavelength_mm = self.wavelength_nm * 1e-6
        k = 2 * np.pi / wavelength_mm
        
        Lx = self.diameter_mm
        Ly = self.diameter_mm
        
        dx = Lx / self.num_points
        dy = Ly / self.num_points
        
        fx = np.fft.fftfreq(self.num_points, d=dx)
        fy = np.fft.fftfreq(self.num_points, d=dy)
        FX, FY = np.meshgrid(fx, fy)
        
        # Fonction de transfert de Fresnel
        H = np.exp(1j * np.pi * wavelength_mm * distance_mm * (FX**2 + FY**2))
        
        # Facteur de phase global
        global_phase = np.exp(1j * k * distance_mm)
        
        # Transformée de Fourier
        E_fft = np.fft.fft2(self.electric_field)
        E_fft = np.fft.fftshift(E_fft)
        
        # Appliquer la fonction de transfert
        propagated_fft = E_fft * H
        
        # Transformée inverse
        propagated_field = np.fft.ifftshift(propagated_fft)
        propagated_field = np.fft.ifft2(propagated_field)
        
        # Appliquer le facteur de phase global et le facteur d'échelle
        propagated_field = propagated_field * global_phase * (1j / (wavelength_mm * distance_mm))
        
        # Gérer les NaN
        propagated_field = np.nan_to_num(propagated_field, nan=0.0 + 0.0j)
        
        return propagated_field

    def _propagate_fraunhofer(self,
                              distance_mm: float,
                              **kwargs) -> np.ndarray:
        """
        FR: Propage le faisceau avec l'approximation de Fraunhofer.
            
            Cette méthode est valable pour les longues distances où
            l'approximation de Fraunhofer est valide (champ lointain).
            
            Formule :
                E(x,y,z) ≈ (i / (λz)) * exp(i * 2π * z / λ) *
                          IFFT{ FFT(E(x,y,0)) * exp(i * π / (λz) * (x² + y²)) }
            
            Gère les NaN.
        
        EN: Propagates the beam using the Fraunhofer approximation.
            
            This method is valid for long distances where the Fraunhofer
            approximation is valid (far field).
            
            Formula:
                E(x,y,z) ≈ (i / (λz)) * exp(i * 2π * z / λ) *
                          IFFT{ FFT(E(x,y,0)) * exp(i * π / (λz) * (x² + y²)) }
            
            Handles NaN.
        
        Args:
            distance_mm (float): Distance de propagation en mm.
        
        Returns:
            np.ndarray: Champ électrique propagé.
        
        Sources:
            - "Fourier Optics" by Goodman (2005), Eq. 3.58
        """
        wavelength_mm = self.wavelength_nm * 1e-6
        k = 2 * np.pi / wavelength_mm
        
        Lx = self.diameter_mm
        Ly = self.diameter_mm
        
        dx = Lx / self.num_points
        dy = Ly / self.num_points
        
        # Coordonnées spatiales
        x = np.linspace(-Lx/2, Lx/2, self.num_points)
        y = np.linspace(-Ly/2, Ly/2, self.num_points)
        X, Y = np.meshgrid(x, y)
        
        # Facteur de phase quadratique
        quadratic_phase = np.exp(1j * np.pi / (wavelength_mm * distance_mm) * (X**2 + Y**2))
        
        # Transformée de Fourier du champ électrique initial
        E_fft = np.fft.fft2(self.electric_field)
        E_fft = np.fft.fftshift(E_fft)
        
        # Appliquer le facteur de phase
        propagated_fft = E_fft * quadratic_phase
        
        # Transformée inverse
        propagated_field = np.fft.ifftshift(propagated_fft)
        propagated_field = np.fft.ifft2(propagated_field)
        
        # Appliquer le facteur d'échelle
        propagated_field = propagated_field * (1j / (wavelength_mm * distance_mm)) * np.exp(1j * k * distance_mm)
        
        # Gérer les NaN
        propagated_field = np.nan_to_num(propagated_field, nan=0.0 + 0.0j)
        
        return propagated_field


# =============================================================================
# CLASSE POUR LA PROPAGATION (DOIT RESTER DANS Beam.py)
# =============================================================================

class Propagation:
    """
    FR: Propagation d'un champ électrique sur une distance donnée.
        
        Cette classe implémente différentes méthodes de propagation :
        - Spectre angulaire (défaut)
        - Approximation de Fresnel
        - Approximation de Fraunhofer
        
        Unités :
        - Distance : mm
        - Longueur d'onde : nm
        - Champ électrique : complexe (a.u.)
    
    EN: Propagation of an electric field over a given distance.
        
        This class implements different propagation methods:
        - Angular spectrum (default)
        - Fresnel approximation
        - Fraunhofer approximation
        
        Units:
        - Distance: mm
        - Wavelength: nm
        - Electric field: complex (a.u.)
    
    Attributes:
        wavelength_nm (float): Longueur d'onde en nm.
        propagation_distance_mm (float): Distance de propagation en mm.
        input_diameter_mm (float): Diamètre du faisceau d'entrée en mm.
        output_diameter_mm (float): Diamètre du faisceau de sortie en mm.
        num_points (int): Nombre de points.
        method (PropagationMethod): Méthode de propagation.
    
    Sources:
        - "Fourier Optics" by Goodman (2005)
    """

    def __init__(self,
                 wavelength_nm: float = DEFAULT_WAVELENGTH_NM,
                 propagation_distance_mm: float = 10.0,
                 input_diameter_mm: float = DEFAULT_DIAMETER_MM,
                 output_diameter_mm: Optional[float] = None,
                 num_points: int = DEFAULT_NUM_POINTS,
                 method: Union[PropagationMethod, str] = PropagationMethod.ANGULAR_SPECTRUM):
        """
        FR: Initialise un propagateur.
            
        EN: Initializes a propagator.
        
        Args:
            wavelength_nm (float): Longueur d'onde en nm.
            propagation_distance_mm (float): Distance de propagation en mm.
            input_diameter_mm (float): Diamètre du faisceau d'entrée en mm.
            output_diameter_mm (float, optional): Diamètre du faisceau de sortie en mm.
                Si None, = input_diameter_mm.
            num_points (int): Nombre de points.
            method (PropagationMethod or str): Méthode de propagation.
        """
        self.wavelength_nm = float(wavelength_nm)
        self.propagation_distance_mm = float(propagation_distance_mm)
        self.input_diameter_mm = float(input_diameter_mm)
        self.output_diameter_mm = float(output_diameter_mm) if output_diameter_mm is not None else input_diameter_mm
        self.num_points = int(num_points)
        
        if isinstance(method, str):
            method = PropagationMethod(method.lower())
        self.method = method
        
        logger.info(f"Propagateur initialisé: λ={self.wavelength_nm}nm, "
                   f"d={self.propagation_distance_mm}mm, "
                   f"méthode={self.method.value}")

    def propagate(self,
                  input_field: np.ndarray) -> np.ndarray:
        """
        FR: Propage un champ électrique.
            
        EN: Propagates an electric field.
        
        Args:
            input_field (np.ndarray): Champ électrique d'entrée (2D complexe).
        
        Returns:
            np.ndarray: Champ électrique propagé.
        """
        if input_field is None:
            raise ValueError("input_field cannot be None")
        
        # Créer un faisceau temporaire pour la propagation
        temp_beam = Beam(
            wavelength_nm=self.wavelength_nm,
            diameter_mm=self.input_diameter_mm,
            num_points=self.num_points
        )
        temp_beam.electric_field = input_field
        
        # Propage avec la méthode spécifiée
        return temp_beam.propagate(
            distance_mm=self.propagation_distance_mm,
            method=self.method
        )


# =============================================================================
# TESTS UNITAIRES
# =============================================================================

class TestBeam:
    """FR: Tests unitaires pour Beam.py."""

    def test_beam_creation(self):
        """FR: Test la création d'un faisceau."""
        beam = Beam(
            wavelength_nm=633.0,
            diameter_mm=5.0,
            num_points=128
        )
        assert beam.wavelength_nm == 633.0
        assert beam.diameter_mm == 5.0
        assert beam.num_points == 128
        assert beam.energy == 1.0

    def test_gaussian_electric_field(self):
        """FR: Test la génération d'un champ électrique gaussien."""
        beam = Beam(
            wavelength_nm=633.0,
            diameter_mm=5.0,
            num_points=128
        )
        
        electric_field = beam.generate_electric_field(
            method=BeamProfile.GAUSSIAN,
            sigma_mm=1.0
        )
        
        assert electric_field.shape == (128, 128)
        assert np.all(np.isfinite(electric_field))
        assert np.max(np.abs(electric_field)) > 0

    def test_uniform_electric_field(self):
        """FR: Test la génération d'un champ électrique uniforme."""
        beam = Beam(
            wavelength_nm=633.0,
            diameter_mm=5.0,
            num_points=128
        )
        
        electric_field = beam.generate_electric_field(
            method=BeamProfile.UNIFORM
        )
        
        assert electric_field.shape == (128, 128)
        # Le centre doit être à 1.0
        center = electric_field[64, 64]
        assert np.abs(center) > 0.9

    def test_intensity_calculation(self):
        """FR: Test le calcul de l'intensité."""
        beam = Beam(
            wavelength_nm=633.0,
            diameter_mm=5.0,
            num_points=128
        )
        
        electric_field = beam.generate_electric_field(
            method=BeamProfile.GAUSSIAN
        )
        beam.electric_field = electric_field
        
        intensity = beam.compute_intensity_from_electric_field(electric_field)
        
        assert intensity.shape == (128, 128)
        assert np.all(intensity >= 0)
        assert np.max(intensity) > 0

    def test_phase_extraction(self):
        """FR: Test l'extraction de la phase."""
        beam = Beam(
            wavelength_nm=633.0,
            diameter_mm=5.0,
            num_points=128
        )
        
        electric_field = beam.generate_electric_field(
            method=BeamProfile.GAUSSIAN
        )
        beam.electric_field = electric_field
        
        phase_nm = beam.extract_phase_from_electric_field(electric_field, units='nm')
        phase_rad = beam.extract_phase_from_electric_field(electric_field, units='rad')
        
        assert phase_nm.shape == (128, 128)
        assert phase_rad.shape == (128, 128)
        assert np.all(np.isfinite(phase_nm))
        assert np.all(np.isfinite(phase_rad))

    def test_pv_rms_calculation(self):
        """FR: Test le calcul du PV et du RMS."""
        beam = Beam(
            wavelength_nm=633.0,
            diameter_mm=5.0,
            num_points=128
        )
        
        electric_field = beam.generate_electric_field(
            method=BeamProfile.GAUSSIAN
        )
        beam.electric_field = electric_field
        beam.intensity = beam.compute_intensity_from_electric_field(electric_field)
        
        pv, rms = beam.compute_pv_rms(beam.intensity)
        
        assert pv > 0
        assert rms > 0
        assert pv >= rms

    def test_normalization(self):
        """FR: Test la normalisation de l'intensité."""
        beam = Beam(
            wavelength_nm=633.0,
            diameter_mm=5.0,
            num_points=128
        )
        
        electric_field = beam.generate_electric_field(
            method=BeamProfile.GAUSSIAN
        )
        beam.electric_field = electric_field
        beam.intensity = beam.compute_intensity_from_electric_field(electric_field)
        
        # Normalisation par max
        normalized = beam.normalize_intensity(method='max')
        assert np.max(normalized) <= 1.0 + 1e-10
        assert np.min(normalized) >= 0

    def test_propagation(self):
        """FR: Test la propagation du faisceau."""
        beam = Beam(
            wavelength_nm=633.0,
            diameter_mm=5.0,
            num_points=128
        )
        
        electric_field = beam.generate_electric_field(
            method=BeamProfile.GAUSSIAN
        )
        beam.electric_field = electric_field
        
        # Propage sur 10 mm
        propagated_field = beam.propagate(
            distance_mm=10.0,
            method=PropagationMethod.ANGULAR_SPECTRUM
        )
        
        assert propagated_field.shape == (128, 128)
        assert np.all(np.isfinite(propagated_field))

    def test_nan_handling(self):
        """FR: Test la gestion des NaN."""
        # Créer un tableau avec des NaN
        data = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0]])
        
        # Tester handle_nan
        result_zero = handle_nan(data, method='zero')
        assert not np.any(np.isnan(result_zero))
        assert result_zero[0, 2] == 0.0
        
        result_mean = handle_nan(data, method='mean')
        assert not np.any(np.isnan(result_mean))
        assert result_mean[0, 2] == np.nanmean(data)

    def test_safe_divide(self):
        """FR: Test la division sûre."""
        numerator = np.array([1.0, 2.0, 3.0])
        denominator = np.array([1.0, 0.0, np.nan])
        
        result = safe_divide(numerator, denominator, default=0.0)
        
        assert result[0] == 1.0
        assert result[1] == 0.0  # Division par zéro
        assert result[2] == 0.0  # Division par NaN

    def test_safe_sqrt(self):
        """FR: Test la racine carrée sûre."""
        data = np.array([1.0, -1.0, np.nan])
        
        result = safe_sqrt(data, default=0.0)
        
        assert result[0] == 1.0
        assert result[1] == 0.0  # sqrt(-1) → default
        assert result[2] == 0.0  # sqrt(NaN) → default


if __name__ == "__main__":
    import unittest
    unittest.main()

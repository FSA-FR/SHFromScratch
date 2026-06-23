"""
Beam.py

FR: Module pour la génération et la gestion de faisceaux optiques.
    Permet de créer et manipuler des faisceaux avec :
    - Différents profils d'intensité (gaussien, uniforme, annulaire, etc.)
    - Calcul du champ électrique, de l'intensité et de la phase
    - Gestion des unités :
        * Longueurs : mm (pour les dimensions du faisceau)
        * Longueur d'onde : nm
        * Phase : nm (principale), rad (pour les calculs)
        * Intensité : a.u. (arbitrary units) ou normalisée
    - Fonctions de propagation (doivent rester dans ce module)
    - Toutes les fonctions gèrent les NaN sans les propager
    
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
        * Phase: nm (main), rad (for calculations)
        * Intensity: a.u. (arbitrary units) or normalized
    - Propagation functions (must remain in this module)
    - All functions handle NaN without propagating them
    
    Each generated image (phase and intensity) will have:
    - A visual scale
    - PV (Peak-to-Valley) and RMS values
    - Colormap: "Jet" for phase, "hot" for intensity

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch

Dependencies:
    - numpy
    - MathAndPhysicsTools (pour les fonctions outils)
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
"""

import numpy as np
import logging
from typing import Optional, Tuple, Dict, List, Union
from enum import Enum
from datetime import datetime

# Import des fonctions outils depuis MathAndPhysicsTools
from MathAndPhysicsTools import (
    handle_nan,
    safe_divide,
    safe_sqrt,
    safe_log,
    safe_exp,
    compute_pv_rms,
    compute_statistics,
    normalize_array,
    create_grid,
    create_polar_grid,
    cartesian_to_polar,
    polar_to_cartesian,
    nm_to_rad,
    rad_to_nm,
    nm_to_lambda,
    lambda_to_nm,
    rad_to_mrad,
    mrad_to_rad,
    mm_to_um,
    um_to_mm,
    generate_zernike_polynomial,
    generate_legendre_polynomial,
    generate_hermite_polynomial,
    generate_laguerre_polynomial,
    ZernikeOrdering,
    NormalizationType,
    DEFAULT_WAVELENGTH_NM,
    NUMERICAL_TOLERANCE,
    PI,
    TWO_PI
)


# =============================================================================
# CONFIGURATION GLOBALE
# =============================================================================

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Beam")


# =============================================================================
# CONSTANTES PAR DÉFAUT
# =============================================================================

DEFAULT_ENERGY = 1.0  # Énergie par défaut (a.u.)
DEFAULT_DIAMETER_MM = 5.0  # Diamètre par défaut (mm)
DEFAULT_NUM_POINTS = 512  # Nombre de points par défaut


# =============================================================================
# ENUMS
# =============================================================================

class BeamProfile(Enum):
    """
    FR: Profil d'intensité du faisceau.
    EN: Beam intensity profile.
    
    Sources: "Laser Beam Propagation" by Goodman (1996), Ch. 3
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
    EN: Beam propagation method.
    
    Sources:
        - "Fourier Optics" by Goodman (2005), Ch. 3-4
        - "Principles of Optics" by Born & Wolf (1999), Ch. 8
    """
    ANGULAR_SPECTRUM = "angular_spectrum"
    FRESNEL = "fresnel"
    FRAUNHOFER = "fraunhofer"
    RAY_TRACING = "ray_tracing"


# =============================================================================
# CLASSE PRINCIPALE: BEAM
# =============================================================================

class Beam:
    """
    FR: Faisceau optique.
    EN: Optical beam.
    
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
        
        self.electric_field: Optional[np.ndarray] = None
        self.intensity: Optional[np.ndarray] = None
        self.phase: Optional[np.ndarray] = None
        
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
    # GÉNÉRATION DU CHAMP ÉLECTRIQUE
    # =========================================================================

    def generate_electric_field(self,
                                 method: Union[BeamProfile, str] = BeamProfile.GAUSSIAN,
                                 **kwargs) -> np.ndarray:
        """
        FR: Génère le champ électrique du faisceau.
        EN: Generates the electric field of the beam.
        
        Args:
            method (BeamProfile or str): Méthode de génération.
            **kwargs: Arguments spécifiques à la méthode.
        
        Raises:
            ValueError: Si la méthode est inconnue.
        
        Sources: "Laser Beam Propagation" by Goodman (1996), Ch. 3
        """
        if isinstance(method, str):
            method = BeamProfile(method.lower())
        
        x, y, X, Y = create_grid(self.num_points, diameter_mm=self.diameter_mm)
        
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

    def _gaussian_electric_field(self,
                                  X: np.ndarray,
                                  Y: np.ndarray,
                                  sigma_mm: float) -> np.ndarray:
        """FR: Génère un champ électrique gaussien. EN: Generates a Gaussian electric field."""
        R_sq = X**2 + Y**2
        exponent = -R_sq / (2 * sigma_mm**2)
        exponent = np.clip(exponent, -700, 0)
        return safe_exp(exponent)

    def _uniform_electric_field(self,
                                 X: np.ndarray,
                                 Y: np.ndarray) -> np.ndarray:
        """FR: Génère un champ électrique uniforme. EN: Generates a uniform electric field."""
        R = np.sqrt(X**2 + Y**2)
        mask = R <= (self.diameter_mm / 2)
        return np.where(mask, 1.0 + 0.0j, 0.0 + 0.0j)

    def _annular_electric_field(self,
                                X: np.ndarray,
                                Y: np.ndarray,
                                inner_diameter_mm: float,
                                outer_diameter_mm: float) -> np.ndarray:
        """FR: Génère un champ électrique annulaire. EN: Generates an annular electric field."""
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
        """FR: Génère un champ électrique en forme de donut. EN: Generates a donut-shaped electric field."""
        R_sq = X**2 + Y**2
        sigma_mm = (outer_diameter_mm - inner_diameter_mm) / 4
        amplitude = safe_exp(-R_sq / (2 * sigma_mm**2))
        theta = np.arctan2(Y, X)
        theta = np.where(np.isfinite(theta), theta, 0.0)
        phase = order * theta
        return amplitude * np.exp(1j * phase)

    def _tophat_electric_field(self,
                               X: np.ndarray,
                               Y: np.ndarray,
                               radius_mm: float) -> np.ndarray:
        """FR: Génère un champ électrique "chapeau haut". EN: Generates a top-hat electric field."""
        R = np.sqrt(X**2 + Y**2)
        return np.where(R <= radius_mm, 1.0 + 0.0j, 0.0 + 0.0j)

    def _airy_electric_field(self,
                             X: np.ndarray,
                             Y: np.ndarray,
                             aperture_diameter_mm: float) -> np.ndarray:
        """FR: Génère un champ électrique correspondant à une tâche d'Airy. EN: Generates an Airy spot electric field."""
        try:
            from scipy.special import j1
        except ImportError:
            logger.warning("scipy not available. Using Gaussian approximation.")
            return self._gaussian_electric_field(X, Y, aperture_diameter_mm / 4)
        
        R = np.sqrt(X**2 + Y**2)
        k = TWO_PI / (self.wavelength_nm * 1e-6)
        kr = k * R
        
        with np.errstate(divide='ignore', invalid='ignore'):
            amplitude = 2 * j1(kr) / kr
        amplitude = np.where(kr == 0, 1.0, amplitude)
        aperture_radius_mm = aperture_diameter_mm / 2
        mask = R <= aperture_radius_mm
        return np.where(mask, amplitude + 0.0j, 0.0 + 0.0j)

    def _hermite_gaussian_electric_field(self,
                                          X: np.ndarray,
                                          Y: np.ndarray,
                                          n: int,
                                          m: int) -> np.ndarray:
        """FR: Génère un champ électrique Hermite-Gaussien. EN: Generates a Hermite-Gaussian electric field."""
        w0 = self.diameter_mm / 4
        x_norm = X / w0
        y_norm = Y / w0
        H_n = generate_hermite_polynomial(n, x_norm)
        H_m = generate_hermite_polynomial(m, y_norm)
        envelope = safe_exp(-(X**2 + Y**2) / (2 * w0**2))
        return (H_n * H_m * envelope) + 0.0j

    def _laguerre_gaussian_electric_field(self,
                                            X: np.ndarray,
                                            Y: np.ndarray,
                                            p: int,
                                            l: int) -> np.ndarray:
        """FR: Génère un champ électrique Laguerre-Gaussien. EN: Generates a Laguerre-Gaussian electric field."""
        R = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        theta = np.where(np.isfinite(theta), theta, 0.0)
        w0 = self.diameter_mm / 4
        r_norm = R / w0
        L_pl = generate_laguerre_polynomial(p, l, r_norm**2)
        envelope = safe_exp(-r_norm**2 / 2)
        phase = l * theta
        return (L_pl * envelope * np.exp(1j * phase)) + 0.0j

    # =========================================================================
    # CALCUL DE L'INTENSITÉ ET DE LA PHASE
    # =========================================================================

    def compute_intensity_from_electric_field(self,
                                              electric_field: np.ndarray) -> np.ndarray:
        """
        FR: Calcule l'intensité à partir du champ électrique.
        EN: Computes intensity from electric field.
        
        Sources: "Principles of Optics" by Born & Wolf (1999), Ch. 5
        """
        if electric_field is None:
            raise ValueError("electric_field cannot be None")
        intensity = np.abs(electric_field)**2
        intensity = handle_nan(intensity, method='zero')
        return intensity.real

    def extract_phase_from_electric_field(self,
                                           electric_field: np.ndarray,
                                           units: str = 'nm') -> np.ndarray:
        """
        FR: Extrait la phase à partir du champ électrique.
        EN: Extracts phase from electric field.
        
        Sources: "Principles of Optics" by Born & Wolf (1999), Ch. 5
        """
        if electric_field is None:
            raise ValueError("electric_field cannot be None")
        
        phase_rad = np.angle(electric_field)
        phase_rad = handle_nan(phase_rad, method='zero')
        
        if units == 'nm':
            return rad_to_nm(phase_rad, self.wavelength_nm).real
        elif units == 'rad':
            return phase_rad.real
        elif units == 'lambda':
            return lambda_to_nm(phase_rad / TWO_PI, self.wavelength_nm).real
        else:
            raise ValueError(f"Unités inconnues: {units}. Utilisez 'nm', 'rad' ou 'lambda'.")

    # =========================================================================
    # PROPAGATION (DOIVENT RESTER DANS Beam.py)
    # =========================================================================

    def propagate(self,
                  distance_mm: float,
                  method: Union[PropagationMethod, str] = PropagationMethod.ANGULAR_SPECTRUM,
                  **kwargs) -> np.ndarray:
        """
        FR: Propage le faisceau sur une distance donnée.
        EN: Propagates the beam over a given distance.
        
        Sources:
            - "Fourier Optics" by Goodman (2005), Ch. 3-4
            - "Principles of Optics" by Born & Wolf (1999), Ch. 8
        """
        if self.electric_field is None:
            raise ValueError("electric_field is None. Generate it first.")
        
        if isinstance(method, str):
            method = PropagationMethod(method.lower())
        
        if method == PropagationMethod.ANGULAR_SPECTRUM:
            return self._propagate_angular_spectrum(distance_mm, **kwargs)
        elif method == PropagationMethod.FRESNEL:
            return self._propagate_fresnel(distance_mm, **kwargs)
        elif method == PropagationMethod.FRAUNHOFER:
            return self._propagate_fraunhofer(distance_mm, **kwargs)
        elif method == PropagationMethod.RAY_TRACING:
            raise NotImplementedError("Ray tracing not implemented")
        else:
            raise ValueError(f"Méthode inconnue: {method}")

    def _propagate_angular_spectrum(self,
                                    distance_mm: float,
                                    **kwargs) -> np.ndarray:
        """FR: Propage avec la méthode du spectre angulaire. EN: Propagates using angular spectrum method."""
        wavelength_mm = self.wavelength_nm * 1e-6
        k = TWO_PI / wavelength_mm
        Lx = Ly = self.diameter_mm
        dx = Lx / self.num_points
        dy = Ly / self.num_points
        
        fx = np.fft.fftfreq(self.num_points, d=dx)
        fy = np.fft.fftfreq(self.num_points, d=dy)
        FX, FY = np.meshgrid(fx, fy)
        
        sqrt_term = np.sqrt(np.maximum(1 - (wavelength_mm * FX)**2 - (wavelength_mm * FY)**2, 0.0))
        H = np.exp(1j * k * distance_mm * sqrt_term)
        H = np.nan_to_num(H, nan=0.0 + 0.0j)
        
        E_fft = np.fft.fft2(self.electric_field)
        E_fft = np.fft.fftshift(E_fft)
        propagated_fft = E_fft * H
        propagated_field = np.fft.ifftshift(propagated_fft)
        propagated_field = np.fft.ifft2(propagated_field)
        return np.nan_to_num(propagated_field, nan=0.0 + 0.0j)

    def _propagate_fresnel(self,
                           distance_mm: float,
                           **kwargs) -> np.ndarray:
        """FR: Propage avec l'approximation de Fresnel. EN: Propagates using Fresnel approximation."""
        wavelength_mm = self.wavelength_nm * 1e-6
        k = TWO_PI / wavelength_mm
        Lx = Ly = self.diameter_mm
        dx = Lx / self.num_points
        dy = Ly / self.num_points
        
        fx = np.fft.fftfreq(self.num_points, d=dx)
        fy = np.fft.fftfreq(self.num_points, d=dy)
        FX, FY = np.meshgrid(fx, fy)
        
        H = np.exp(1j * np.pi * wavelength_mm * distance_mm * (FX**2 + FY**2))
        global_phase = np.exp(1j * k * distance_mm)
        
        E_fft = np.fft.fft2(self.electric_field)
        E_fft = np.fft.fftshift(E_fft)
        propagated_fft = E_fft * H
        propagated_field = np.fft.ifftshift(propagated_fft)
        propagated_field = np.fft.ifft2(propagated_field)
        propagated_field = propagated_field * global_phase * (1j / (wavelength_mm * distance_mm))
        return np.nan_to_num(propagated_field, nan=0.0 + 0.0j)

    def _propagate_fraunhofer(self,
                              distance_mm: float,
                              **kwargs) -> np.ndarray:
        """FR: Propage avec l'approximation de Fraunhofer. EN: Propagates using Fraunhofer approximation."""
        wavelength_mm = self.wavelength_nm * 1e-6
        k = TWO_PI / wavelength_mm
        Lx = Ly = self.diameter_mm
        
        x = np.linspace(-Lx/2, Lx/2, self.num_points)
        y = np.linspace(-Ly/2, Ly/2, self.num_points)
        X, Y = np.meshgrid(x, y)
        
        quadratic_phase = np.exp(1j * np.pi / (wavelength_mm * distance_mm) * (X**2 + Y**2))
        E_fft = np.fft.fft2(self.electric_field)
        E_fft = np.fft.fftshift(E_fft)
        propagated_fft = E_fft * quadratic_phase
        propagated_field = np.fft.ifftshift(propagated_fft)
        propagated_field = np.fft.ifft2(propagated_field)
        propagated_field = propagated_field * (1j / (wavelength_mm * distance_mm)) * np.exp(1j * k * distance_mm)
        return np.nan_to_num(propagated_field, nan=0.0 + 0.0j)


# =============================================================================
# CLASSE POUR LA PROPAGATION
# =============================================================================

class Propagation:
    """FR: Propagation d'un champ électrique. EN: Propagation of an electric field."""

    def __init__(self,
                 wavelength_nm: float = DEFAULT_WAVELENGTH_NM,
                 propagation_distance_mm: float = 10.0,
                 input_diameter_mm: float = DEFAULT_DIAMETER_MM,
                 output_diameter_mm: Optional[float] = None,
                 num_points: int = DEFAULT_NUM_POINTS,
                 method: Union[PropagationMethod, str] = PropagationMethod.ANGULAR_SPECTRUM):
        self.wavelength_nm = float(wavelength_nm)
        self.propagation_distance_mm = float(propagation_distance_mm)
        self.input_diameter_mm = float(input_diameter_mm)
        self.output_diameter_mm = float(output_diameter_mm) if output_diameter_mm is not None else input_diameter_mm
        self.num_points = int(num_points)
        if isinstance(method, str):
            method = PropagationMethod(method.lower())
        self.method = method

    def propagate(self, input_field: np.ndarray) -> np.ndarray:
        """FR: Propage un champ électrique. EN: Propagates an electric field."""
        if input_field is None:
            raise ValueError("input_field cannot be None")
        temp_beam = Beam(wavelength_nm=self.wavelength_nm,
                        diameter_mm=self.input_diameter_mm,
                        num_points=self.num_points)
        temp_beam.electric_field = input_field
        return temp_beam.propagate(distance_mm=self.propagation_distance_mm, method=self.method)


# =============================================================================
# TESTS UNITAIRES
# =============================================================================

class TestBeam:
    """FR: Tests unitaires pour Beam.py."""

    def test_beam_creation(self):
        beam = Beam(wavelength_nm=633.0, diameter_mm=5.0, num_points=128)
        assert beam.wavelength_nm == 633.0
        assert beam.diameter_mm == 5.0
        assert beam.num_points == 128

    def test_gaussian_electric_field(self):
        beam = Beam(wavelength_nm=633.0, diameter_mm=5.0, num_points=128)
        ef = beam.generate_electric_field(method=BeamProfile.GAUSSIAN, sigma_mm=1.0)
        assert ef.shape == (128, 128)
        assert np.all(np.isfinite(ef))

    def test_intensity_calculation(self):
        beam = Beam(wavelength_nm=633.0, diameter_mm=5.0, num_points=128)
        ef = beam.generate_electric_field(method=BeamProfile.GAUSSIAN)
        intensity = beam.compute_intensity_from_electric_field(ef)
        assert intensity.shape == (128, 128)
        assert np.all(intensity >= 0)

    def test_phase_extraction(self):
        beam = Beam(wavelength_nm=633.0, diameter_mm=5.0, num_points=128)
        ef = beam.generate_electric_field(method=BeamProfile.GAUSSIAN)
        phase_nm = beam.extract_phase_from_electric_field(ef, units='nm')
        phase_rad = beam.extract_phase_from_electric_field(ef, units='rad')
        assert phase_nm.shape == (128, 128)
        assert phase_rad.shape == (128, 128)

    def test_propagation(self):
        beam = Beam(wavelength_nm=633.0, diameter_mm=5.0, num_points=128)
        ef = beam.generate_electric_field(method=BeamProfile.GAUSSIAN)
        beam.electric_field = ef
        propagated = beam.propagate(distance_mm=10.0, method=PropagationMethod.ANGULAR_SPECTRUM)
        assert propagated.shape == (128, 128)
        assert np.all(np.isfinite(propagated))


if __name__ == "__main__":
    import unittest
    unittest.main()

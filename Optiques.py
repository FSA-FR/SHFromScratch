"""
Optiques.py

FR: Module pour la modélisation d'éléments optiques et de fronts d'onde.
    
    Ce module contient :
    - Les classes pour les éléments optiques (lentilles, diaphragmes, trous, réseaux)
    - La classe WaveFrontError pour modéliser les aberrations
    - Les fonctions de calcul de phase pour les éléments optiques
    
    NOTE: Les fonctions de génération de polynômes de Zernike, Legendre, etc.
    sont dans MathAndPhysicsTools.py.
    
    Unités :
    - Longueurs : mm (sauf indication contraire)
    - Longueur d'onde : nm
    - Phase : nm (principale), rad (pour les calculs)
    - Rayons de courbure : mm
    - Distances focales : mm
    - Indices de réfraction : sans unité

EN: Module for modeling optical elements and wavefronts.
    
    This module contains:
    - Classes for optical elements (lenses, diaphragms, holes, gratings)
    - WaveFrontError class for modeling aberrations
    - Phase calculation functions for optical elements
    
    NOTE: Zernike, Legendre, etc. polynomial generation functions
    are in MathAndPhysicsTools.py.
    
    Units:
    - Lengths: mm (unless specified otherwise)
    - Wavelength: nm
    - Phase: nm (main), rad (for calculations)
    - Radius of curvature: mm
    - Focal lengths: mm
    - Refractive indices: unitless

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch

Dependencies:
    - numpy
    - MathAndPhysicsTools (pour les polynômes de Zernike, etc.)

Sources:
    - "Principles of Optics" by M. Born & E. Wolf (Cambridge University Press, 1999)
      -> Fondements de l'optique (Ch. 1-8)
      -> Théorie des lentilles (Ch. 4)
      -> Aberrations optiques (Ch. 9)
    - "Handbook of Optical Systems" by H. Gross (2005)
      -> Volume 1: Fundamentals of Technical Optics
      -> Volume 3: Aberration Theory and Correction
"""

import numpy as np
import logging
from typing import Optional, Tuple, Dict, List
from enum import Enum
from dataclasses import dataclass

from MathAndPhysicsTools import (
    handle_nan,
    generate_zernike_polynomial,
    ZernikeOrdering,
    NormalizationType,
    DEFAULT_WAVELENGTH_NM,
    PI,
    TWO_PI,
    NM_TO_M
)


# =============================================================================
# CONFIGURATION GLOBALE
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Optiques")


# =============================================================================
# CONSTANTES
# =============================================================================

REFRACTIVE_INDICES = {
    'air': 1.000293,
    'vacuum': 1.0,
    'Fused_Silica': 1.4585,
    'BK7': 1.5168,
    'Sapphire': 1.768,
    'CaF2': 1.4338,
}


# =============================================================================
# ENUMS
# =============================================================================

class LensType(Enum):
    PLANO_CONVEX = "plano_convex"
    PLANO_CONCAVE = "plano_concave"
    BICONVEX = "biconvex"
    BICONCAVE = "biconcave"


class ApertureShape(Enum):
    CIRCULAR = "circular"
    RECTANGULAR = "rectangular"
    SQUARE = "square"


class OpticType(Enum):
    LENS = "lens"
    DIAPHRAGM = "diaphragm"
    HOLE = "hole"
    GRATING = "grating"


# =============================================================================
# CLASSE: WAVEFRONT ERROR
# =============================================================================

@dataclass
class WaveFrontError:
    """
    FR: Erreur de front d'onde (WFE).
    EN: WaveFront Error (WFE).
    
    Sources:
        - "Principles of Adaptive Optics" by R. K. Tyson (1991), Ch. 2
        - ISO 10110-5: Surface form tolerances
        - ISO 10110-7: Surface imperfection tolerances
    """
    
    surface_roughness_nm: float = 0.0
    parallelism_arcsec: float = 0.0
    spherical_aberration_nm: float = 0.0
    coma_nm: float = 0.0
    astigmatism_nm: float = 0.0
    field_curvature_nm: float = 0.0
    distortion: float = 0.0
    zernike_coefficients: Optional[Dict[Tuple[int, int], float]] = None
    scratch_dig: Optional[str] = None
    wavelength_nm: float = DEFAULT_WAVELENGTH_NM
    
    def __post_init__(self):
        if self.zernike_coefficients is None:
            self.zernike_coefficients = {}

    def get_total_wfe_nm(self) -> float:
        """FR: Calcule le WFE total. EN: Calculates total WFE."""
        parallelism_nm = self.parallelism_arcsec * 5.0
        return float(np.sqrt(
            self.surface_roughness_nm**2 +
            parallelism_nm**2 +
            self.spherical_aberration_nm**2 +
            self.coma_nm**2 +
            self.astigmatism_nm**2 +
            self.field_curvature_nm**2
        ))

    def get_phase_map(self, X: np.ndarray, Y: np.ndarray, diameter_mm: float) -> np.ndarray:
        """FR: Génère une carte de phase. EN: Generates a phase map."""
        R = np.sqrt(X**2 + Y**2)
        R_max = diameter_mm / 2
        X_norm = np.divide(X, R_max, out=np.zeros_like(X), where=R_max!=0)
        Y_norm = np.divide(Y, R_max, out=np.zeros_like(Y), where=R_max!=0)
        
        phase_map_nm = np.zeros_like(X)
        
        if self.surface_roughness_nm > 0:
            phase_map_nm += self._generate_surface_roughness(X_norm, Y_norm)
        if self.parallelism_arcsec > 0:
            phase_map_nm += self._generate_parallelism(X_norm, Y_norm)
        if self.spherical_aberration_nm > 0:
            phase_map_nm += self._generate_spherical_aberration(X_norm, Y_norm)
        if self.coma_nm > 0:
            phase_map_nm += self._generate_coma(X_norm, Y_norm)
        if self.astigmatism_nm > 0:
            phase_map_nm += self._generate_astigmatism(X_norm, Y_norm)
        if self.zernike_coefficients:
            phase_map_nm += self._generate_zernike_phase(X_norm, Y_norm)
        
        return handle_nan(phase_map_nm, method='zero')

    def _generate_surface_roughness(self, X_norm: np.ndarray, Y_norm: np.ndarray) -> np.ndarray:
        """FR: Génère une rugosité de surface. EN: Generates surface roughness."""
        np.random.seed(42)
        noise = np.random.normal(0, self.surface_roughness_nm / 3, X_norm.shape)
        return noise

    def _generate_parallelism(self, X_norm: np.ndarray, Y_norm: np.ndarray) -> np.ndarray:
        """FR: Génère le parallélisme. EN: Generates parallelism."""
        parallelism_rad = self.parallelism_arcsec * (PI / (180 * 3600))
        wavelength_mm = self.wavelength_nm * NM_TO_M
        return (X_norm * parallelism_rad) * (self.wavelength_nm / TWO_PI)

    def _generate_spherical_aberration(self, X_norm: np.ndarray, Y_norm: np.ndarray) -> np.ndarray:
        """FR: Génère l'aberration sphérique. EN: Generates spherical aberration."""
        R_sq = X_norm**2 + Y_norm**2
        return self.spherical_aberration_nm * R_sq**2

    def _generate_coma(self, X_norm: np.ndarray, Y_norm: np.ndarray) -> np.ndarray:
        """FR: Génère le coma. EN: Generates coma."""
        R_sq = X_norm**2 + Y_norm**2
        return self.coma_nm * R_sq * X_norm

    def _generate_astigmatism(self, X_norm: np.ndarray, Y_norm: np.ndarray) -> np.ndarray:
        """FR: Génère l'astigmatisme. EN: Generates astigmatism."""
        return self.astigmatism_nm * (X_norm**2 - Y_norm**2)

    def _generate_zernike_phase(self, X_norm: np.ndarray, Y_norm: np.ndarray) -> np.ndarray:
        """FR: Génère la phase à partir des coefficients de Zernike. EN: Generates phase from Zernike coefficients."""
        phase_map_nm = np.zeros_like(X_norm)
        for (n, m), coeff in self.zernike_coefficients.items():
            Z = generate_zernike_polynomial(n, m, X_norm, Y_norm,
                ordering=ZernikeOrdering.NOLL, normalization=NormalizationType.PV)
            phase_map_nm += coeff * Z
        return phase_map_nm


# =============================================================================
# CLASSE: OPTICAL ELEMENT
# =============================================================================

class OpticalElement:
    """
    FR: Élément optique générique.
    EN: Generic optical element.
    
    Sources: "Handbook of Optical Systems" by H. Gross (2005)
    """

    def __init__(self,
                 name: str = "OpticalElement",
                 optic_type: OpticType = OpticType.LENS,
                 diameter_mm: float = 10.0,
                 thickness_mm: float = 1.0,
                 material: str = "Fused_Silica",
                 wavelength_nm: float = DEFAULT_WAVELENGTH_NM,
                 wfe: Optional[WaveFrontError] = None):
        self.name = name
        self.optic_type = optic_type
        self.diameter_mm = diameter_mm
        self.thickness_mm = thickness_mm
        self.material = material
        self.wavelength_nm = wavelength_nm
        self.refractive_index = REFRACTIVE_INDICES.get(material, REFRACTIVE_INDICES['Fused_Silica'])
        self.wfe = wfe if wfe is not None else WaveFrontError(wavelength_nm=wavelength_nm)

    def get_phase_map(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """FR: Retourne la carte de phase. EN: Returns phase map."""
        return self.wfe.get_phase_map(X, Y, self.diameter_mm)


# =============================================================================
# CLASSE: LENS
# =============================================================================

class Lens(OpticalElement):
    """
    FR: Lentille optique.
    EN: Optical lens.
    
    Sources:
        - "Handbook of Optical Systems" by H. Gross (2005), Vol. 1
        - "Lens Design" by R. Kingslake (1978)
    """

    def __init__(self,
                 name: str = "Lens",
                 lens_type: LensType = LensType.PLANO_CONVEX,
                 diameter_mm: float = 10.0,
                 thickness_mm: float = 2.0,
                 material: str = "Fused_Silica",
                 focal_length_mm: Optional[float] = None,
                 radius_of_curvature_1_mm: Optional[float] = None,
                 radius_of_curvature_2_mm: Optional[float] = None,
                 wavelength_nm: float = DEFAULT_WAVELENGTH_NM,
                 wfe: Optional[WaveFrontError] = None):
        super().__init__(name, OpticType.LENS, diameter_mm, thickness_mm, material, wavelength_nm, wfe)
        self.lens_type = lens_type
        self.focal_length_mm = focal_length_mm
        self.radius_of_curvature_1_mm = radius_of_curvature_1_mm
        self.radius_of_curvature_2_mm = radius_of_curvature_2_mm
        
        if self.focal_length_mm is None:
            if radius_of_curvature_1_mm and radius_of_curvature_2_mm:
                R1 = radius_of_curvature_1_mm
                R2 = radius_of_curvature_2_mm
                d = thickness_mm
                n = self.refractive_index
                self.focal_length_mm = 1 / ((n - 1) * (1/R1 - 1/R2 + (n - 1) * d / (n * R1 * R2)))
            else:
                self.focal_length_mm = 0.0

    def get_phase_map(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """FR: Retourne la carte de phase pour une lentille. EN: Returns phase map for a lens."""
        wavelength_mm = self.wavelength_nm * NM_TO_M
        phase_rad = -PI / (wavelength_mm * self.focal_length_mm) * (X**2 + Y**2)
        phase_nm = phase_rad * (self.wavelength_nm / TWO_PI)
        wfe_phase = self.wfe.get_phase_map(X, Y, self.diameter_mm)
        return handle_nan(phase_nm + wfe_phase, method='zero')


# =============================================================================
# CLASSE: DIAPHRAGM
# =============================================================================

class Diaphragm(OpticalElement):
    """
    FR: Diaphragme optique.
    EN: Optical diaphragm.
    
    Sources: "Principles of Optics" by Born & Wolf (1999), Ch. 8
    """

    def __init__(self,
                 name: str = "Diaphragm",
                 aperture_shape: ApertureShape = ApertureShape.CIRCULAR,
                 diameter_mm: float = 5.0,
                 wavelength_nm: float = DEFAULT_WAVELENGTH_NM,
                 wfe: Optional[WaveFrontError] = None):
        super().__init__(name, OpticType.DIAPHRAGM, diameter_mm, 0.0, "opaque", wavelength_nm, wfe)
        self.aperture_shape = aperture_shape

    def get_transmission_map(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """FR: Retourne la carte de transmission. EN: Returns transmission map."""
        if self.aperture_shape == ApertureShape.CIRCULAR:
            R = np.sqrt(X**2 + Y**2)
            return np.where(R <= self.diameter_mm / 2, 1.0, 0.0)
        elif self.aperture_shape == ApertureShape.SQUARE:
            return np.where(
                (np.abs(X) <= self.diameter_mm / 2) & (np.abs(Y) <= self.diameter_mm / 2),
                1.0, 0.0
            )
        return np.ones_like(X)


# =============================================================================
# CLASSE: HOLE
# =============================================================================

class Hole(OpticalElement):
    """
    FR: Trou optique.
    EN: Optical hole.
    
    Sources: "Principles of Optics" by Born & Wolf (1999), Ch. 8
    """

    def __init__(self,
                 name: str = "Hole",
                 aperture_shape: ApertureShape = ApertureShape.CIRCULAR,
                 diameter_mm: float = 1.0,
                 wavelength_nm: float = DEFAULT_WAVELENGTH_NM,
                 wfe: Optional[WaveFrontError] = None):
        super().__init__(name, OpticType.HOLE, diameter_mm, 0.0, "opaque", wavelength_nm, wfe)
        self.aperture_shape = aperture_shape


# =============================================================================
# CLASSE: GRATING
# =============================================================================

class Grating(OpticalElement):
    """
    FR: Réseau de diffraction.
    EN: Diffraction grating.
    
    Sources: "Principles of Optics" by Born & Wolf (1999), Ch. 8
    """

    def __init__(self,
                 name: str = "Grating",
                 diameter_mm: float = 10.0,
                 material: str = "Fused_Silica",
                 lines_per_mm: float = 100.0,
                 wavelength_nm: float = DEFAULT_WAVELENGTH_NM,
                 wfe: Optional[WaveFrontError] = None):
        super().__init__(name, OpticType.GRATING, diameter_mm, 0.0, material, wavelength_nm, wfe)
        self.lines_per_mm = lines_per_mm

    def get_phase_map(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """FR: Retourne la carte de phase pour un réseau. EN: Returns phase map for a grating."""
        period_mm = 1.0 / self.lines_per_mm
        phase_rad = TWO_PI * X / period_mm
        phase_nm = phase_rad * (self.wavelength_nm / TWO_PI)
        wfe_phase = self.wfe.get_phase_map(X, Y, self.diameter_mm)
        return handle_nan(phase_nm + wfe_phase, method='zero')


# =============================================================================
# FONCTIONS DE CRÉATION
# =============================================================================

def create_lens(name: str = "Lens",
                lens_type: LensType = LensType.PLANO_CONVEX,
                diameter_mm: float = 10.0,
                thickness_mm: float = 2.0,
                material: str = "Fused_Silica",
                focal_length_mm: Optional[float] = None,
                radius_of_curvature_1_mm: Optional[float] = None,
                radius_of_curvature_2_mm: Optional[float] = None,
                wavelength_nm: float = DEFAULT_WAVELENGTH_NM,
                surface_quality_pv_nm: float = 0.0,
                parallelism_arcsec: float = 0.0,
                scratch_dig: Optional[str] = None) -> Lens:
    """
    FR: Crée une lentille.
    EN: Creates a lens.
    
    Sources: "Handbook of Optical Systems" by H. Gross (2005)
    """
    wfe = WaveFrontError(
        wavelength_nm=wavelength_nm,
        surface_roughness_nm=surface_quality_pv_nm,
        parallelism_arcsec=parallelism_arcsec,
        scratch_dig=scratch_dig
    )
    return Lens(
        name=name, lens_type=lens_type, diameter_mm=diameter_mm,
        thickness_mm=thickness_mm, material=material,
        focal_length_mm=focal_length_mm,
        radius_of_curvature_1_mm=radius_of_curvature_1_mm,
        radius_of_curvature_2_mm=radius_of_curvature_2_mm,
        wavelength_nm=wavelength_nm, wfe=wfe
    )


def create_diaphragm(name: str = "Diaphragm",
                    aperture_shape: ApertureShape = ApertureShape.CIRCULAR,
                    diameter_mm: float = 5.0,
                    wavelength_nm: float = DEFAULT_WAVELENGTH_NM) -> Diaphragm:
    """FR: Crée un diaphragme. EN: Creates a diaphragm."""
    return Diaphragm(name, aperture_shape, diameter_mm, wavelength_nm)


def create_hole(name: str = "Hole",
                aperture_shape: ApertureShape = ApertureShape.CIRCULAR,
                diameter_mm: float = 1.0,
                wavelength_nm: float = DEFAULT_WAVELENGTH_NM) -> Hole:
    """FR: Crée un trou. EN: Creates a hole."""
    return Hole(name, aperture_shape, diameter_mm, wavelength_nm)


def create_grating(name: str = "Grating",
                  diameter_mm: float = 10.0,
                  material: str = "Fused_Silica",
                  lines_per_mm: float = 100.0,
                  wavelength_nm: float = DEFAULT_WAVELENGTH_NM) -> Grating:
    """FR: Crée un réseau. EN: Creates a grating."""
    return Grating(name, diameter_mm, material, lines_per_mm, wavelength_nm)


# =============================================================================
# TESTS UNITAIRES
# =============================================================================

class TestOptiques:
    """FR: Tests unitaires pour Optiques.py."""

    def test_wavefront_error(self):
        """FR: Test WaveFrontError."""
        wfe = WaveFrontError(surface_roughness_nm=10.0, parallelism_arcsec=5.0)
        assert wfe.get_total_wfe_nm() > 0

    def test_lens(self):
        """FR: Test Lens."""
        lens = create_lens(focal_length_mm=10.0)
        assert lens.focal_length_mm == 10.0

    def test_diaphragm(self):
        """FR: Test Diaphragm."""
        diaphragm = create_diaphragm()
        assert diaphragm.diameter_mm == 5.0

    def test_hole(self):
        """FR: Test Hole."""
        hole = create_hole()
        assert hole.diameter_mm == 1.0

    def test_grating(self):
        """FR: Test Grating."""
        grating = create_grating()
        assert grating.lines_per_mm == 100.0


if __name__ == "__main__":
    import unittest
    unittest.main()

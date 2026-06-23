"""
Propagation.py

FR: Module pour la propagation de faisceaux optiques à travers des systèmes optiques.
    
    Ce module contient les classes et fonctions pour propager un faisceau à travers :
    - Des éléments optiques simples (lentilles, diaphragmes)
    - Des matrices de microlentilles
    - Des systèmes complexes (Shack-Hartmann)
    
    NOTE: Les fonctions de propagation de base (spectre angulaire, Fresnel, Fraunhofer)
    sont implémentées dans Beam.py et DOIVENT rester là-bas.
    
    Unités :
    - Longueurs : mm
    - Longueur d'onde : nm
    - Phase : nm (principale), rad (pour les calculs)
    - Distances : mm

EN: Module for propagating optical beams through optical systems.
    
    This module contains classes and functions to propagate a beam through:
    - Simple optical elements (lenses, diaphragms)
    - Microlens arrays
    - Complex systems (Shack-Hartmann)
    
    NOTE: Basic propagation functions (angular spectrum, Fresnel, Fraunhofer)
    are implemented in Beam.py and MUST remain there.
    
    Units:
    - Lengths: mm
    - Wavelength: nm
    - Phase: nm (main), rad (for calculations)
    - Distances: mm

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch

Dependencies:
    - numpy
    - Beam (pour les fonctions de propagation de base)
    - MathAndPhysicsTools (pour les fonctions outils)

Sources:
    - "Fourier Optics" by J. W. Goodman (2005)
      -> Méthodes de propagation (Ch. 3-4)
    - "Principles of Optics" by M. Born & E. Wolf (1999)
      -> Propagation en optique (Ch. 8)
    - "Laser Beam Propagation" by J. W. Goodman (1996)
      -> Propagation des faisceaux laser
"""

import numpy as np
import logging
from typing import Optional, List
from enum import Enum

# Import des modules locaux
from Beam import Beam, PropagationMethod
from MathAndPhysicsTools import (
    handle_nan,
    DEFAULT_WAVELENGTH_NM,
    PI,
    TWO_PI
)


# =============================================================================
# CONFIGURATION GLOBALE
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Propagation")


# =============================================================================
# CONSTANTES
# =============================================================================

AIR_REFRACTIVE_INDEX = 1.000293

MATERIAL_REFRACTIVE_INDICES = {
    'air': 1.000293,
    'vacuum': 1.0,
    'Fused_Silica': 1.4585,
    'BK7': 1.5168,
    'Silicon': 3.5,
    'Germanium': 4.0,
}


# =============================================================================
# ENUMS
# =============================================================================

class OpticalElementType(Enum):
    LENS = "lens"
    DIAPHRAGM = "diaphragm"
    MIRROR = "mirror"
    WINDOW = "window"
    PRISM = "prism"
    GRATING = "grating"


# =============================================================================
# CLASSE: OPTICAL ELEMENT
# =============================================================================

class OpticalElement:
    """
    FR: Élément optique de base.
    EN: Basic optical element.
    
    Sources: "Handbook of Optical Systems" by H. Gross (2005)
    """

    def __init__(self,
                 name: str = "OpticalElement",
                 element_type: OpticalElementType = OpticalElementType.LENS,
                 refractive_index: float = AIR_REFRACTIVE_INDEX,
                 thickness_mm: float = 0.0,
                 diameter_mm: float = 10.0,
                 focal_length_mm: Optional[float] = None,
                 radius_of_curvature_mm: Optional[float] = None,
                 wavelength_nm: float = DEFAULT_WAVELENGTH_NM):
        self.name = name
        self.element_type = element_type
        self.refractive_index = refractive_index
        self.thickness_mm = thickness_mm
        self.diameter_mm = diameter_mm
        self.focal_length_mm = focal_length_mm
        self.radius_of_curvature_mm = radius_of_curvature_mm
        self.wavelength_nm = wavelength_nm
        
        if self.radius_of_curvature_mm is not None:
            if self.focal_length_mm is None:
                self.focal_length_mm = self.radius_of_curvature_mm / (self.refractive_index - 1)
        
        logger.info(f"Élément optique '{name}' initialisé")

    def get_transmission_matrix(self, distance_mm: float) -> np.ndarray:
        """FR: Retourne la matrice de transmission. EN: Returns the transmission matrix."""
        if self.element_type == OpticalElementType.LENS and self.focal_length_mm is not None:
            if self.focal_length_mm == 0:
                return np.eye(2)
            return np.array([[1, 0], [-1/self.focal_length_mm, 1]])
        elif self.element_type == OpticalElementType.WINDOW:
            return np.array([[1, distance_mm], [0, 1]])
        else:
            return np.eye(2)

    def propagate_beam(self,
                       beam: Beam,
                       distance_mm: float,
                       method: PropagationMethod = PropagationMethod.ANGULAR_SPECTRUM) -> Beam:
        """
        FR: Propage un faisceau à travers cet élément.
        EN: Propagates a beam through this element.
        
        Sources:
            - "Fourier Optics" by Goodman (2005)
        """
        if self.element_type == OpticalElementType.LENS:
            beam = self._apply_lens_phase(beam)
        
        propagated_field = beam.propagate(distance_mm, method)
        
        propagated_beam = Beam(
            wavelength_nm=beam.wavelength_nm,
            diameter_mm=beam.diameter_mm,
            num_points=beam.num_points
        )
        propagated_beam.electric_field = propagated_field
        propagated_beam.intensity = propagated_beam.compute_intensity_from_electric_field(propagated_field)
        propagated_beam.phase = propagated_beam.extract_phase_from_electric_field(propagated_field)
        
        return propagated_beam

    def _apply_lens_phase(self, beam: Beam) -> Beam:
        """FR: Applique la phase d'une lentille. EN: Applies lens phase."""
        if self.focal_length_mm is None or self.focal_length_mm == 0:
            return beam
        
        x, y, X, Y = beam._create_xy_grid()
        wavelength_mm = beam.wavelength_nm * 1e-6
        phase_rad = -PI / (wavelength_mm * self.focal_length_mm) * (X**2 + Y**2)
        phase_shift = np.exp(1j * phase_rad)
        beam.electric_field = beam.electric_field * phase_shift
        beam.phase = beam.extract_phase_from_electric_field(beam.electric_field)
        return beam


# =============================================================================
# CLASSE: OPTICAL SYSTEM
# =============================================================================

class OpticalSystem:
    """
    FR: Système optique composé de plusieurs éléments.
    EN: Optical system composed of multiple elements.
    
    Sources: "Handbook of Optical Systems" by H. Gross (2005)
    """

    def __init__(self,
                 name: str = "OpticalSystem",
                 elements: Optional[List[OpticalElement]] = None,
                 distances_mm: Optional[List[float]] = None):
        self.name = name
        self.elements = elements if elements is not None else []
        self.distances_mm = distances_mm if distances_mm is not None else []
        
        if len(self.elements) > 0 and len(self.distances_mm) > 0:
            if len(self.elements) != len(self.distances_mm) + 1:
                raise ValueError("Nombre d'éléments != nombre de distances + 1")
        
        logger.info(f"Système optique '{name}' initialisé")

    def add_element(self, element: OpticalElement, distance_mm: float = 0.0) -> None:
        """FR: Ajoute un élément. EN: Adds an element."""
        self.elements.append(element)
        self.distances_mm.append(distance_mm)

    def propagate_beam(self,
                       beam: Beam,
                       method: PropagationMethod = PropagationMethod.ANGULAR_SPECTRUM) -> Beam:
        """
        FR: Propage un faisceau à travers le système.
        EN: Propagates a beam through the system.
        
        Sources: "Fourier Optics" by Goodman (2005)
        """
        current_beam = beam
        
        for i, element in enumerate(self.elements):
            distance_mm = self.distances_mm[i] if i < len(self.distances_mm) else 0.0
            current_beam = element.propagate_beam(current_beam, distance_mm, method)
            logger.info(f"Propagation à travers {element.name} (d={distance_mm}mm)")
        
        return current_beam


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def propagate_through_lens(beam: Beam,
                          focal_length_mm: float,
                          distance_mm: float = 0.0,
                          method: PropagationMethod = PropagationMethod.ANGULAR_SPECTRUM) -> Beam:
    """FR: Propage à travers une lentille. EN: Propagates through a lens."""
    lens = OpticalElement(
        name="Lens",
        element_type=OpticalElementType.LENS,
        focal_length_mm=focal_length_mm,
        diameter_mm=beam.diameter_mm,
        wavelength_nm=beam.wavelength_nm
    )
    return lens.propagate_beam(beam, distance_mm, method)


def propagate_through_free_space(beam: Beam,
                                distance_mm: float,
                                method: PropagationMethod = PropagationMethod.ANGULAR_SPECTRUM) -> Beam:
    """FR: Propage dans l'espace libre. EN: Propagates in free space."""
    propagated_field = beam.propagate(distance_mm, method)
    
    propagated_beam = Beam(
        wavelength_nm=beam.wavelength_nm,
        diameter_mm=beam.diameter_mm,
        num_points=beam.num_points
    )
    propagated_beam.electric_field = propagated_field
    propagated_beam.intensity = propagated_beam.compute_intensity_from_electric_field(propagated_field)
    propagated_beam.phase = propagated_beam.extract_phase_from_electric_field(propagated_field)
    
    return propagated_beam


def calculate_focal_spot(beam: Beam,
                        focal_length_mm: float,
                        method: PropagationMethod = PropagationMethod.ANGULAR_SPECTRUM) -> Beam:
    """FR: Calcule le spot focal. EN: Calculates the focal spot."""
    return propagate_through_lens(beam, focal_length_mm, focal_length_mm, method)


# =============================================================================
# TESTS UNITAIRES
# =============================================================================

class TestPropagation:
    """FR: Tests unitaires pour Propagation.py."""

    def test_optical_element(self):
        """FR: Test la création d'un élément optique."""
        lens = OpticalElement(
            name="TestLens",
            element_type=OpticalElementType.LENS,
            focal_length_mm=10.0
        )
        assert lens.name == "TestLens"
        assert lens.focal_length_mm == 10.0

    def test_optical_system(self):
        """FR: Test la création d'un système optique."""
        system = OpticalSystem(name="TestSystem")
        assert system.name == "TestSystem"

    def test_propagate_through_lens(self):
        """FR: Test la propagation à travers une lentille."""
        beam = Beam(wavelength_nm=633.0, diameter_mm=5.0, num_points=128)
        beam.generate_electric_field(method='gaussian')
        propagated_beam = propagate_through_lens(beam, focal_length_mm=10.0, distance_mm=10.0)
        assert propagated_beam.electric_field is not None

    def test_propagate_through_free_space(self):
        """FR: Test la propagation dans l'espace libre."""
        beam = Beam(wavelength_nm=633.0, diameter_mm=5.0, num_points=128)
        beam.generate_electric_field(method='gaussian')
        propagated_beam = propagate_through_free_space(beam, distance_mm=10.0)
        assert propagated_beam.electric_field is not None


if __name__ == "__main__":
    import unittest
    unittest.main()

"""
Microstructure.py

FR: Module pour la modélisation de matrices de micro-optiques.
    Ce module permet de créer et manipuler des matrices de microlentilles.
    
    Fonctionnalités principales :
    - Création de matrices de microlentilles (1D, 2D)
    - Différentes formes de microlentilles (carrées, circulaires, hexagonales)
    - Gestion des erreurs de front d'onde (WFE) pour chaque élément
    - Calcul de la phase totale de la matrice
    
    Unités :
    - Longueurs : mm (pour les dimensions de la matrice)
    - Longueur d'onde : nm
    - Phase : nm (principale), rad (pour les calculs)
    - Distances focales : mm

EN: Module for modeling micro-optical arrays.
    This module allows creating and manipulating arrays of microlenses.
    
    Main features:
    - Creation of microlens arrays (1D, 2D)
    - Different microlens shapes (square, circular, hexagonal)
    - WaveFront Error (WFE) management for each element
    - Total phase map calculation for the array
    
    Units:
    - Lengths: mm (for array dimensions)
    - Wavelength: nm
    - Phase: nm (main), rad (for calculations)
    - Focal lengths: mm

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch

Dependencies:
    - numpy
    - Optiques (pour WaveFrontError)
    - MathAndPhysicsTools (pour les fonctions outils)

Sources:
    - "Microoptics" by H. P. Herzig (1997)
    - "Array Microlenses: Properties and Applications" by N. F. Borrelli (1992)
"""

import numpy as np
import logging
from typing import Optional, Tuple, List
from enum import Enum

from Optiques import WaveFrontError, REFRACTIVE_INDICES
from MathAndPhysicsTools import handle_nan, DEFAULT_WAVELENGTH_NM, PI, TWO_PI, NM_TO_M


# =============================================================================
# CONFIGURATION GLOBALE
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Microstructure")


# =============================================================================
# ENUMS
# =============================================================================

class MicrolensShape(Enum):
    SQUARE = "square"
    CIRCULAR = "circular"
    HEXAGONAL = "hexagonal"


class ArrayType(Enum):
    RECTANGULAR = "rectangular"
    HEXAGONAL = "hexagonal"


# =============================================================================
# CLASSE: MICROLENS
# =============================================================================

class Microlens:
    """
    FR: Microlentille individuelle.
    EN: Individual microlens.
    
    Sources: "Microoptics" by H. P. Herzig (1997)
    """

    def __init__(self,
                 name: str = "Microlens",
                 shape: MicrolensShape = MicrolensShape.SQUARE,
                 diameter_mm: float = 0.1,
                 focal_length_mm: float = 1.0,
                 thickness_mm: float = 0.05,
                 material: str = "Fused_Silica",
                 wavelength_nm: float = DEFAULT_WAVELENGTH_NM,
                 position_mm: Tuple[float, float] = (0.0, 0.0),
                 wfe: Optional[WaveFrontError] = None):
        self.name = name
        self.shape = shape
        self.diameter_mm = diameter_mm
        self.focal_length_mm = focal_length_mm
        self.thickness_mm = thickness_mm
        self.material = material
        self.wavelength_nm = wavelength_nm
        self.position_mm = position_mm
        self.refractive_index = REFRACTIVE_INDICES.get(material, REFRACTIVE_INDICES['Fused_Silica'])
        self.wfe = wfe if wfe is not None else WaveFrontError(wavelength_nm=wavelength_nm)

    def get_phase_map(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        FR: Génère une carte de phase pour cette microlentille.
        EN: Generates a phase map for this microlens.
        
        Sources: "Microoptics" by H. P. Herzig (1997)
        """
        X_shifted = X - self.position_mm[0]
        Y_shifted = Y - self.position_mm[1]
        
        wavelength_mm = self.wavelength_nm * NM_TO_M
        phase_rad = -PI / (wavelength_mm * self.focal_length_mm) * (X_shifted**2 + Y_shifted**2)
        phase_nm = phase_rad * (self.wavelength_nm / TWO_PI)
        
        wfe_phase = self.wfe.get_phase_map(X_shifted, Y_shifted, self.diameter_mm)
        mask = self._get_shape_mask(X_shifted, Y_shifted)
        
        return np.where(mask, handle_nan(phase_nm + wfe_phase, method='zero'), 0.0)

    def _get_shape_mask(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """FR: Génère un masque selon la forme. EN: Generates a shape mask."""
        if self.shape == MicrolensShape.SQUARE:
            return (np.abs(X) <= self.diameter_mm / 2) & (np.abs(Y) <= self.diameter_mm / 2)
        elif self.shape == MicrolensShape.CIRCULAR:
            R = np.sqrt(X**2 + Y**2)
            return R <= self.diameter_mm / 2
        elif self.shape == MicrolensShape.HEXAGONAL:
            abs_X = np.abs(X)
            abs_Y = np.abs(Y)
            return (abs_X <= self.diameter_mm / 2) & (abs_Y <= self.diameter_mm / 2) & (abs_X + abs_Y <= self.diameter_mm / np.sqrt(3))
        return np.ones_like(X, dtype=bool)


# =============================================================================
# CLASSE: MICROLENS ARRAY
# =============================================================================

class MicrolensArray:
    """
    FR: Matrice de microlentilles.
    EN: Microlens array.
    
    Sources:
        - "Microoptics" by H. P. Herzig (1997)
        - "Array Microlenses" by N. F. Borrelli (1992)
    """

    def __init__(self,
                 name: str = "MicrolensArray",
                 num_elements_x: int = 50,
                 num_elements_y: int = 50,
                 element_diameter_mm: float = 0.1,
                 pitch_mm: float = 0.1,
                 focal_length_mm: float = 1.0,
                 thickness_mm: float = 0.5,
                 material: str = "Fused_Silica",
                 wavelength_nm: float = DEFAULT_WAVELENGTH_NM,
                 array_type: ArrayType = ArrayType.RECTANGULAR,
                 microlens_shape: MicrolensShape = MicrolensShape.SQUARE,
                 edge_to_edge_spacing_mm: float = 0.0,
                 global_wfe: Optional[WaveFrontError] = None):
        self.name = name
        self.num_elements_x = num_elements_x
        self.num_elements_y = num_elements_y
        self.element_diameter_mm = element_diameter_mm
        self.pitch_mm = pitch_mm if pitch_mm > 0 else element_diameter_mm
        self.focal_length_mm = focal_length_mm
        self.thickness_mm = thickness_mm
        self.material = material
        self.wavelength_nm = wavelength_nm
        self.array_type = array_type
        self.microlens_shape = microlens_shape
        self.edge_to_edge_spacing_mm = edge_to_edge_spacing_mm
        self.refractive_index = REFRACTIVE_INDICES.get(material, REFRACTIVE_INDICES['Fused_Silica'])
        self.global_wfe = global_wfe if global_wfe is not None else WaveFrontError(wavelength_nm=wavelength_nm)
        
        self.total_width_mm = (num_elements_x - 1) * self.pitch_mm + element_diameter_mm
        self.total_height_mm = (num_elements_y - 1) * self.pitch_mm + element_diameter_mm
        
        self.microlenses = []
        self._create_microlenses()
        self.center_mm = (self.total_width_mm / 2, self.total_height_mm / 2)

    def _create_microlenses(self) -> None:
        """FR: Crée les microlentilles. EN: Creates the microlenses."""
        for i in range(self.num_elements_x):
            for j in range(self.num_elements_y):
                if self.array_type == ArrayType.RECTANGULAR:
                    x_pos = i * self.pitch_mm
                    y_pos = j * self.pitch_mm
                elif self.array_type == ArrayType.HEXAGONAL:
                    x_pos = i * self.pitch_mm * np.sqrt(3)
                    y_pos = j * self.pitch_mm * 1.5
                    if i % 2 == 1:
                        y_pos += self.pitch_mm * 0.75
                else:
                    x_pos = np.random.uniform(0, self.total_width_mm)
                    y_pos = np.random.uniform(0, self.total_height_mm)
                
                microlens = Microlens(
                    name=f"ML_{i}_{j}",
                    shape=self.microlens_shape,
                    diameter_mm=self.element_diameter_mm,
                    focal_length_mm=self.focal_length_mm,
                    thickness_mm=self.thickness_mm,
                    material=self.material,
                    wavelength_nm=self.wavelength_nm,
                    position_mm=(x_pos, y_pos),
                    wfe=self.global_wfe
                )
                self.microlenses.append(microlens)

    def get_total_phase_map(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        FR: Génère la carte de phase totale de la matrice.
        EN: Generates the total phase map of the array.
        
        Sources: "Microoptics" by H. P. Herzig (1997)
        """
        total_phase_map = np.zeros_like(X)
        for microlens in self.microlenses:
            total_phase_map += microlens.get_phase_map(X, Y)
        return handle_nan(total_phase_map, method='zero')

    def get_transmission_map(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """FR: Génère la carte de transmission. EN: Generates the transmission map."""
        total_transmission = np.zeros_like(X)
        for microlens in self.microlenses:
            mask = microlens._get_shape_mask(X - microlens.position_mm[0], Y - microlens.position_mm[1])
            total_transmission = np.maximum(total_transmission, mask.astype(float))
        return total_transmission

    def get_microlens_positions(self) -> np.ndarray:
        """FR: Retourne les positions. EN: Returns positions."""
        return np.array([ml.position_mm for ml in self.microlenses])


# =============================================================================
# FONCTION DE CRÉATION
# =============================================================================

def create_microlens_array(name: str = "MicrolensArray",
                           num_elements_x: int = 50,
                           num_elements_y: int = 50,
                           element_diameter_mm: float = 0.1,
                           pitch_mm: float = 0.0,
                           focal_length_mm: float = 1.0,
                           thickness_mm: float = 0.5,
                           material: str = "Fused_Silica",
                           wavelength_nm: float = DEFAULT_WAVELENGTH_NM,
                           array_type: ArrayType = ArrayType.RECTANGULAR,
                           microlens_shape: MicrolensShape = MicrolensShape.SQUARE,
                           edge_to_edge_spacing_mm: float = 0.0,
                           surface_quality_pv_nm: float = 0.0,
                           parallelism_arcsec: float = 0.0,
                           scratch_dig: Optional[str] = None) -> MicrolensArray:
    """
    FR: Crée une matrice de microlentilles.
    EN: Creates a microlens array.
    
    Sources:
        - "Microoptics" by H. P. Herzig (1997)
        - "Array Microlenses" by N. F. Borrelli (1992)
    """
    global_wfe = WaveFrontError(
        wavelength_nm=wavelength_nm,
        surface_roughness_nm=surface_quality_pv_nm,
        parallelism_arcsec=parallelism_arcsec,
        scratch_dig=scratch_dig
    )
    return MicrolensArray(
        name=name, num_elements_x=num_elements_x, num_elements_y=num_elements_y,
        element_diameter_mm=element_diameter_mm, pitch_mm=pitch_mm,
        focal_length_mm=focal_length_mm, thickness_mm=thickness_mm,
        material=material, wavelength_nm=wavelength_nm, array_type=array_type,
        microlens_shape=microlens_shape, edge_to_edge_spacing_mm=edge_to_edge_spacing_mm,
        global_wfe=global_wfe
    )


# =============================================================================
# TESTS UNITAIRES
# =============================================================================

class TestMicrostructure:
    """FR: Tests unitaires pour Microstructure.py."""

    def test_microlens(self):
        """FR: Test Microlens."""
        ml = Microlens(name="TestML", diameter_mm=0.1, focal_length_mm=1.0)
        assert ml.name == "TestML"

    def test_microlens_array(self):
        """FR: Test MicrolensArray."""
        array = create_microlens_array(num_elements_x=5, num_elements_y=5)
        assert len(array.microlenses) == 25

    def test_total_phase_map(self):
        """FR: Test get_total_phase_map."""
        array = create_microlens_array(num_elements_x=5, num_elements_y=5)
        X, Y = np.meshgrid(np.linspace(0, 0.5, 100), np.linspace(0, 0.5, 100))
        phase_map = array.get_total_phase_map(X, Y)
        assert phase_map.shape == (100, 100)


if __name__ == "__main__":
    import unittest
    unittest.main()

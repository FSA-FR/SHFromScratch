"""
Material_Behaviour.py

FR: Module pour la modélisation du comportement des matériaux optiques.
    Ce module contient les classes et fonctions pour décrire les propriétés
    optiques des matériaux :
    - Indice de réfraction
    - Dispersion chromatique
    - Transmission et absorption
    - Réflexion
    
    Unités :
    - Longueurs : mm (pour les épaisseurs)
    - Longueur d'onde : nm
    - Indice de réfraction : sans unité
    - Coefficient d'absorption : mm⁻¹

EN: Module for modeling optical material behavior.
    This module contains classes and functions to describe optical properties
    of materials:
    - Refractive index
    - Chromatic dispersion
    - Transmission and absorption
    - Reflection
    
    Units:
    - Lengths: mm (for thicknesses)
    - Wavelength: nm
    - Refractive index: unitless
    - Absorption coefficient: mm⁻¹

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch

Dependencies:
    - numpy
    - MathAndPhysicsTools (pour les fonctions outils)

Sources:
    - "Handbook of Optics" by W. G. Driscoll (1978)
      -> Propriétés optiques des matériaux (Vol. I, Ch. 2)
    - "Optical Properties of Materials" by M. Fox (2010)
      -> Indice de réfraction et dispersion
    - "Fundamentals of Photonics" by B. E. A. Saleh & M. C. Teich (2007)
      -> Propagation dans les matériaux
"""

import numpy as np
import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

from MathAndPhysicsTools import handle_nan, DEFAULT_WAVELENGTH_NM


# =============================================================================
# CONFIGURATION GLOBALE
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Material_Behaviour")


# =============================================================================
# CONSTANTES
# =============================================================================

# Indices de réfraction à 633 nm (He-Ne laser)
REFRACTIVE_INDICES_633NM = {
    'air': 1.000293,
    'vacuum': 1.0,
    'Fused_Silica': 1.4585,
    'BK7': 1.5168,
    'Sapphire': 1.768,
    'CaF2': 1.4338,
    'Silicon': 3.5,
    'Germanium': 4.0,
    'ZnSe': 2.4,
    'GaAs': 3.3,
    'InP': 3.2,
}

# Coefficients de Sellmeier pour quelques matériaux
# n² = 1 + B1λ²/(λ² - C1) + B2λ²/(λ² - C2) + B3λ²/(λ² - C3)
# où λ est en µm
SELLMEIER_COEFFICIENTS = {
    'Fused_Silica': {'B1': 0.6961663, 'C1': 0.0684043, 'B2': 0.4079426, 'C2': 0.1162414, 'B3': 0.8974794, 'C3': 9.896161},
    'BK7': {'B1': 1.03961212, 'C1': 0.00600069867, 'B2': 0.231792344, 'C2': 0.0200179144, 'B3': 1.01046945, 'C3': 103.560653},
    'CaF2': {'B1': 0.5675888, 'C1': 0.00354178, 'B2': 0.4716143, 'C2': 0.0117549, 'B3': 3.8484723, 'C3': 1200.5558},
    'Sapphire': {'B1': 1.023798, 'C1': 0.0037754, 'B2': 1.503182, 'C2': 0.0122544, 'B3': 5.509158, 'C3': 321.7706},
}


# =============================================================================
# ENUMS
# =============================================================================

class MaterialType(Enum):
    """
    FR: Type de matériau.
        
    EN: Material type.
    """
    DIELECTRIC = "dielectric"
    SEMICONDUCTOR = "semiconductor"
    METAL = "metal"
    GAS = "gas"
    LIQUID = "liquid"


class DispersionModel(Enum):
    """
    FR: Modèle de dispersion.
        
    EN: Dispersion model.
    """
    CONSTANT = "constant"  # Indice constant
    SELLMEIER = "sellmeier"  # Formule de Sellmeier
    CAUCHY = "cauchy"  # Formule de Cauchy
    HERZBERGER = "herzberger"  # Formule de Herzberger


# =============================================================================
# CLASSE: OPTICAL MATERIAL
# =============================================================================

@dataclass
class OpticalMaterial:
    """
    FR: Matériau optique.
        
        Décrit les propriétés optiques d'un matériau.
        
    EN: Optical material.
        
        Describes the optical properties of a material.
    
    Attributes:
        name (str): Nom du matériau.
        material_type (MaterialType): Type de matériau.
        refractive_index (float): Indice de réfraction à la longueur d'onde de référence.
        reference_wavelength_nm (float): Longueur d'onde de référence en nm.
        dispersion_model (DispersionModel): Modèle de dispersion.
        dispersion_coefficients (Dict): Coefficients du modèle de dispersion.
        absorption_coefficient (float): Coefficient d'absorption en mm⁻¹.
        thickness_mm (float): Épaisseur du matériau en mm.
    
    Sources:
        - "Handbook of Optics" by W. G. Driscoll (1978), Vol. I, Ch. 2
    """
    
    name: str = "Unknown"
    material_type: MaterialType = MaterialType.DIELECTRIC
    refractive_index: float = 1.5
    reference_wavelength_nm: float = DEFAULT_WAVELENGTH_NM
    dispersion_model: DispersionModel = DispersionModel.CONSTANT
    dispersion_coefficients: Optional[Dict] = None
    absorption_coefficient: float = 0.0
    thickness_mm: float = 1.0

    def __post_init__(self):
        """FR: Initialisation après création."""
        if self.dispersion_coefficients is None:
            self.dispersion_coefficients = {}

    def get_refractive_index(self, wavelength_nm: float) -> float:
        """
        FR: Retourne l'indice de réfraction à une longueur d'onde donnée.
            
        EN: Returns the refractive index at a given wavelength.
        
        Args:
            wavelength_nm (float): Longueur d'onde en nm.
        
        Returns:
            float: Indice de réfraction.
        
        Sources:
            - "Handbook of Optics" by W. G. Driscoll (1978), Vol. I, Ch. 2
        """
        if self.dispersion_model == DispersionModel.CONSTANT:
            return self.refractive_index
        
        elif self.dispersion_model == DispersionModel.SELLMEIER:
            return self._sellmeier_formula(wavelength_nm)
        
        elif self.dispersion_model == DispersionModel.CAUCHY:
            return self._cauchy_formula(wavelength_nm)
        
        elif self.dispersion_model == DispersionModel.HERZBERGER:
            return self._herzberger_formula(wavelength_nm)
        
        else:
            return self.refractive_index

    def _sellmeier_formula(self, wavelength_nm: float) -> float:
        """
        FR: Formule de Sellmeier.
            
            n² = 1 + B1λ²/(λ² - C1) + B2λ²/(λ² - C2) + B3λ²/(λ² - C3)
            où λ est en µm.
            
        EN: Sellmeier formula.
            
            n² = 1 + B1λ²/(λ² - C1) + B2λ²/(λ² - C2) + B3λ²/(λ² - C3)
            where λ is in µm.
        
        Args:
            wavelength_nm (float): Longueur d'onde en nm.
        
        Returns:
            float: Indice de réfraction.
        
        Sources:
            - "Handbook of Optics" by W. G. Driscoll (1978), Vol. I, Ch. 2
        """
        wavelength_um = wavelength_nm / 1000.0
        
        n_squared = 1.0
        for i in range(1, 4):
            B = self.dispersion_coefficients.get(f'B{i}', 0.0)
            C = self.dispersion_coefficients.get(f'C{i}', 0.0)
            n_squared += B * (wavelength_um**2) / (wavelength_um**2 - C)
        
        return np.sqrt(n_squared)

    def _cauchy_formula(self, wavelength_nm: float) -> float:
        """
        FR: Formule de Cauchy.
            
            n = A + B/λ² + C/λ⁴
            où λ est en µm.
            
        EN: Cauchy formula.
            
            n = A + B/λ² + C/λ⁴
            where λ is in µm.
        
        Args:
            wavelength_nm (float): Longueur d'onde en nm.
        
        Returns:
            float: Indice de réfraction.
        
        Sources:
            - "Handbook of Optics" by W. G. Driscoll (1978), Vol. I, Ch. 2
        """
        wavelength_um = wavelength_nm / 1000.0
        
        A = self.dispersion_coefficients.get('A', self.refractive_index)
        B = self.dispersion_coefficients.get('B', 0.0)
        C = self.dispersion_coefficients.get('C', 0.0)
        
        return A + B / (wavelength_um**2) + C / (wavelength_um**4)

    def _herzberger_formula(self, wavelength_nm: float) -> float:
        """
        FR: Formule de Herzberger.
            
        EN: Herzberger formula.
        
        Args:
            wavelength_nm (float): Longueur d'onde en nm.
        
        Returns:
            float: Indice de réfraction.
        
        Sources:
            - "Handbook of Optics" by W. G. Driscoll (1978), Vol. I, Ch. 2
        """
        # Pour simplifier, utiliser la formule de Cauchy
        return self._cauchy_formula(wavelength_nm)

    def get_transmission(self, wavelength_nm: float) -> float:
        """
        FR: Calcule la transmission à une longueur d'onde donnée.
            
            La transmission dépend de :
            - La réflexion aux interfaces
            - L'absorption dans le matériau
            
        EN: Calculates the transmission at a given wavelength.
            
            Transmission depends on:
            - Reflection at interfaces
            - Absorption in the material
        
        Args:
            wavelength_nm (float): Longueur d'onde en nm.
        
        Returns:
            float: Transmission (0-1).
        
        Sources:
            - "Handbook of Optics" by W. G. Driscoll (1978), Vol. I, Ch. 2
        """
        # Obtenir l'indice de réfraction
        n = self.get_refractive_index(wavelength_nm)
        
        # Calculer la réflexion à une interface (air-matériau)
        # R = ((n1 - n2) / (n1 + n2))²
        n_air = 1.0
        R_single = ((n - n_air) / (n + n_air))**2
        
        # Pour deux interfaces (air-matériau-air), la transmission est :
        # T = (1 - R)² * exp(-α * d)
        # où α est le coefficient d'absorption et d est l'épaisseur
        T_reflection = (1 - R_single)**2
        T_absorption = np.exp(-self.absorption_coefficient * self.thickness_mm)
        
        return T_reflection * T_absorption

    def get_reflection(self, wavelength_nm: float) -> float:
        """
        FR: Calcule la réflexion à une longueur d'onde donnée.
            
        EN: Calculates the reflection at a given wavelength.
        
        Args:
            wavelength_nm (float): Longueur d'onde en nm.
        
        Returns:
            float: Réflexion (0-1).
        """
        n = self.get_refractive_index(wavelength_nm)
        n_air = 1.0
        R_single = ((n - n_air) / (n + n_air))**2
        
        # Pour deux interfaces, la réflexion totale est :
        # R_total = 2R / (1 + R)
        # (approximation pour des interfaces parallèles)
        return 2 * R_single / (1 + R_single)


# =============================================================================
# CLASSE: MATERIAL DATABASE
# =============================================================================

class MaterialDatabase:
    """
    FR: Base de données de matériaux optiques.
        
    EN: Optical material database.
    
    Sources:
        - "Handbook of Optics" by W. G. Driscoll (1978)
    """

    def __init__(self):
        """FR: Initialise la base de données. EN: Initializes the database."""
        self.materials = {}
        self._initialize_database()

    def _initialize_database(self) -> None:
        """FR: Initialise la base de données avec des matériaux courants. EN: Initializes the database with common materials."""
        # Fused Silica
        self.add_material(
            name="Fused_Silica",
            material_type=MaterialType.DIELECTRIC,
            refractive_index=1.4585,
            reference_wavelength_nm=633.0,
            dispersion_model=DispersionModel.SELLMEIER,
            dispersion_coefficients=SELLMEIER_COEFFICIENTS['Fused_Silica'],
            absorption_coefficient=0.001  # Très faible absorption dans le visible
        )
        
        # BK7
        self.add_material(
            name="BK7",
            material_type=MaterialType.DIELECTRIC,
            refractive_index=1.5168,
            reference_wavelength_nm=633.0,
            dispersion_model=DispersionModel.SELLMEIER,
            dispersion_coefficients=SELLMEIER_COEFFICIENTS['BK7'],
            absorption_coefficient=0.001
        )
        
        # CaF2
        self.add_material(
            name="CaF2",
            material_type=MaterialType.DIELECTRIC,
            refractive_index=1.4338,
            reference_wavelength_nm=633.0,
            dispersion_model=DispersionModel.SELLMEIER,
            dispersion_coefficients=SELLMEIER_COEFFICIENTS['CaF2'],
            absorption_coefficient=0.0001
        )
        
        # Sapphire
        self.add_material(
            name="Sapphire",
            material_type=MaterialType.DIELECTRIC,
            refractive_index=1.768,
            reference_wavelength_nm=633.0,
            dispersion_model=DispersionModel.SELLMEIER,
            dispersion_coefficients=SELLMEIER_COEFFICIENTS['Sapphire'],
            absorption_coefficient=0.001
        )
        
        # Air
        self.add_material(
            name="air",
            material_type=MaterialType.GAS,
            refractive_index=1.000293,
            reference_wavelength_nm=633.0,
            dispersion_model=DispersionModel.CONSTANT,
            absorption_coefficient=0.0
        )
        
        # Silicon
        self.add_material(
            name="Silicon",
            material_type=MaterialType.SEMICONDUCTOR,
            refractive_index=3.5,
            reference_wavelength_nm=1550.0,  # Infrarouge
            dispersion_model=DispersionModel.CONSTANT,
            absorption_coefficient=0.1  # Absorption significative dans le visible
        )

    def add_material(self, **kwargs) -> None:
        """FR: Ajoute un matériau à la base de données. EN: Adds a material to the database."""
        material = OpticalMaterial(**kwargs)
        self.materials[material.name] = material

    def get_material(self, name: str) -> Optional[OpticalMaterial]:
        """FR: Retourne un matériau par son nom. EN: Returns a material by name."""
        return self.materials.get(name)

    def get_refractive_index(self, name: str, wavelength_nm: float) -> float:
        """FR: Retourne l'indice de réfraction d'un matériau à une longueur d'onde. EN: Returns the refractive index of a material at a wavelength."""
        material = self.get_material(name)
        if material is not None:
            return material.get_refractive_index(wavelength_nm)
        else:
            logger.warning(f"Matériau '{name}' non trouvé. Retourne 1.5.")
            return 1.5


# =============================================================================
# FONCTIONS DE CRÉATION
# =============================================================================

def create_material(name: str = "Unknown",
                    material_type: MaterialType = MaterialType.DIELECTRIC,
                    refractive_index: float = 1.5,
                    reference_wavelength_nm: float = DEFAULT_WAVELENGTH_NM,
                    dispersion_model: DispersionModel = DispersionModel.CONSTANT,
                    dispersion_coefficients: Optional[Dict] = None,
                    absorption_coefficient: float = 0.0,
                    thickness_mm: float = 1.0) -> OpticalMaterial:
    """
    FR: Crée un matériau optique.
        
    EN: Creates an optical material.
    
    Sources:
        - "Handbook of Optics" by W. G. Driscoll (1978)
    """
    return OpticalMaterial(
        name=name,
        material_type=material_type,
        refractive_index=refractive_index,
        reference_wavelength_nm=reference_wavelength_nm,
        dispersion_model=dispersion_model,
        dispersion_coefficients=dispersion_coefficients,
        absorption_coefficient=absorption_coefficient,
        thickness_mm=thickness_mm
    )


def create_material_database() -> MaterialDatabase:
    """
    FR: Crée une base de données de matériaux.
        
    EN: Creates a material database.
    
    Sources:
        - "Handbook of Optics" by W. G. Driscoll (1978)
    """
    return MaterialDatabase()


# =============================================================================
# TESTS UNITAIRES
# =============================================================================

class TestMaterialBehaviour:
    """FR: Tests unitaires pour Material_Behaviour.py."""

    def test_material_creation(self):
        """FR: Test la création d'un matériau."""
        material = create_material(
            name="TestMaterial",
            refractive_index=1.5
        )
        assert material.name == "TestMaterial"
        assert material.refractive_index == 1.5

    def test_get_refractive_index(self):
        """FR: Test l'obtention de l'indice de réfraction."""
        material = create_material(
            name="TestMaterial",
            refractive_index=1.5,
            dispersion_model=DispersionModel.CONSTANT
        )
        n = material.get_refractive_index(633.0)
        assert n == 1.5

    def test_sellmeier_formula(self):
        """FR: Test la formule de Sellmeier."""
        material = create_material(
            name="Fused_Silica",
            refractive_index=1.4585,
            dispersion_model=DispersionModel.SELLMEIER,
            dispersion_coefficients=SELLMEIER_COEFFICIENTS['Fused_Silica']
        )
        n = material.get_refractive_index(633.0)
        assert 1.45 < n < 1.46

    def test_transmission(self):
        """FR: Test le calcul de la transmission."""
        material = create_material(
            name="TestMaterial",
            refractive_index=1.5,
            absorption_coefficient=0.01,
            thickness_mm=1.0
        )
        T = material.get_transmission(633.0)
        assert 0 <= T <= 1

    def test_material_database(self):
        """FR: Test la base de données de matériaux."""
        db = create_material_database()
        material = db.get_material("Fused_Silica")
        assert material is not None
        assert material.name == "Fused_Silica"


if __name__ == "__main__":
    import unittest
    unittest.main()

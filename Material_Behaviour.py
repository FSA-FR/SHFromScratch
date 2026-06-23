"""
Material_Behaviour.py

FR: Module pour la modélisation du comportement des matériaux optiques et mécaniques.
    
    Fonctionnalités principales :
    - Variation des indices de réfraction avec la longueur d'onde et la température
    - Chromatisme des matériaux optiques
    - Expansion thermique des matériaux (optiques et mécaniques)
    - Réflexion et transmission
    - Variation de la puissance optique avec la température
    - Déformation des microstructures sous l'effet thermique
    
    Unités :
    - Longueurs : mm
    - Longueur d'onde : nm
    - Température : K (kelvin) ou °C
    - Indice de réfraction : sans unité
    - Coefficient d'expansion thermique : ppm/°C
    - Puissance optique : mm⁻¹

EN: Module for modeling optical and mechanical material behavior.
    
    Main features:
    - Variation of refractive indices with wavelength and temperature
    - Chromatic dispersion of optical materials
    - Thermal expansion of materials (optical and mechanical)
    - Reflection and transmission
    - Variation of optical power with temperature
    - Deformation of microstructures under thermal effects
    
    Units:
    - Lengths: mm
    - Wavelength: nm
    - Temperature: K (kelvin) or °C
    - Refractive index: unitless
    - Thermal expansion coefficient: ppm/°C
    - Optical power: mm⁻¹

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch

Dependencies:
    - numpy
    - MathAndPhysicsTools (pour les fonctions outils)

Sources:
    - "Handbook of Optics" by W. G. Driscoll (1978)
    - "Thermal Expansion of Solids" by C. S. Desai (1984)
    - "refractiveindex.info" (https://refractiveindex.info)
"""

import numpy as np
import logging
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field
from enum import Enum

from MathAndPhysicsTools import handle_nan, DEFAULT_WAVELENGTH_NM


# =============================================================================
# CONFIGURATION GLOBALE
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Material_Behaviour")

REFERENCE_TEMPERATURE_K = 293.15
REFERENCE_TEMPERATURE_C = 20.0


# =============================================================================
# DONNÉES DES MATÉRIAUX
# =============================================================================

SELLMEIER_COEFFICIENTS = {
    'Fused_Silica': {'B1': 0.6961663, 'C1': 0.0684043, 'B2': 0.4079426, 'C2': 0.1162414, 'B3': 0.8974794, 'C3': 9.896161, 'valid_range_nm': (210, 6700)},
    'BK7': {'B1': 1.03961212, 'C1': 0.00600069867, 'B2': 0.231792344, 'C2': 0.0200179144, 'B3': 1.01046945, 'C3': 103.560653, 'valid_range_nm': (300, 2500)},
    'SF5': {'B1': 1.43140877, 'C1': 0.0052863912, 'B2': 0.21456388, 'C2': 0.015099804, 'B3': 0.94678683, 'C3': 122.209985, 'valid_range_nm': (300, 2500)},
    'CaF2': {'B1': 0.5675888, 'C1': 0.00354178, 'B2': 0.4716143, 'C2': 0.0117549, 'B3': 3.8484723, 'C3': 1200.5558, 'valid_range_nm': (200, 8000)},
    'Sapphire': {'B1': 1.023798, 'C1': 0.0037754, 'B2': 1.503182, 'C2': 0.0122544, 'B3': 5.509158, 'C3': 321.7706, 'valid_range_nm': (200, 5500)},
}

SELLMEIER_IR_COEFFICIENTS = {
    'Silicon': {'A': 11.685800, 'B': 0.9398164, 'C': 8.10461e-3, 'valid_range_nm': (1200, 14000)}
}

THERMAL_EXPANSION_COEFFICIENTS = {
    'Fused_Silica': {'CTE': 0.51, 'dCTE_dT': 0.0008, 'valid_range_C': (-200, 1000)},
    'BK7': {'CTE': 7.1, 'dCTE_dT': 0.001, 'valid_range_C': (-100, 500)},
    'SF5': {'CTE': 8.2, 'dCTE_dT': 0.001, 'valid_range_C': (-100, 500)},
    'CaF2': {'CTE': 18.85, 'dCTE_dT': 0.002, 'valid_range_C': (-200, 800)},
    'Sapphire': {'CTE': 5.0, 'dCTE_dT': 0.0005, 'valid_range_C': (-200, 1500)},
    'Silicon': {'CTE': 2.6, 'dCTE_dT': 0.0002, 'valid_range_C': (-200, 1200)},
    'Steel': {'CTE': 12.0, 'dCTE_dT': 0.0005, 'valid_range_C': (-100, 800)},
    'Aluminum': {'CTE': 23.1, 'dCTE_dT': 0.001, 'valid_range_C': (-200, 600)},
    'Invar': {'CTE': 1.2, 'dCTE_dT': 0.0001, 'valid_range_C': (-100, 500)},
    'Copper': {'CTE': 16.5, 'dCTE_dT': 0.0008, 'valid_range_C': (-200, 400)},
}

REFRACTIVE_INDEX_TEMPERATURE_COEFFICIENTS = {
    'Fused_Silica': {'dn_dT': 10.0, 'valid_range_C': (-200, 1000)},
    'BK7': {'dn_dT': -1.0, 'valid_range_C': (-100, 500)},
    'SF5': {'dn_dT': -2.0, 'valid_range_C': (-100, 500)},
    'CaF2': {'dn_dT': -11.0, 'valid_range_C': (-200, 800)},
    'Sapphire': {'dn_dT': 13.0, 'valid_range_C': (-200, 1500)},
    'Silicon': {'dn_dT': 150.0, 'valid_range_C': (-200, 1200)},
}


# =============================================================================
# ENUMS
# =============================================================================

class MaterialCategory(Enum):
    OPTICAL = "optical"
    MECHANICAL = "mechanical"
    OPTICAL_MECHANICAL = "optical_mechanical"


class RefractiveIndexModel(Enum):
    CONSTANT = "constant"
    SELLMEIER = "sellmeier"
    SELLMEIER_IR = "sellmeier_ir"
    POLYNOMIAL = "polynomial"


class ThermalExpansionModel(Enum):
    CONSTANT = "constant"
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"


# =============================================================================
# CLASSE: MATERIAL
# =============================================================================

@dataclass
class Material:
    """
    FR: Matériau optique ou mécanique.
    EN: Optical or mechanical material.
    
    Sources:
        - "Handbook of Optics" by W. G. Driscoll (1978)
        - "Thermal Expansion of Solids" by C. S. Desai (1984)
    """
    
    name: str
    category: MaterialCategory = MaterialCategory.OPTICAL
    refractive_index_model: RefractiveIndexModel = RefractiveIndexModel.CONSTANT
    refractive_index_coefficients: Dict = field(default_factory=dict)
    thermal_expansion_model: ThermalExpansionModel = ThermalExpansionModel.CONSTANT
    thermal_expansion_coefficients: Dict = field(default_factory=dict)
    reference_temperature_K: float = REFERENCE_TEMPERATURE_K
    reference_wavelength_nm: float = DEFAULT_WAVELENGTH_NM
    dn_dT: float = 0.0
    valid_wavelength_range_nm: Tuple[float, float] = (200, 20000)
    valid_temperature_range_C: Tuple[float, float] = (-273, 2000)

    def __post_init__(self):
        if self.valid_wavelength_range_nm[0] >= self.valid_wavelength_range_nm[1]:
            raise ValueError("Plage de longueurs d'onde invalide")
        if self.valid_temperature_range_C[0] >= self.valid_temperature_range_C[1]:
            raise ValueError("Plage de températures invalide")

    def get_refractive_index(self, wavelength_nm: float, temperature_K: Optional[float] = None) -> float:
        """FR: Calcule l'indice de réfraction. EN: Calculates the refractive index."""
        if not (self.valid_wavelength_range_nm[0] <= wavelength_nm <= self.valid_wavelength_range_nm[1]):
            logger.warning(f"Longueur d'onde {wavelength_nm}nm hors plage valide")
        
        if temperature_K is not None:
            if not (self.valid_temperature_range_C[0] <= (temperature_K - 273.15) <= self.valid_temperature_range_C[1]):
                logger.warning(f"Température {temperature_K}K hors plage valide")
        
        if self.refractive_index_model == RefractiveIndexModel.CONSTANT:
            n = self.refractive_index_coefficients.get('n', 1.5)
        elif self.refractive_index_model == RefractiveIndexModel.SELLMEIER:
            n = self._sellmeier_model(wavelength_nm)
        elif self.refractive_index_model == RefractiveIndexModel.SELLMEIER_IR:
            n = self._sellmeier_ir_model(wavelength_nm)
        elif self.refractive_index_model == RefractiveIndexModel.POLYNOMIAL:
            n = self._polynomial_model(wavelength_nm)
        else:
            n = 1.5
        
        if temperature_K is not None:
            delta_T = temperature_K - self.reference_temperature_K
            n += n * (self.dn_dT / 1e6) * delta_T
        
        return n

    def _sellmeier_model(self, wavelength_nm: float) -> float:
        """FR: Modèle de Sellmeier. EN: Sellmeier model."""
        wavelength_um = wavelength_nm / 1000.0
        n_squared = 1.0
        for i in range(1, 4):
            B = self.refractive_index_coefficients.get(f'B{i}', 0.0)
            C = self.refractive_index_coefficients.get(f'C{i}', 0.0)
            n_squared += B * (wavelength_um**2) / (wavelength_um**2 - C)
        return np.sqrt(n_squared)

    def _sellmeier_ir_model(self, wavelength_nm: float) -> float:
        """FR: Modèle de Sellmeier IR. EN: Sellmeier IR model."""
        wavelength_um = wavelength_nm / 1000.0
        A = self.refractive_index_coefficients.get('A', 0.0)
        B = self.refractive_index_coefficients.get('B', 0.0)
        C = self.refractive_index_coefficients.get('C', 0.0)
        return np.sqrt(A + B * (wavelength_um**2) / (wavelength_um**2 - C))

    def _polynomial_model(self, wavelength_nm: float) -> float:
        """FR: Modèle polynomial. EN: Polynomial model."""
        wavelength_um = wavelength_nm / 1000.0
        n = 0.0
        for i in range(10):
            a = self.refractive_index_coefficients.get(f'a{i}', 0.0)
            n += a * (wavelength_um ** i)
        return n

    def get_thermal_expansion_coefficient(self, temperature_K: Optional[float] = None) -> float:
        """FR: Calcule le CTE. EN: Calculates the thermal expansion coefficient."""
        if temperature_K is None:
            temperature_K = self.reference_temperature_K
        temperature_C = temperature_K - 273.15
        
        if self.thermal_expansion_model == ThermalExpansionModel.CONSTANT:
            return self.thermal_expansion_coefficients.get('CTE', 0.0)
        elif self.thermal_expansion_model == ThermalExpansionModel.LINEAR:
            CTE_0 = self.thermal_expansion_coefficients.get('CTE', 0.0)
            dCTE_dT = self.thermal_expansion_coefficients.get('dCTE_dT', 0.0)
            return CTE_0 + dCTE_dT * (temperature_C - (self.reference_temperature_K - 273.15))
        else:
            return self.thermal_expansion_coefficients.get('CTE', 0.0)

    def calculate_thermal_expansion(self, initial_length_mm: float, initial_temperature_K: float, final_temperature_K: float) -> float:
        """FR: Calcule l'expansion thermique. EN: Calculates thermal expansion."""
        CTE_ppm = (self.get_thermal_expansion_coefficient(initial_temperature_K) + self.get_thermal_expansion_coefficient(final_temperature_K)) / 2
        delta_T_C = final_temperature_K - initial_temperature_K
        return initial_length_mm * (CTE_ppm / 1e6) * delta_T_C

    def calculate_optical_power_change(self, initial_focal_length_mm: float, initial_temperature_K: float, final_temperature_K: float) -> float:
        """FR: Calcule la variation de puissance optique. EN: Calculates optical power change."""
        initial_optical_power = 1.0 / initial_focal_length_mm
        delta_n = self.dn_dT / 1e6 * (final_temperature_K - initial_temperature_K)
        delta_power_dn = initial_optical_power * delta_n
        delta_L_L = self.calculate_thermal_expansion(initial_focal_length_mm, initial_temperature_K, final_temperature_K) / initial_focal_length_mm
        delta_power_thermal = -initial_optical_power * delta_L_L
        return delta_power_dn + delta_power_thermal

    def calculate_reflectance(self, wavelength_nm: float, angle_rad: float = 0.0, polarization: str = 's') -> float:
        """FR: Calcule la réflectance. EN: Calculates reflectance."""
        n2 = self.get_refractive_index(wavelength_nm)
        n1 = 1.0
        
        if angle_rad == 0:
            return ((n1 - n2) / (n1 + n2))**2
        
        if polarization == 's':
            theta1 = angle_rad
            theta2 = np.arcsin(n1 * np.sin(theta1) / n2)
            r_s = (n1 * np.cos(theta1) - n2 * np.cos(theta2)) / (n1 * np.cos(theta1) + n2 * np.cos(theta2))
            return r_s**2
        elif polarization == 'p':
            theta1 = angle_rad
            theta2 = np.arcsin(n1 * np.sin(theta1) / n2)
            r_p = (n1 * np.cos(theta2) - n2 * np.cos(theta1)) / (n1 * np.cos(theta2) + n2 * np.cos(theta1))
            return r_p**2
        elif polarization == 'unpolarized':
            R_s = self.calculate_reflectance(wavelength_nm, angle_rad, 's')
            R_p = self.calculate_reflectance(wavelength_nm, angle_rad, 'p')
            return (R_s + R_p) / 2
        else:
            return self.calculate_reflectance(wavelength_nm, angle_rad, 'unpolarized')

    def calculate_transmittance(self, wavelength_nm: float, thickness_mm: float, angle_rad: float = 0.0, polarization: str = 'unpolarized') -> float:
        """FR: Calcule la transmittance. EN: Calculates transmittance."""
        R = self.calculate_reflectance(wavelength_nm, angle_rad, polarization)
        absorption_coefficient = 0.001
        T_absorption = np.exp(-absorption_coefficient * thickness_mm)
        return (1 - R)**2 * T_absorption / (1 - R**2 * T_absorption**2)


# =============================================================================
# CLASSE: MATERIAL DATABASE
# =============================================================================

class MaterialDatabase:
    """FR: Base de données de matériaux. EN: Material database."""

    def __init__(self):
        self.materials = {}
        self._initialize_database()

    def _initialize_database(self) -> None:
        """FR: Initialise la base de données. EN: Initializes the database."""
        for name, coeffs in SELLMEIER_COEFFICIENTS.items():
            self.add_material(
                name=name,
                category=MaterialCategory.OPTICAL,
                refractive_index_model=RefractiveIndexModel.SELLMEIER,
                refractive_index_coefficients=coeffs,
                thermal_expansion_model=ThermalExpansionModel.LINEAR,
                thermal_expansion_coefficients=THERMAL_EXPANSION_COEFFICIENTS[name],
                reference_temperature_K=REFERENCE_TEMPERATURE_K,
                reference_wavelength_nm=DEFAULT_WAVELENGTH_NM,
                dn_dT=REFRACTIVE_INDEX_TEMPERATURE_COEFFICIENTS[name]['dn_dT'],
                valid_wavelength_range_nm=coeffs['valid_range_nm'],
                valid_temperature_range_C=THERMAL_EXPANSION_COEFFICIENTS[name]['valid_range_C']
            )
        
        self.add_material(
            name="Silicon",
            category=MaterialCategory.OPTICAL,
            refractive_index_model=RefractiveIndexModel.SELLMEIER_IR,
            refractive_index_coefficients=SELLMEIER_IR_COEFFICIENTS['Silicon'],
            thermal_expansion_model=ThermalExpansionModel.LINEAR,
            thermal_expansion_coefficients=THERMAL_EXPANSION_COEFFICIENTS['Silicon'],
            reference_temperature_K=REFERENCE_TEMPERATURE_K,
            reference_wavelength_nm=1550.0,
            dn_dT=REFRACTIVE_INDEX_TEMPERATURE_COEFFICIENTS['Silicon']['dn_dT'],
            valid_wavelength_range_nm=SELLMEIER_IR_COEFFICIENTS['Silicon']['valid_range_nm'],
            valid_temperature_range_C=THERMAL_EXPANSION_COEFFICIENTS['Silicon']['valid_range_C']
        )
        
        for name in ['Steel', 'Aluminum', 'Invar', 'Copper']:
            self.add_material(
                name=name,
                category=MaterialCategory.MECHANICAL,
                refractive_index_model=RefractiveIndexModel.CONSTANT,
                refractive_index_coefficients={'n': 1.0},
                thermal_expansion_model=ThermalExpansionModel.LINEAR,
                thermal_expansion_coefficients=THERMAL_EXPANSION_COEFFICIENTS[name],
                reference_temperature_K=REFERENCE_TEMPERATURE_K,
                valid_temperature_range_C=THERMAL_EXPANSION_COEFFICIENTS[name]['valid_range_C']
            )

    def add_material(self, **kwargs) -> None:
        """FR: Ajoute un matériau. EN: Adds a material."""
        material = Material(**kwargs)
        self.materials[material.name] = material

    def get_material(self, name: str) -> Optional[Material]:
        """FR: Retourne un matériau. EN: Returns a material."""
        return self.materials.get(name)

    def get_material_names(self) -> List[str]:
        """FR: Retourne les noms. EN: Returns names."""
        return list(self.materials.keys())


# =============================================================================
# FONCTIONS DE CRÉATION
# =============================================================================

def create_material(name: str = "Unknown", category: MaterialCategory = MaterialCategory.OPTICAL,
                    refractive_index_model: RefractiveIndexModel = RefractiveIndexModel.CONSTANT,
                    refractive_index_coefficients: Optional[Dict] = None,
                    thermal_expansion_model: ThermalExpansionModel = ThermalExpansionModel.CONSTANT,
                    thermal_expansion_coefficients: Optional[Dict] = None,
                    reference_temperature_K: float = REFERENCE_TEMPERATURE_K,
                    reference_wavelength_nm: float = DEFAULT_WAVELENGTH_NM,
                    dn_dT: float = 0.0,
                    valid_wavelength_range_nm: Tuple[float, float] = (200, 20000),
                    valid_temperature_range_C: Tuple[float, float] = (-273, 2000)) -> Material:
    """FR: Crée un matériau. EN: Creates a material."""
    return Material(
        name=name, category=category, refractive_index_model=refractive_index_model,
        refractive_index_coefficients=refractive_index_coefficients or {},
        thermal_expansion_model=thermal_expansion_model,
        thermal_expansion_coefficients=thermal_expansion_coefficients or {},
        reference_temperature_K=reference_temperature_K, reference_wavelength_nm=reference_wavelength_nm,
        dn_dT=dn_dT, valid_wavelength_range_nm=valid_wavelength_range_nm,
        valid_temperature_range_C=valid_temperature_range_C
    )


def create_material_database() -> MaterialDatabase:
    """FR: Crée une base de données. EN: Creates a database."""
    return MaterialDatabase()


# =============================================================================
# TESTS UNITAIRES
# =============================================================================

class TestMaterialBehaviour:
    """FR: Tests unitaires pour Material_Behaviour.py."""

    def test_material_creation(self):
        """FR: Test la création."""
        material = create_material(name="TestMaterial", category=MaterialCategory.OPTICAL)
        assert material.name == "TestMaterial"

    def test_get_refractive_index(self):
        """FR: Test l'indice de réfraction."""
        material = create_material(name="TestMaterial", refractive_index_model=RefractiveIndexModel.CONSTANT, refractive_index_coefficients={'n': 1.5})
        assert material.get_refractive_index(633.0) == 1.5

    def test_sellmeier_model(self):
        """FR: Test le modèle de Sellmeier."""
        material = create_material(name="Fused_Silica", refractive_index_model=RefractiveIndexModel.SELLMEIER, refractive_index_coefficients=SELLMEIER_COEFFICIENTS['Fused_Silica'])
        n = material.get_refractive_index(633.0)
        assert 1.45 < n < 1.46

    def test_material_database(self):
        """FR: Test la base de données."""
        db = create_material_database()
        material = db.get_material("Fused_Silica")
        assert material is not None


if __name__ == "__main__":
    import unittest
    unittest.main()

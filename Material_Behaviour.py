"""
Material_Behaviour.py
FR: Module pour la gestion du comportement des matériaux optiques et mécaniques.
    Permet de calculer :
    - L'indice de réfraction en fonction de la longueur d'onde et de la température.
    - La dilatation/contraction thermique des matériaux.
    - La réflectance et la transmittance en fonction de la longueur d'onde, de l'épaisseur et de la polarisation.
    - La variation de puissance optique pour les optiques réfractives.
    
    Matériaux optiques principaux : Fused_Silica, BK7, SF5, Silicium.
    Matériaux mécaniques : Acier, Aluminium.

EN: Module for managing the behavior of optical and mechanical materials.
    Allows calculating:
    - Refractive index as a function of wavelength and temperature.
    - Thermal expansion/contraction of materials.
    - Reflectance and transmittance as a function of wavelength, thickness, and polarization.
    - Optical power variation for refractive optics.
    
    Main optical materials: Fused_Silica, BK7, SF5, Silicon.
    Mechanical materials: Steel, Aluminum.

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch

Sources:
    - Refractive index data from https://refractiveindex.info
    - Thermal expansion coefficients from various scientific publications
"""

import numpy as np
import logging
from typing import Optional, Tuple, Dict, Union, List
from dataclasses import dataclass, field
from enum import Enum
import requests
import json
from scipy.interpolate import interp1d


# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Material_Behaviour")


# =============================================================================
# 1. ENUMS ET CONSTANTES / ENUMS AND CONSTANTS
# =============================================================================

class MaterialType(Enum):
    """FR: Type de matériau (optique ou mécanique)."""
    OPTICAL = "optical"
    MECHANICAL = "mechanical"


class Polarization(Enum):
    """FR: Polarisation de la lumière."""
    NONE = "none"      # Pas de polarisation (par défaut)
    S = "s"           # Polarisation s (perpendiculaire au plan d'incidence)
    P = "p"           # Polarisation p (parallèle au plan d'incidence)
    CIRCULAR = "circular"  # Polarisation circulaire
    ELLIPTICAL = "elliptical"  # Polarisation elliptique


# Constantes physiques
C_LIGHT_M_S = 299792458  # Vitesse de la lumière en m/s
K_TO_C = 273.15  # Conversion Kelvin → Celsius
STANDARD_TEMPERATURE_K = 293.15  # Température standard (20°C)


# =============================================================================
# 2. BASE DE DONNÉES DES MATÉRIAUX / MATERIAL DATABASE
# =============================================================================

@dataclass
class RefractiveIndexModel:
    """
    FR: Modèle pour l'indice de réfraction d'un matériau.
        Stocke les coefficients pour différents modèles (Sellmeier, Cauchy, etc.).

    EN: Model for the refractive index of a material.
        Stores coefficients for different models (Sellmeier, Cauchy, etc.).

    Attributes:
        model_type (str): Type de modèle ("Sellmeier", "Cauchy", "Polynomial", "Tabulated").
        coefficients (Dict): Coefficients du modèle.
        wavelength_range_nm (Tuple[float, float]): Plage de validité en nm.
        temperature_K (float): Température de référence en Kelvin.
        source (str): Source des données.
    """
    model_type: str
    coefficients: Dict[str, float]
    wavelength_range_nm: Tuple[float, float]
    temperature_K: float = STANDARD_TEMPERATURE_K
    source: str = ""


@dataclass
class ThermalExpansionModel:
    """
    FR: Modèle pour l'expansion thermique d'un matériau.
        Stocke les coefficients du CTE (Coefficient of Thermal Expansion).

    EN: Model for the thermal expansion of a material.
        Stores CTE (Coefficient of Thermal Expansion) coefficients.

    Attributes:
        cte_model (str): Type de modèle ("constant", "linear", "polynomial").
        coefficients (Dict): Coefficients du modèle.
        temperature_range_K (Tuple[float, float]): Plage de validité en Kelvin.
        source (str): Source des données.
    """
    cte_model: str
    coefficients: Dict[str, float]
    temperature_range_K: Tuple[float, float]
    source: str = ""


@dataclass
class OpticalProperties:
    """
    FR: Propriétés optiques d'un matériau.

    EN: Optical properties of a material.

    Attributes:
        refractive_index_model (RefractiveIndexModel): Modèle de l'indice de réfraction.
        thermal_expansion_model (ThermalExpansionModel): Modèle d'expansion thermique.
        absorption_coefficient (float): Coefficient d'absorption en m⁻¹ (à une longueur d'onde de référence).
        reference_wavelength_nm (float): Longueur d'onde de référence pour l'absorption.
    """
    refractive_index_model: RefractiveIndexModel
    thermal_expansion_model: ThermalExpansionModel
    absorption_coefficient: float = 0.0
    reference_wavelength_nm: float = 633.0


@dataclass
class Material:
    """
    FR: Classe représentant un matériau (optique ou mécanique).
        Contient toutes les propriétés nécessaires pour simuler son comportement.

    EN: Class representing a material (optical or mechanical).
        Contains all properties needed to simulate its behavior.

    Attributes:
        name (str): Nom du matériau.
        material_type (MaterialType): Type de matériau (optique ou mécanique).
        optical_properties (Optional[OpticalProperties]): Propriétés optiques (si applicable).
        thermal_expansion_model (ThermalExpansionModel): Modèle d'expansion thermique.
        density_kg_m3 (float): Masse volumique en kg/m³.
        young_modulus_Pa (float): Module de Young en Pascals.
        poisson_ratio (float): Coefficient de Poisson.
        thermal_conductivity_W_mK (float): Conductivité thermique en W/(m·K).
        specific_heat_J_kgK (float): Chaleur spécifique en J/(kg·K).
    """
    name: str
    material_type: MaterialType
    optical_properties: Optional[OpticalProperties] = None
    thermal_expansion_model: ThermalExpansionModel = field(default_factory=ThermalExpansionModel)
    density_kg_m3: float = 0.0
    young_modulus_Pa: float = 0.0
    poisson_ratio: float = 0.0
    thermal_conductivity_W_mK: float = 0.0
    specific_heat_J_kgK: float = 0.0


# =============================================================================
# 3. BASE DE DONNÉES DES MATÉRIAUX PRÉDÉFINIS / PREDEFINED MATERIAL DATABASE
# =============================================================================

# --- Données des indices de réfraction (depuis refractiveindex.info) ---
# Format : {"matériau": {"modèle": ..., "coefficients": {...}, "plage_nm": [...], "source": ...}}
REFRACTIVE_INDEX_DATA = {
    # Fused Silica (SiO₂) - Malacara (2007) - 0.21-6.7 µm
    "Fused_Silica": {
        "Sellmeier": {
            "coefficients": {
                "B1": 0.6961663,
                "B2": 0.4079426,
                "B3": 0.8974794,
                "C1": 0.0684043**2,  # µm² → nm²
                "C2": 0.1162414**2,
                "C3": 9.896161**2,
            },
            "wavelength_range_nm": (210.0, 6700.0),
            "temperature_K": STANDARD_TEMPERATURE_K,
            "source": "Malacara, Optical Shop Testing, Wiley (2007)",
        },
        # Modèle alternatif : Sellmeier (Horiba)
        "Sellmeier_Horiba": {
            "coefficients": {
                "B1": 0.696750,
                "B2": 0.408218,
                "B3": 0.890815,
                "C1": 0.069066**2,
                "C2": 0.115662**2,
                "C3": 9.900559**2,
            },
            "wavelength_range_nm": (185.0, 2500.0),
            "temperature_K": STANDARD_TEMPERATURE_K,
            "source": "Horiba Jobin Yvon",
        },
        # Modèle pour la dépendance en température (dn/dT)
        "dn_dT": {
            "coefficients": {
                "D0": 7.0e-6,   # Coefficient principal
                "D1": 1.0e-8,   # Coefficient secondaire
                "T0": 293.15,   # Température de référence
            },
            "temperature_range_K": (200.0, 500.0),
            "source": "Experimental data",
        },
    },
    
    # BK7 (Schott) - 0.3-2.5 µm
    "BK7": {
        "Sellmeier": {
            "coefficients": {
                "B1": 1.03961212,
                "B2": 0.231792344,
                "B3": 1.01046945,
                "C1": 0.00600069867**2,
                "C2": 0.0200179144**2,
                "C3": 103.560653**2,
            },
            "wavelength_range_nm": (300.0, 2500.0),
            "temperature_K": STANDARD_TEMPERATURE_K,
            "source": "Schott Glass Catalog",
        },
        "dn_dT": {
            "coefficients": {
                "D0": 7.1e-6,
                "D1": 1.2e-8,
                "T0": 293.15,
            },
            "temperature_range_K": (200.0, 500.0),
            "source": "Schott technical data",
        },
    },
    
    # SF5 (Schott) - 0.3-2.5 µm
    "SF5": {
        "Sellmeier": {
            "coefficients": {
                "B1": 1.43171315,
                "B2": 0.26447592,
                "B3": 1.09976717,
                "C1": 0.00947447347**2,
                "C2": 0.0447937948**2,
                "C3": 97.9952058**2,
            },
            "wavelength_range_nm": (300.0, 2500.0),
            "temperature_K": STANDARD_TEMPERATURE_K,
            "source": "Schott Glass Catalog",
        },
        "dn_dT": {
            "coefficients": {
                "D0": 8.2e-6,
                "D1": 1.5e-8,
                "T0": 293.15,
            },
            "temperature_range_K": (200.0, 500.0),
            "source": "Schott technical data",
        },
    },
    
    # Silicium (Si) - 1.2-14 µm (IR)
    "Silicon": {
        "Sellmeier_IR": {
            "coefficients": {
                "A": 11.6858,
                "B": 0.939816,
                "C": 8.10461e-3,  # µm⁻²
            },
            "wavelength_range_nm": (1200.0, 14000.0),
            "temperature_K": STANDARD_TEMPERATURE_K,
            "source": "Li, Optics Letters 8, 308 (1983)",
        },
        # Modèle alternatif : Polynomial
        "Polynomial": {
            "coefficients": {
                "a0": 3.41696,
                "a1": -0.128356,
                "a2": 0.013921,
                "a3": -0.000439,
            },
            "wavelength_range_nm": (1200.0, 8000.0),
            "temperature_K": STANDARD_TEMPERATURE_K,
            "source": "Edwards, Applied Optics 24, 4483 (1985)",
        },
        "dn_dT": {
            "coefficients": {
                "D0": 1.8e-5,
                "D1": 5.0e-9,
                "T0": 293.15,
            },
            "temperature_range_K": (200.0, 500.0),
            "source": "Experimental data",
        },
    },
}

# --- Données d'expansion thermique ---
# Format : {"matériau": {"modèle": ..., "coefficients": {...}, "plage_K": [...], "source": ...}}
THERMAL_EXPANSION_DATA = {
    # Matériaux optiques
    "Fused_Silica": {
        "constant": {
            "coefficients": {"CTE": 0.51e-6},  # ppm/°C
            "temperature_range_K": (0.0, 1000.0),
            "source": "Corning, Fused Silica Technical Data",
        },
        "polynomial": {
            "coefficients": {
                "a": 0.51e-6,
                "b": 0.0,
                "c": 0.0,
            },
            "temperature_range_K": (0.0, 1000.0),
            "source": "Corning, Fused Silica Technical Data",
        },
    },
    "BK7": {
        "constant": {
            "coefficients": {"CTE": 7.1e-6},  # ppm/°C
            "temperature_range_K": (0.0, 300.0),
            "source": "Schott Glass Catalog",
        },
    },
    "SF5": {
        "constant": {
            "coefficients": {"CTE": 8.2e-6},  # ppm/°C
            "temperature_range_K": (0.0, 300.0),
            "source": "Schott Glass Catalog",
        },
    },
    "Silicon": {
        "polynomial": {
            "coefficients": {
                "a": 2.56e-6,   # CTE à 20°C
                "b": 1.2e-9,    # Coefficient linéaire
                "c": 0.0,       # Coefficient quadratique
            },
            "temperature_range_K": (0.0, 500.0),
            "source": "Okada, Journal of Applied Physics 45, 3564 (1974)",
        },
    },
    
    # Matériaux mécaniques
    "Steel": {
        "constant": {
            "coefficients": {"CTE": 12.0e-6},  # ppm/°C (acier doux)
            "temperature_range_K": (0.0, 500.0),
            "source": "Engineering Toolbox",
        },
    },
    "Aluminum": {
        "constant": {
            "coefficients": {"CTE": 23.1e-6},  # ppm/°C
            "temperature_range_K": (0.0, 500.0),
            "source": "Engineering Toolbox",
        },
    },
    "Invar": {
        "constant": {
            "coefficients": {"CTE": 1.2e-6},  # ppm/°C (allage Fe-Ni)
            "temperature_range_K": (0.0, 200.0),
            "source": "Engineering Toolbox",
        },
    },
    "Copper": {
        "constant": {
            "coefficients": {"CTE": 16.5e-6},  # ppm/°C
            "temperature_range_K": (0.0, 500.0),
            "source": "Engineering Toolbox",
        },
    },
}

# --- Autres propriétés des matériaux ---
MATERIAL_PROPERTIES = {
    "Fused_Silica": {
        "density_kg_m3": 2200.0,
        "young_modulus_Pa": 73.1e9,
        "poisson_ratio": 0.17,
        "thermal_conductivity_W_mK": 1.38,
        "specific_heat_J_kgK": 745.0,
        "absorption_coefficient_m_minus_1": 0.0,  # Négligeable dans le visible/IR
    },
    "BK7": {
        "density_kg_m3": 2510.0,
        "young_modulus_Pa": 82.0e9,
        "poisson_ratio": 0.206,
        "thermal_conductivity_W_mK": 1.114,
        "specific_heat_J_kgK": 858.0,
        "absorption_coefficient_m_minus_1": 0.0,
    },
    "SF5": {
        "density_kg_m3": 3860.0,
        "young_modulus_Pa": 76.8e9,
        "poisson_ratio": 0.257,
        "thermal_conductivity_W_mK": 0.858,
        "specific_heat_J_kgK": 520.0,
        "absorption_coefficient_m_minus_1": 0.0,
    },
    "Silicon": {
        "density_kg_m3": 2329.0,
        "young_modulus_Pa": 190.0e9,
        "poisson_ratio": 0.28,
        "thermal_conductivity_W_mK": 149.0,
        "specific_heat_J_kgK": 700.0,
        "absorption_coefficient_m_minus_1": 10.0,  # ~10 m⁻¹ dans l'IR
    },
    "Steel": {
        "density_kg_m3": 7850.0,
        "young_modulus_Pa": 200.0e9,
        "poisson_ratio": 0.28,
        "thermal_conductivity_W_mK": 50.0,
        "specific_heat_J_kgK": 500.0,
        "absorption_coefficient_m_minus_1": 0.0,
    },
    "Aluminum": {
        "density_kg_m3": 2700.0,
        "young_modulus_Pa": 69.0e9,
        "poisson_ratio": 0.33,
        "thermal_conductivity_W_mK": 235.0,
        "specific_heat_J_kgK": 900.0,
        "absorption_coefficient_m_minus_1": 0.0,
    },
    "Invar": {
        "density_kg_m3": 8050.0,
        "young_modulus_Pa": 148.0e9,
        "poisson_ratio": 0.26,
        "thermal_conductivity_W_mK": 11.0,
        "specific_heat_J_kgK": 500.0,
        "absorption_coefficient_m_minus_1": 0.0,
    },
    "Copper": {
        "density_kg_m3": 8960.0,
        "young_modulus_Pa": 120.0e9,
        "poisson_ratio": 0.34,
        "thermal_conductivity_W_mK": 401.0,
        "specific_heat_J_kgK": 385.0,
        "absorption_coefficient_m_minus_1": 0.0,
    },
}


# =============================================================================
# 4. CLASSE MATERIAL / MATERIAL CLASS
# =============================================================================

class MaterialBehaviour:
    """
    FR: Classe principale pour la gestion du comportement des matériaux.
        Permet de calculer l'indice de réfraction, l'expansion thermique,
        la réflectance, la transmittance, et la variation de puissance optique.

    EN: Main class for managing material behavior.
        Allows calculating refractive index, thermal expansion,
        reflectance, transmittance, and optical power variation.

    Attributes:
        material (Material): Matériau à analyser.
        logger (logging.Logger): Logger pour le débogage.
    """

    def __init__(self, material_name: str):
        """
        FR: Initialise le comportement du matériau.

        EN: Initializes the material behavior.

        Args:
            material_name (str): Nom du matériau. Options:
                - "Fused_Silica"
                - "BK7"
                - "SF5"
                - "Silicon"
                - "Steel"
                - "Aluminum"
                - "Invar"
                - "Copper"

        Raises:
            ValueError: Si le matériau est inconnu.
        """
        if material_name not in REFRACTIVE_INDEX_DATA and material_name not in THERMAL_EXPANSION_DATA:
            raise ValueError(
                f"Matériau inconnu : {material_name}. "
                f"Options disponibles : {list(REFRACTIVE_INDEX_DATA.keys()) + list(THERMAL_EXPANSION_DATA.keys())}"
            )

        self.material_name = material_name
        self.material_type = MaterialType.OPTICAL if material_name in REFRACTIVE_INDEX_DATA else MaterialType.MECHANICAL
        
        # Charger les propriétés
        self._load_material_properties()
        
        # Configuration du logger
        self.logger = logging.getLogger("MaterialBehaviour")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        
        self.logger.info("MaterialBehaviour initialized with material: %s", material_name)

    def _load_material_properties(self) -> None:
        """
        FR: Charge les propriétés du matériau depuis les bases de données.

        EN: Loads material properties from databases.
        """
        # Déterminer le type de matériau
        if self.material_name in REFRACTIVE_INDEX_DATA:
            # Matériau optique
            self.material_type = MaterialType.OPTICAL
            
            # Charger le modèle d'indice de réfraction
            refractive_data = REFRACTIVE_INDEX_DATA[self.material_name]
            # Utiliser le premier modèle disponible
            model_type = list(refractive_data.keys())[0]
            model_data = refractive_data[model_type]
            
            self.refractive_index_model = RefractiveIndexModel(
                model_type=model_type,
                coefficients=model_data["coefficients"],
                wavelength_range_nm=model_data["wavelength_range_nm"],
                temperature_K=model_data.get("temperature_K", STANDARD_TEMPERATURE_K),
                source=model_data.get("source", "")
            )
            
            # Charger le modèle dn/dT si disponible
            if "dn_dT" in refractive_data:
                dn_dT_data = refractive_data["dn_dT"]
                self.dn_dT_model = RefractiveIndexModel(
                    model_type="dn_dT",
                    coefficients=dn_dT_data["coefficients"],
                    wavelength_range_nm=dn_dT_data.get("wavelength_range_nm", (0.0, 1e6)),
                    temperature_K=dn_dT_data.get("temperature_K", STANDARD_TEMPERATURE_K),
                    source=dn_dT_data.get("source", "")
                )
            else:
                self.dn_dT_model = None
            
            # Charger les propriétés optiques
            optical_props = MATERIAL_PROPERTIES.get(self.material_name, {})
            thermal_expansion_data = THERMAL_EXPANSION_DATA.get(self.material_name, {})
            thermal_model_type = list(thermal_expansion_data.keys())[0] if thermal_expansion_data else "constant"
            thermal_model_data = thermal_expansion_data.get(thermal_model_type, {})
            
            self.thermal_expansion_model = ThermalExpansionModel(
                cte_model=thermal_model_type,
                coefficients=thermal_model_data.get("coefficients", {}),
                temperature_range_K=thermal_model_data.get("temperature_range_K", (0.0, 1000.0)),
                source=thermal_model_data.get("source", "")
            )
            
            self.optical_properties = OpticalProperties(
                refractive_index_model=self.refractive_index_model,
                thermal_expansion_model=self.thermal_expansion_model,
                absorption_coefficient=optical_props.get("absorption_coefficient_m_minus_1", 0.0),
                reference_wavelength_nm=optical_props.get("reference_wavelength_nm", 633.0)
            )
        else:
            # Matériau mécanique
            self.material_type = MaterialType.MECHANICAL
            self.optical_properties = None
            
            # Charger le modèle d'expansion thermique
            thermal_expansion_data = THERMAL_EXPANSION_DATA.get(self.material_name, {})
            thermal_model_type = list(thermal_expansion_data.keys())[0] if thermal_expansion_data else "constant"
            thermal_model_data = thermal_expansion_data.get(thermal_model_type, {})
            
            self.thermal_expansion_model = ThermalExpansionModel(
                cte_model=thermal_model_type,
                coefficients=thermal_model_data.get("coefficients", {}),
                temperature_range_K=thermal_model_data.get("temperature_range_K", (0.0, 1000.0)),
                source=thermal_model_data.get("source", "")
            )
        
        # Charger les autres propriétés
        self.density_kg_m3 = MATERIAL_PROPERTIES.get(self.material_name, {}).get("density_kg_m3", 0.0)
        self.young_modulus_Pa = MATERIAL_PROPERTIES.get(self.material_name, {}).get("young_modulus_Pa", 0.0)
        self.poisson_ratio = MATERIAL_PROPERTIES.get(self.material_name, {}).get("poisson_ratio", 0.0)
        self.thermal_conductivity_W_mK = MATERIAL_PROPERTIES.get(self.material_name, {}).get("thermal_conductivity_W_mK", 0.0)
        self.specific_heat_J_kgK = MATERIAL_PROPERTIES.get(self.material_name, {}).get("specific_heat_J_kgK", 0.0)

    # =========================================================================
    # Méthodes pour l'indice de réfraction / Refractive Index Methods
    # =========================================================================

    def get_refractive_index(
        self,
        wavelength_nm: Union[float, np.ndarray],
        temperature_K: float = STANDARD_TEMPERATURE_K,
        model: Optional[str] = None,
    ) -> Union[float, np.ndarray]:
        """
        FR: Calcule l'indice de réfraction du matériau pour une longueur d'onde donnée.
            Si le matériau a un modèle dn/dT, la température est prise en compte.

        EN: Calculates the refractive index of the material for a given wavelength.
            If the material has a dn/dT model, temperature is taken into account.

        Args:
            wavelength_nm (float or np.ndarray): Longueur d'onde en nm.
            temperature_K (float): Température en Kelvin (défaut: 293.15 K = 20°C).
            model (str, optional): Modèle à utiliser (défaut: premier modèle disponible).

        Returns:
            float or np.ndarray: Indice de réfraction (n).

        Raises:
            ValueError: Si le matériau n'a pas de modèle d'indice de réfraction.
        """
        if self.optical_properties is None:
            raise ValueError(f"Le matériau {self.material_name} n'a pas de propriétés optiques.")
        
        if model is None:
            model = self.refractive_index_model.model_type
        
        # Convertir la longueur d'onde en µm pour les formules
        wavelength_um = wavelength_nm / 1000.0
        
        if model == "Sellmeier":
            return self._sellmeier_model(wavelength_um, temperature_K)
        elif model == "Sellmeier_Horiba":
            return self._sellmeier_model(wavelength_um, temperature_K)
        elif model == "Sellmeier_IR":
            return self._sellmeier_ir_model(wavelength_um, temperature_K)
        elif model == "Polynomial":
            return self._polynomial_model(wavelength_um, temperature_K)
        else:
            raise ValueError(f"Modèle inconnu : {model}")

    def _sellmeier_model(
        self,
        wavelength_um: Union[float, np.ndarray],
        temperature_K: float,
    ) -> Union[float, np.ndarray]:
        """
        FR: Calcule l'indice de réfraction avec le modèle de Sellmeier.
            n² = 1 + B1*λ²/(λ² - C1) + B2*λ²/(λ² - C2) + B3*λ²/(λ² - C3)

        EN: Calculates the refractive index using the Sellmeier model.
            n² = 1 + B1*λ²/(λ² - C1) + B2*λ²/(λ² - C2) + B3*λ²/(λ² - C3)

        Args:
            wavelength_um (float or np.ndarray): Longueur d'onde en µm.
            temperature_K (float): Température en Kelvin.

        Returns:
            float or np.ndarray: Indice de réfraction (n).
        """
        coeffs = self.refractive_index_model.coefficients
        
        # Calculer n²
        term1 = coeffs["B1"] * wavelength_um**2 / (wavelength_um**2 - coeffs["C1"])
        term2 = coeffs["B2"] * wavelength_um**2 / (wavelength_um**2 - coeffs["C2"])
        term3 = coeffs["B3"] * wavelength_um**2 / (wavelength_um**2 - coeffs["C3"])
        n_squared = 1.0 + term1 + term2 + term3
        
        # Calculer n
        n = np.sqrt(n_squared)
        
        # Appliquer la correction de température si disponible
        if self.dn_dT_model is not None:
            dn_dT = self._get_dn_dT(temperature_K)
            n_reference = self.get_refractive_index(wavelength_um * 1000.0, STANDARD_TEMPERATURE_K, model)
            n = n_reference + dn_dT * (temperature_K - STANDARD_TEMPERATURE_K)
        
        return n

    def _sellmeier_ir_model(
        self,
        wavelength_um: Union[float, np.ndarray],
        temperature_K: float,
    ) -> Union[float, np.ndarray]:
        """
        FR: Calcule l'indice de réfraction avec le modèle de Sellmeier pour l'IR (Silicium).
            n² = A + B*λ²/(λ² - C)

        EN: Calculates the refractive index using the IR Sellmeier model (Silicon).
            n² = A + B*λ²/(λ² - C)

        Args:
            wavelength_um (float or np.ndarray): Longueur d'onde en µm.
            temperature_K (float): Température en Kelvin.

        Returns:
            float or np.ndarray: Indice de réfraction (n).
        """
        coeffs = self.refractive_index_model.coefficients
        
        # Calculer n²
        n_squared = coeffs["A"] + coeffs["B"] * wavelength_um**2 / (wavelength_um**2 - coeffs["C"])
        
        # Calculer n
        n = np.sqrt(n_squared)
        
        # Appliquer la correction de température si disponible
        if self.dn_dT_model is not None:
            dn_dT = self._get_dn_dT(temperature_K)
            n_reference = self.get_refractive_index(wavelength_um * 1000.0, STANDARD_TEMPERATURE_K)
            n = n_reference + dn_dT * (temperature_K - STANDARD_TEMPERATURE_K)
        
        return n

    def _polynomial_model(
        self,
        wavelength_um: Union[float, np.ndarray],
        temperature_K: float,
    ) -> Union[float, np.ndarray]:
        """
        FR: Calcule l'indice de réfraction avec un modèle polynomial.
            n = a0 + a1*λ + a2*λ² + a3*λ³ + ...

        EN: Calculates the refractive index using a polynomial model.
            n = a0 + a1*λ + a2*λ² + a3*λ³ + ...

        Args:
            wavelength_um (float or np.ndarray): Longueur d'onde en µm.
            temperature_K (float): Température en Kelvin.

        Returns:
            float or np.ndarray: Indice de réfraction (n).
        """
        coeffs = self.refractive_index_model.coefficients
        
        # Calculer n
        n = np.zeros_like(wavelength_um) if isinstance(wavelength_um, np.ndarray) else 0.0
        for i, key in enumerate(sorted(coeffs.keys(), key=lambda x: int(x[1:]))):
            power = int(key[1:]) if key != "a0" else 0
            n += coeffs[key] * (wavelength_um ** power)
        
        # Appliquer la correction de température si disponible
        if self.dn_dT_model is not None:
            dn_dT = self._get_dn_dT(temperature_K)
            n_reference = self.get_refractive_index(wavelength_um * 1000.0, STANDARD_TEMPERATURE_K)
            n += dn_dT * (temperature_K - STANDARD_TEMPERATURE_K)
        
        return n

    def _get_dn_dT(self, temperature_K: float) -> float:
        """
        FR: Calcule la dérivée de l'indice de réfraction par rapport à la température (dn/dT).

        EN: Calculates the derivative of the refractive index with respect to temperature (dn/dT).

        Args:
            temperature_K (float): Température en Kelvin.

        Returns:
            float: dn/dT en K⁻¹.
        """
        if self.dn_dT_model is None:
            return 0.0
        
        coeffs = self.dn_dT_model.coefficients
        T0 = coeffs.get("T0", STANDARD_TEMPERATURE_K)
        
        if self.dn_dT_model.model_type == "dn_dT":
            # Modèle simple : dn/dT = D0 + D1*(T - T0)
            return coeffs.get("D0", 0.0) + coeffs.get("D1", 0.0) * (temperature_K - T0)
        else:
            return coeffs.get("D0", 0.0)

    # =========================================================================
    # Méthodes pour l'expansion thermique / Thermal Expansion Methods
    # =========================================================================

    def get_thermal_expansion(
        self,
        temperature_K: Union[float, np.ndarray],
        reference_temperature_K: float = STANDARD_TEMPERATURE_K,
    ) -> Union[float, np.ndarray]:
        """
        FR: Calcule le facteur d'expansion thermique pour une température donnée.
            Retourne ΔL/L = CTE * (T - T_ref), où CTE est le coefficient d'expansion thermique.

        EN: Calculates the thermal expansion factor for a given temperature.
            Returns ΔL/L = CTE * (T - T_ref), where CTE is the coefficient of thermal expansion.

        Args:
            temperature_K (float or np.ndarray): Température en Kelvin.
            reference_temperature_K (float): Température de référence en Kelvin (défaut: 293.15 K).

        Returns:
            float or np.ndarray: Facteur d'expansion thermique (ΔL/L).
        """
        cte_model = self.thermal_expansion_model.cte_model
        coeffs = self.thermal_expansion_model.coefficients
        
        if cte_model == "constant":
            cte = coeffs.get("CTE", 0.0)
            return cte * (temperature_K - reference_temperature_K)
        elif cte_model == "linear":
            cte0 = coeffs.get("CTE0", 0.0)
            dCTE_dT = coeffs.get("dCTE_dT", 0.0)
            T0 = coeffs.get("T0", reference_temperature_K)
            cte = cte0 + dCTE_dT * (temperature_K - T0)
            return cte * (temperature_K - reference_temperature_K)
        elif cte_model == "polynomial":
            cte = coeffs.get("a", 0.0) + coeffs.get("b", 0.0) * temperature_K + coeffs.get("c", 0.0) * temperature_K**2
            return cte * (temperature_K - reference_temperature_K)
        else:
            raise ValueError(f"Modèle d'expansion thermique inconnu : {cte_model}")

    def get_thermal_dilation(
        self,
        length_m: float,
        temperature_K: float,
        reference_temperature_K: float = STANDARD_TEMPERATURE_K,
    ) -> float:
        """
        FR: Calcule la dilatation thermique absolue d'une longueur donnée.
            ΔL = L0 * CTE * (T - T_ref)

        EN: Calculates the absolute thermal dilation of a given length.
            ΔL = L0 * CTE * (T - T_ref)

        Args:
            length_m (float): Longueur initiale en mètres.
            temperature_K (float): Température en Kelvin.
            reference_temperature_K (float): Température de référence en Kelvin (défaut: 293.15 K).

        Returns:
            float: Dilatation thermique en mètres (ΔL).
        """
        expansion_factor = self.get_thermal_expansion(temperature_K, reference_temperature_K)
        return length_m * expansion_factor

    def get_thermal_contraction(
        self,
        length_m: float,
        temperature_K: float,
        reference_temperature_K: float = STANDARD_TEMPERATURE_K,
    ) -> float:
        """
        FR: Calcule la contraction thermique (valeur négative de la dilatation).

        EN: Calculates the thermal contraction (negative value of dilation).

        Args:
            length_m (float): Longueur initiale en mètres.
            temperature_K (float): Température en Kelvin.
            reference_temperature_K (float): Température de référence en Kelvin (défaut: 293.15 K).

        Returns:
            float: Contraction thermique en mètres (ΔL, négatif si T < T_ref).
        """
        return -self.get_thermal_dilation(length_m, temperature_K, reference_temperature_K)

    def get_thermal_expansion_coefficient(
        self,
        temperature_K: float = STANDARD_TEMPERATURE_K,
    ) -> float:
        """
        FR: Retourne le coefficient d'expansion thermique (CTE) à une température donnée.

        EN: Returns the coefficient of thermal expansion (CTE) at a given temperature.

        Args:
            temperature_K (float): Température en Kelvin (défaut: 293.15 K).

        Returns:
            float: CTE en K⁻¹.
        """
        cte_model = self.thermal_expansion_model.cte_model
        coeffs = self.thermal_expansion_model.coefficients
        
        if cte_model == "constant":
            return coeffs.get("CTE", 0.0)
        elif cte_model == "linear":
            cte0 = coeffs.get("CTE0", 0.0)
            dCTE_dT = coeffs.get("dCTE_dT", 0.0)
            T0 = coeffs.get("T0", STANDARD_TEMPERATURE_K)
            return cte0 + dCTE_dT * (temperature_K - T0)
        elif cte_model == "polynomial":
            return coeffs.get("a", 0.0) + coeffs.get("b", 0.0) * temperature_K + coeffs.get("c", 0.0) * temperature_K**2
        else:
            raise ValueError(f"Modèle d'expansion thermique inconnu : {cte_model}")

    # =========================================================================
    # Méthodes pour la réflectance et la transmittance / Reflectance and Transmittance Methods
    # =========================================================================

    def get_reflectance(
        self,
        wavelength_nm: float,
        angle_deg: float = 0.0,
        polarization: Polarization = Polarization.NONE,
        n_medium: float = 1.0,
    ) -> float:
        """
        FR: Calcule la réflectance (coefficient de réflexion) pour une interface entre deux milieux.
            Utilise les équations de Fresnel.

        EN: Calculates the reflectance (reflection coefficient) for an interface between two media.
            Uses Fresnel equations.

        Args:
            wavelength_nm (float): Longueur d'onde en nm.
            angle_deg (float): Angle d'incidence en degrés (défaut: 0.0 = incidence normale).
            polarization (Polarization): Polarisation de la lumière (défaut: NONE).
            n_medium (float): Indice de réfraction du milieu incident (défaut: 1.0 = air).

        Returns:
            float: Réflectance (0 ≤ R ≤ 1).

        Notes:
            - Pour une incidence normale, R = ((n2 - n1)/(n2 + n1))².
            - Pour une incidence oblique, utilise les équations de Fresnel.
        """
        if self.optical_properties is None:
            raise ValueError(f"Le matériau {self.material_name} n'a pas de propriétés optiques.")
        
        # Calculer l'indice de réfraction du matériau à la longueur d'onde donnée
        n_material = self.get_refractive_index(wavelength_nm)
        
        # Incidence normale
        if angle_deg == 0.0:
            return ((n_material - n_medium) / (n_material + n_medium)) ** 2
        
        # Incidence oblique
        angle_rad = np.deg2rad(angle_deg)
        
        # Calculer les composantes s et p
        if polarization == Polarization.S or polarization == Polarization.NONE:
            # Réflectance pour la polarisation s
            r_s = (n_medium * np.cos(angle_rad) - n_material * np.sqrt(1 - (n_medium / n_material * np.sin(angle_rad))**2)) / \
                  (n_medium * np.cos(angle_rad) + n_material * np.sqrt(1 - (n_medium / n_material * np.sin(angle_rad))**2))
            R_s = r_s**2
        else:
            R_s = 0.0
        
        if polarization == Polarization.P or polarization == Polarization.NONE:
            # Réflectance pour la polarisation p
            r_p = (n_medium * np.sqrt(1 - (n_medium / n_material * np.sin(angle_rad))**2) - n_material * np.cos(angle_rad)) / \
                  (n_medium * np.sqrt(1 - (n_medium / n_material * np.sin(angle_rad))**2) + n_material * np.cos(angle_rad))
            R_p = r_p**2
        else:
            R_p = 0.0
        
        # Retourner la réflectance appropriée
        if polarization == Polarization.S:
            return float(R_s)
        elif polarization == Polarization.P:
            return float(R_p)
        elif polarization == Polarization.CIRCULAR or polarization == Polarization.NONE:
            # Réflectance moyenne pour lumière non polarisée
            return float((R_s + R_p) / 2)
        else:
            return float((R_s + R_p) / 2)

    def get_transmittance(
        self,
        wavelength_nm: float,
        thickness_mm: float,
        angle_deg: float = 0.0,
        polarization: Polarization = Polarization.NONE,
        n_medium: float = 1.0,
    ) -> float:
        """
        FR: Calcule la transmittance à travers un matériau d'épaisseur donnée.
            Prend en compte l'absorption et les réflexions multiples.

        EN: Calculates the transmittance through a material of given thickness.
            Takes into account absorption and multiple reflections.

        Args:
            wavelength_nm (float): Longueur d'onde en nm.
            thickness_mm (float): Épaisseur du matériau en mm.
            angle_deg (float): Angle d'incidence en degrés (défaut: 0.0).
            polarization (Polarization): Polarisation de la lumière (défaut: NONE).
            n_medium (float): Indice de réfraction du milieu incident (défaut: 1.0 = air).

        Returns:
            float: Transmittance (0 ≤ T ≤ 1).

        Notes:
            - La transmittance inclut les pertes par absorption et réflexion.
            - Formules : T = (1-R)² * exp(-α*d) / (1 - R² * exp(-2α*d))
              où R est la réflectance, α le coefficient d'absorption, et d l'épaisseur.
        """
        if self.optical_properties is None:
            raise ValueError(f"Le matériau {self.material_name} n'a pas de propriétés optiques.")
        
        # Calculer la réflectance
        R = self.get_reflectance(wavelength_nm, angle_deg, polarization, n_medium)
        
        # Calculer le coefficient d'absorption à la longueur d'onde donnée
        # (On utilise une interpolation linéaire pour l'instant)
        alpha = self._get_absorption_coefficient(wavelength_nm)
        
        # Convertir l'épaisseur en mètres
        thickness_m = thickness_mm * 1e-3
        
        # Calculer la transmittance
        # T = (1-R)^2 * exp(-α*d) / (1 - R^2 * exp(-2α*d))
        numerator = (1 - R)**2 * np.exp(-alpha * thickness_m)
        denominator = 1 - R**2 * np.exp(-2 * alpha * thickness_m)
        
        # Éviter la division par zéro
        if denominator == 0:
            return 0.0
        
        return float(numerator / denominator)

    def _get_absorption_coefficient(self, wavelength_nm: float) -> float:
        """
        FR: Retourne le coefficient d'absorption à une longueur d'onde donnée.
            Pour l'instant, utilise une valeur constante (à améliorer avec des données spectrale).

        EN: Returns the absorption coefficient at a given wavelength.
            For now, uses a constant value (to be improved with spectral data).

        Args:
            wavelength_nm (float): Longueur d'onde en nm.

        Returns:
            float: Coefficient d'absorption en m⁻¹.
        """
        return self.optical_properties.absorption_coefficient

    # =========================================================================
    # Méthodes pour la puissance optique / Optical Power Methods
    # =========================================================================

    def get_optical_power_variation(
        self,
        focal_length_mm: float,
        temperature_K: float,
        reference_temperature_K: float = STANDARD_TEMPERATURE_K,
        wavelength_nm: float = 633.0,
    ) -> float:
        """
        FR: Calcule la variation de puissance optique (1/f) due à la dilatation thermique.
            Pour une lentille, la puissance optique P = (n - 1) * (1/R1 - 1/R2).
            La variation de puissance vient de :
            - La variation de l'indice de réfraction (dn/dT)
            - La variation des rayons de courbure (dR/dT)

        EN: Calculates the variation of optical power (1/f) due to thermal expansion.
            For a lens, the optical power P = (n - 1) * (1/R1 - 1/R2).
            The power variation comes from:
            - The variation of the refractive index (dn/dT)
            - The variation of the curvature radii (dR/dT)

        Args:
            focal_length_mm (float): Distance focale initiale en mm.
            temperature_K (float): Température en Kelvin.
            reference_temperature_K (float): Température de référence en Kelvin (défaut: 293.15 K).
            wavelength_nm (float): Longueur d'onde en nm (défaut: 633.0).

        Returns:
            float: Variation relative de la puissance optique (ΔP/P).

        Notes:
            - Pour une lentille mince : 1/f = (n - 1) * (1/R1 - 1/R2)
            - Δ(1/f) / (1/f) ≈ (Δn / (n - 1)) - (ΔR / R)
        """
        if self.optical_properties is None:
            raise ValueError(f"Le matériau {self.material_name} n'a pas de propriétés optiques.")
        
        # Calculer l'indice de réfraction à la température de référence
        n_ref = self.get_refractive_index(wavelength_nm, reference_temperature_K)
        
        # Calculer l'indice de réfraction à la nouvelle température
        n_new = self.get_refractive_index(wavelength_nm, temperature_K)
        
        # Variation de l'indice
        delta_n = n_new - n_ref
        
        # Variation relative de n
        delta_n_rel = delta_n / (n_ref - 1.0) if (n_ref - 1.0) != 0 else 0.0
        
        # Variation des rayons de courbure (due à la dilatation thermique)
        # Pour une lentille, R augmente avec la température : ΔR/R = CTE * ΔT
        delta_T = temperature_K - reference_temperature_K
        cte = self.get_thermal_expansion_coefficient(reference_temperature_K)
        delta_R_rel = cte * delta_T
        
        # Variation totale de la puissance optique
        delta_power_rel = delta_n_rel - delta_R_rel
        
        return delta_power_rel

    def get_focal_length_variation(
        self,
        focal_length_mm: float,
        temperature_K: float,
        reference_temperature_K: float = STANDARD_TEMPERATURE_K,
        wavelength_nm: float = 633.0,
    ) -> float:
        """
        FR: Calcule la variation de la distance focale due à la dilatation thermique.

        EN: Calculates the variation of the focal length due to thermal expansion.

        Args:
            focal_length_mm (float): Distance focale initiale en mm.
            temperature_K (float): Température en Kelvin.
            reference_temperature_K (float): Température de référence en Kelvin (défaut: 293.15 K).
            wavelength_nm (float): Longueur d'onde en nm (défaut: 633.0).

        Returns:
            float: Variation de la distance focale en mm (Δf).
        """
        delta_power_rel = self.get_optical_power_variation(
            focal_length_mm, temperature_K, reference_temperature_K, wavelength_nm
        )
        
        # P = 1/f, donc ΔP/P = -Δf/f
        delta_f_rel = -delta_power_rel
        
        return focal_length_mm * delta_f_rel

    # =========================================================================
    # Méthodes utilitaires / Utility Methods
    # =========================================================================

    def get_material_info(self) -> Dict:
        """
        FR: Retourne un dictionnaire avec toutes les informations sur le matériau.

        EN: Returns a dictionary with all material information.

        Returns:
            Dict: Informations sur le matériau.
        """
        info = {
            "name": self.material_name,
            "type": self.material_type.value,
            "density_kg_m3": self.density_kg_m3,
            "young_modulus_Pa": self.young_modulus_Pa,
            "poisson_ratio": self.poisson_ratio,
            "thermal_conductivity_W_mK": self.thermal_conductivity_W_mK,
            "specific_heat_J_kgK": self.specific_heat_J_kgK,
        }
        
        if self.optical_properties is not None:
            info["optical_properties"] = {
                "refractive_index_model": {
                    "model_type": self.refractive_index_model.model_type,
                    "wavelength_range_nm": self.refractive_index_model.wavelength_range_nm,
                    "source": self.refractive_index_model.source,
                },
                "absorption_coefficient_m_minus_1": self.optical_properties.absorption_coefficient,
            }
        
        info["thermal_expansion"] = {
            "model": self.thermal_expansion_model.cte_model,
            "CTE": self.get_thermal_expansion_coefficient(STANDARD_TEMPERATURE_K),
            "temperature_range_K": self.thermal_expansion_model.temperature_range_K,
            "source": self.thermal_expansion_model.source,
        }
        
        return info

    # =========================================================================
    # Méthodes pour télécharger des données depuis refractiveindex.info
    # =========================================================================

    @staticmethod
    def fetch_material_from_refractiveindex(material_name: str) -> Dict:
        """
        FR: Télécharge les données d'un matériau depuis refractiveindex.info.
            Cette méthode est expérimentale et nécessite une connexion Internet.

        EN: Fetches material data from refractiveindex.info.
            This method is experimental and requires an Internet connection.

        Args:
            material_name (str): Nom du matériau (ex: "fused_silica", "bk7").

        Returns:
            Dict: Dictionnaire contenant les données du matériau (indice de réfraction, etc.).

        Raises:
            ConnectionError: Si la connexion à refractiveindex.info échoue.
            ValueError: Si le matériau n'est pas trouvé.

        Notes:
            - L'API de refractiveindex.info n'est pas officielle, cette méthode peut ne pas fonctionner.
            - Les données sont retournées sous forme brute et doivent être traitées.
        """
        # URL de base de refractiveindex.info
        base_url = "https://refractiveindex.info/api"
        
        try:
            # Essayer de récupérer les données via l'API (non officielle)
            response = requests.get(f"{base_url}/materials/{material_name}.json", timeout=10)
            response.raise_for_status()
            data = response.json()
            return data
        except requests.exceptions.RequestException as e:
            logger.warning(f"Impossible de récupérer les données depuis refractiveindex.info: {e}")
            # Retourner un message d'erreur
            return {"error": f"Failed to fetch data for {material_name}: {e}"}

    @staticmethod
    def parse_refractiveindex_data(data: Dict) -> Dict:
        """
        FR: Parse les données de refractiveindex.info pour extraire les coefficients d'indice de réfraction.

        EN: Parses refractiveindex.info data to extract refractive index coefficients.

        Args:
            data (Dict): Données brutes de refractiveindex.info.

        Returns:
            Dict: Dictionnaire contenant les coefficients pour les modèles (Sellmeier, Cauchy, etc.).

        Notes:
            - Cette méthode est un exemple et doit être adaptée selon le format réel des données.
        """
        parsed_data = {}
        
        # Extraire les informations de base
        if "name" in data:
            parsed_data["name"] = data["name"]
        if "id" in data:
            parsed_data["id"] = data["id"]
        
        # Extraire les coefficients d'indice de réfraction
        if "n" in data and isinstance(data["n"], list):
            # Supposons que les données sont sous forme de liste de (wavelength, n)
            wavelengths = []
            indices = []
            for entry in data["n"]:
                if "wavelength" in entry and "value" in entry:
                    wavelengths.append(entry["wavelength"])
                    indices.append(entry["value"])
            
            if wavelengths and indices:
                # Créer une fonction d'interpolation
                parsed_data["interpolation_function"] = interp1d(
                    wavelengths, indices, kind='cubic', fill_value='extrapolate'
                )
        
        # Extraire les coefficients de température si disponibles
        if "temperature" in data:
            parsed_data["temperature_data"] = data["temperature"]
        
        return parsed_data


# =============================================================================
# 5. FONCTIONS UTILITAIRES / UTILITY FUNCTIONS
# =============================================================================

def get_available_materials() -> List[str]:
    """
    FR: Retourne la liste des matériaux disponibles.

    EN: Returns the list of available materials.

    Returns:
        List[str]: Liste des noms des matériaux.
    """
    return list(REFRACTIVE_INDEX_DATA.keys()) + list(THERMAL_EXPANSION_DATA.keys())


def create_material(
    name: str,
    refractive_index_model: Optional[RefractiveIndexModel] = None,
    thermal_expansion_model: Optional[ThermalExpansionModel] = None,
    **kwargs,
) -> Material:
    """
    FR: Crée un nouveau matériau avec des propriétés personnalisées.

    EN: Creates a new material with custom properties.

    Args:
        name (str): Nom du matériau.
        refractive_index_model (RefractiveIndexModel, optional): Modèle d'indice de réfraction.
        thermal_expansion_model (ThermalExpansionModel, optional): Modèle d'expansion thermique.
        **kwargs: Autres propriétés (density, young_modulus, etc.).

    Returns:
        Material: Nouveau matériau.
    """
    material_type = MaterialType.OPTICAL if refractive_index_model is not None else MaterialType.MECHANICAL
    
    optical_props = None
    if refractive_index_model is not None:
        optical_props = OpticalProperties(
            refractive_index_model=refractive_index_model,
            thermal_expansion_model=thermal_expansion_model,
            **{k: v for k, v in kwargs.items() if k in ["absorption_coefficient", "reference_wavelength_nm"]}
        )
    
    return Material(
        name=name,
        material_type=material_type,
        optical_properties=optical_props,
        thermal_expansion_model=thermal_expansion_model if thermal_expansion_model is not None else ThermalExpansionModel(),
        **{k: v for k, v in kwargs.items() if k in ["density_kg_m3", "young_modulus_Pa", "poisson_ratio", "thermal_conductivity_W_mK", "specific_heat_J_kgK"]}
    )


# =============================================================================
# 6. TESTS UNITAIRES / UNIT TESTS
# =============================================================================

class TestMaterialBehaviour:
    """
    FR: Classe de tests unitaires pour Material_Behaviour.py.
    EN: Unit test class for Material_Behaviour.py.
    """

    def test_fused_silica_refractive_index(self):
        """Test l'indice de réfraction de la silice fondue."""
        material = MaterialBehaviour("Fused_Silica")
        n_633 = material.get_refractive_index(633.0)
        # Valeur attendue pour 633 nm : ~1.458
        self.assertAlmostEqual(n_633, 1.458, delta=0.01)

    def test_bk7_refractive_index(self):
        """Test l'indice de réfraction du BK7."""
        material = MaterialBehaviour("BK7")
        n_633 = material.get_refractive_index(633.0)
        # Valeur attendue pour 633 nm : ~1.516
        self.assertAlmostEqual(n_633, 1.516, delta=0.01)

    def test_silicon_refractive_index(self):
        """Test l'indice de réfraction du silicium."""
        material = MaterialBehaviour("Silicon")
        n_1550 = material.get_refractive_index(1550.0)
        # Valeur attendue pour 1550 nm : ~3.47
        self.assertAlmostEqual(n_1550, 3.47, delta=0.05)

    def test_thermal_expansion_fused_silica(self):
        """Test l'expansion thermique de la silice fondue."""
        material = MaterialBehaviour("Fused_Silica")
        cte = material.get_thermal_expansion_coefficient()
        # Valeur attendue : ~0.51 ppm/°C
        self.assertAlmostEqual(cte, 0.51e-6, delta=0.1e-6)

    def test_thermal_expansion_bk7(self):
        """Test l'expansion thermique du BK7."""
        material = MaterialBehaviour("BK7")
        cte = material.get_thermal_expansion_coefficient()
        # Valeur attendue : ~7.1 ppm/°C
        self.assertAlmostEqual(cte, 7.1e-6, delta=0.5e-6)

    def test_thermal_dilation(self):
        """Test la dilatation thermique."""
        material = MaterialBehaviour("Fused_Silica")
        length_m = 0.1  # 10 cm
        delta_T = 100.0  # ΔT = 100°C
        dilation_m = material.get_thermal_dilation(length_m, STANDARD_TEMPERATURE_K + delta_T)
        # ΔL = L0 * CTE * ΔT = 0.1 * 0.51e-6 * 100 = 5.1e-6 m
        expected_dilation = length_m * 0.51e-6 * delta_T
        self.assertAlmostEqual(dilation_m, expected_dilation, delta=1e-9)

    def test_reflectance_normal_incidence(self):
        """Test la réflectance en incidence normale."""
        material = MaterialBehaviour("Fused_Silica")
        R = material.get_reflectance(633.0, angle_deg=0.0)
        # R = ((n-1)/(n+1))² = ((1.458-1)/(1.458+1))² ≈ 0.035
        expected_R = ((1.458 - 1.0) / (1.458 + 1.0)) ** 2
        self.assertAlmostEqual(R, expected_R, delta=0.01)

    def test_transmittance(self):
        """Test la transmittance."""
        material = MaterialBehaviour("Fused_Silica")
        # Avec absorption nulle et incidence normale
        T = material.get_transmittance(633.0, thickness_mm=10.0, angle_deg=0.0)
        # T ≈ 1 - R ≈ 0.965 (pour une interface air-verre)
        self.assertGreater(T, 0.9)

    def test_optical_power_variation(self):
        """Test la variation de puissance optique."""
        material = MaterialBehaviour("Fused_Silica")
        delta_power = material.get_optical_power_variation(
            focal_length_mm=100.0,
            temperature_K=STANDARD_TEMPERATURE_K + 100.0,
            reference_temperature_K=STANDARD_TEMPERATURE_K,
            wavelength_nm=633.0
        )
        # La variation doit être faible pour la silice fondue
        self.assertLess(abs(delta_power), 0.01)

    def test_get_available_materials(self):
        """Test la récupération des matériaux disponibles."""
        materials = get_available_materials()
        self.assertIn("Fused_Silica", materials)
        self.assertIn("BK7", materials)
        self.assertIn("Silicon", materials)
        self.assertIn("Steel", materials)


if __name__ == "__main__":
    import unittest
    unittest.main()

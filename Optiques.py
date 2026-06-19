"""
Optiques.py - Complete Version with Apertures and Diffraction Gratings
FR: Module pour la génération et la gestion d'optiques diverses (réelles ou parfaites).
    Permet de générer des cartes de phase pour :
    - Lentilles parfaites (paraxiales)
    - Lentilles réelles (plan-convexe, biconvexe, biconcave, plan-concave, ménisque)
    - Doublets collés (achromatiques, etc.)
    - Miroirs (plats, sphériques, paraboliques)
    - Beamsplitters (cubes, lames)
    - Fenêtres optiques (plates, inclinées)
    - Prismes (triangulaires, etc.)
    - Diaphragmes (stops d'ouverture)
    - Réseaux de diffraction
    
    Fonctionnalités clés :
    - Intégration avec Material_Behaviour.py pour les propriétés des matériaux
    - Ajout d'erreurs de front d'onde (WFE) : qualité de surface, parallélisme, aberrations
    - Positionnement : tilt (inclinaison) et décentrement
    - Masquage des zones non couvertes par l'optique
    - Caractéristiques mécaniques complètes (forme, épaisseur, rayons de courbure)
    - Facteurs de forme variés (circulaire, rectangulaire, carré, elliptique, hexagonale, octogonale)
    - Affichage automatique des cartes d'intensité et de phase

EN: Module for generating and managing various optical elements (ideal or real).
    Allows generating phase maps for:
    - Ideal lenses (paraxial)
    - Real lenses (plan-convex, biconvex, biconcave, plan-concave, meniscus)
    - Cemented doublets (achromatic, etc.)
    - Mirrors (flat, spherical, parabolic)
    - Beamsplitters (cube, plate)
    - Optical windows (flat, tilted)
    - Prisms (triangular, etc.)
    - Aperture stops (diaphragms)
    - Diffraction gratings
    
    Key features:
    - Integration with Material_Behaviour.py for material properties
    - Wavefront error (WFE) addition: surface roughness, parallelism, aberrations
    - Positioning: tilt and decentering
    - Masking of areas not covered by the optic
    - Complete mechanical characteristics (shape, thickness, curvature radii)
    - Various form factors (circular, rectangular, square, elliptical, hexagonal, octagonal)
    - Automatic display of intensity and phase maps

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch

Dependencies:
    - numpy
    - Material_Behaviour.py (for material properties)
    - Beam.py (for beam manipulation)
    - MathAndPhysicsTools.py (for grid creation and Zernike polynomials)
    - Visualization.py (for display functions)
    - Propagation.py (for beam propagation)
    - scipy (for FFT and interpolation, optional)
"""

import numpy as np
import logging
import os
import sys
from typing import Optional, Tuple, Dict, Union, List
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


# =============================================================================
# IMPORT DES DÉPENDANCES LOCALES / IMPORT LOCAL DEPENDENCIES
# =============================================================================

# Gestion des imports optionnels
try:
    from Material_Behaviour import MaterialBehaviour, STANDARD_TEMPERATURE_K, Polarization
    MATERIAL_BEHAVIOUR_AVAILABLE = True
except ImportError as e:
    MATERIAL_BEHAVIOUR_AVAILABLE = False
    logging.warning(f"Material_Behaviour module not available: {e}. Material properties will be limited.")

try:
    from Beam import Beam
    BEAM_AVAILABLE = True
except ImportError as e:
    BEAM_AVAILABLE = False
    logging.warning(f"Beam module not available: {e}. Beam manipulation features will be limited.")

try:
    from MathAndPhysicsTools import create_grid, zernike_polynomial
    MATH_TOOLS_AVAILABLE = True
except ImportError as e:
    MATH_TOOLS_AVAILABLE = False
    logging.warning(f"MathAndPhysicsTools module not available: {e}. Grid creation will be limited.")

try:
    from Visualization import plot_intensity, plot_phase, plot_beam_map
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    VISUALIZATION_AVAILABLE = False
    logging.warning(f"Visualization module not available: {e}. Display features will be disabled.")

try:
    from Propagation import Propagation
    PROPAGATION_AVAILABLE = True
except ImportError as e:
    PROPAGATION_AVAILABLE = False
    logging.warning(f"Propagation module not available: {e}. Beam propagation features will be limited.")


# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Optiques")


# =============================================================================
# 1. ENUMS ET CONSTANTES / ENUMS AND CONSTANTS
# =============================================================================

class OpticType(Enum):
    """FR: Type d'optique."""
    IDEAL_LENS = "ideal_lens"              # Lentille paraxiale parfaite
    SIMPLE_LENS = "simple_lens"            # Lentille simple (1 face courbe)
    DOUBLE_LENS = "double_lens"           # Lentille à 2 faces courbes
    DOUBLET_LENS = "doublet_lens"         # Doublet collé (2 lentilles)
    MIRROR = "mirror"                      # Miroir
    BEAMSPLITTER = "beamsplitter"          # Séparateur de faisceau
    WINDOW = "window"                      # Fenêtre optique
    PRISM = "prism"                        # Prisme
    ASPHERIC_LENS = "aspheric_lens"       # Lentille asphérique
    APERTURE_STOP = "aperture_stop"        # Diaphragme (stop d'ouverture)
    DIFFRACTION_GRATING = "diffraction_grating"  # Réseau de diffraction


class LensType(Enum):
    """FR: Type de lentille."""
    PLAN_CONVEX = "plan_convex"         # Plan-convexe
    BICONVEX = "biconvex"               # Biconvexe
    BICONCAVE = "biconcave"             # Biconcave
    PLAN_CONCAVE = "plan_concave"      # Plan-concave
    MENISCUS = "meniscus"               # Ménisque
    PLAN_PLAN = "plan_plan"             # Plan-plan (fenêtre)


class MirrorType(Enum):
    """FR: Type de miroir."""
    FLAT = "flat"                       # Plat
    SPHERICAL = "spherical"            # Sphérique
    PARABOLIC = "parabolic"            # Parabolique
    ELLIPTICAL = "elliptical"          # Elliptique


class BeamsplitterType(Enum):
    """FR: Type de séparateur de faisceau."""
    PLATE = "plate"                     # Lame
    CUBE = "cube"                      # Cube
    POLARIZING = "polarizing"          # Polarisant


class GratingType(Enum):
    """FR: Type de réseau de diffraction."""
    TRANSMISSION = "transmission"      # Réseau de transmission
    REFLECTION = "reflection"          # Réseau de réflexion


class WFESource(Enum):
    """FR: Source des erreurs de front d'onde."""
    SURFACE_ROUGHNESS = "surface_roughness"  # Rugosité de surface
    PARALLELISM = "parallelism"              # Parallélisme
    ZERNIKE = "zernike"                      # Polynômes de Zernike
    CUSTOM = "custom"                        # Carte personnalisée
    FILE = "file"                          # Depuis un fichier


class ApertureShape(Enum):
    """FR: Forme de l'aperture de l'optique."""
    CIRCULAR = "circular"               # Circulaire (défaut)
    RECTANGULAR = "rectangular"         # Rectangulaire
    SQUARE = "square"                   # Carré
    ELLIPTICAL = "elliptical"           # Elliptique
    HEXAGONAL = "hexagonal"             # Hexagonale
    OCTAGONAL = "octagonal"             # Octogonale


# Constantes physiques
C_LIGHT_M_S = 299792458  # Vitesse de la lumière en m/s


# =============================================================================
# 2. CLASSES POUR LES ERREURS DE FRONT D'ONDE / WAVEFRONT ERROR CLASSES
# =============================================================================

@dataclass
class WaveFrontError:
    """
    FR: Classe représentant les erreurs de front d'onde (WFE) d'une optique.

    EN: Class representing wavefront errors (WFE) of an optical element.

    Attributes:
        surface_roughness_nm (float): Rugosité de surface en nm (RMS).
        parallelism_arcsec (float): Erreur de parallélisme en secondes d'arc.
        zernike_coefficients (Dict): Coefficients des polynômes de Zernike.
        custom_phase_map (np.ndarray): Carte de phase personnalisée en nm.
    """
    surface_roughness_nm: float = 0.0
    parallelism_arcsec: float = 0.0
    zernike_coefficients: Dict[Tuple[int, int], float] = field(default_factory=dict)
    custom_phase_map: Optional[np.ndarray] = None
    wfe_source: WFESource = WFESource.SURFACE_ROUGHNESS

    def generate_phase_map(
        self,
        grid_x_mm: np.ndarray,
        grid_y_mm: np.ndarray,
        wavelength_nm: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """FR: Génère une carte de phase représentant les erreurs de front d'onde."""
        if seed is not None:
            np.random.seed(seed)
        
        phase_map = np.zeros_like(grid_x_mm)
        
        # 1. Rugosité de surface
        if self.surface_roughness_nm > 0:
            phase_rms = self.surface_roughness_nm
            noise = np.random.normal(0, phase_rms, grid_x_mm.shape)
            
            if grid_x_mm.shape[0] > 1 and grid_x_mm.shape[1] > 1:
                try:
                    from scipy.fft import fft2, ifft2
                    noise_fft = fft2(noise)
                    rows, cols = noise_fft.shape
                    crow, ccol = rows // 2, cols // 2
                    mask = np.ones_like(noise)
                    radius = min(rows, cols) // 10
                    y, x = np.ogrid[:rows, :cols]
                    mask_area = (x - ccol)**2 + (y - crow)**2 <= radius**2
                    mask[mask_area] = 0
                    noise_fft_filtered = noise_fft * mask
                    noise_filtered = np.real(ifft2(noise_fft_filtered))
                    current_rms = np.std(noise_filtered)
                    if current_rms > 0:
                        noise_filtered = noise_filtered * (phase_rms / current_rms)
                    phase_map += noise_filtered
                except ImportError:
                    phase_map += noise
            else:
                phase_map += noise
        
        # 2. Parallélisme
        if self.parallelism_arcsec > 0:
            parallelism_rad = self.parallelism_arcsec * np.pi / (180 * 3600)
            tilt_direction = np.pi / 4
            phase_map += (grid_x_mm * np.cos(tilt_direction) + 
                         grid_y_mm * np.sin(tilt_direction)) * parallelism_rad * wavelength_nm / 1e3
        
        # 3. Aberrations Zernike
        if self.zernike_coefficients and MATH_TOOLS_AVAILABLE:
            for (n, m), coeff_nm in self.zernike_coefficients.items():
                zernike = zernike_polynomial(n, m, grid_x_mm, grid_y_mm)
                phase_map += coeff_nm * zernike
        
        # 4. Carte personnalisée
        if self.custom_phase_map is not None:
            if self.custom_phase_map.shape != grid_x_mm.shape:
                try:
                    from scipy.interpolate import interp2d
                    x = grid_x_mm[0, :]
                    y = grid_y_mm[:, 0]
                    f = interp2d(x, y, self.custom_phase_map, kind='cubic')
                    new_x = np.linspace(x.min(), x.max(), grid_x_mm.shape[1])
                    new_y = np.linspace(y.min(), y.max(), grid_x_mm.shape[0])
                    phase_map += f(new_x, new_y)
                except ImportError:
                    pass
            else:
                phase_map += self.custom_phase_map
        
        return phase_map


@dataclass
class OpticSpecifications:
    """FR: Spécifications techniques d'une optique."""
    diameter_mm: float
    thickness_mm: float
    material_name: str = "Fused_Silica"
    surface_roughness_nm: float = 1.0
    parallelism_arcsec: float = 10.0
    clear_aperture_ratio: float = 0.9
    edge_thickness_mm: Optional[float] = None
    aperture_shape: ApertureShape = ApertureShape.CIRCULAR
    width_mm: Optional[float] = None
    height_mm: Optional[float] = None


# =============================================================================
# 3. CLASSE DE BASE POUR LES ÉLÉMENTS OPTIQUES
# =============================================================================

@dataclass
class OpticalElement(ABC):
    """FR: Classe de base abstraite pour tous les éléments optiques."""
    name: str
    specifications: OpticSpecifications
    position_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    tilt_deg: Tuple[float, float] = (0.0, 0.0)
    decentering_mm: Tuple[float, float] = (0.0, 0.0)
    temperature_K: float = STANDARD_TEMPERATURE_K if MATERIAL_BEHAVIOUR_AVAILABLE else 293.15
    wavelength_nm: float = 633.0
    display: bool = False
    display_dir: str = "output"
    
    material: Optional[any] = None
    wfe: Optional[WaveFrontError] = None
    
    def __post_init__(self):
        if MATERIAL_BEHAVIOUR_AVAILABLE:
            try:
                self.material = MaterialBehaviour(self.specifications.material_name)
            except Exception as e:
                logger.warning(f"Matériau inconnu: {self.specifications.material_name}. {e}")
                self.material = None
        else:
            self.material = None
        
        self.wfe = WaveFrontError(
            surface_roughness_nm=self.specifications.surface_roughness_nm,
            parallelism_arcsec=self.specifications.parallelism_arcsec
        )
        
        if self.display:
            os.makedirs(self.display_dir, exist_ok=True)
    
    @abstractmethod
    def get_phase_map(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        pass
    
    def get_aperture_mask(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        """FR: Génère un masque binaire pour l'aperture (toutes formes supportées)."""
        shape = self.specifications.aperture_shape
        clear_diameter = self.specifications.diameter_mm * self.specifications.clear_aperture_ratio
        
        x_center = self.decentering_mm[0]
        y_center = self.decentering_mm[1]
        x = grid_x_mm - x_center
        y = grid_y_mm - y_center
        
        if shape == ApertureShape.CIRCULAR:
            r = np.sqrt(x**2 + y**2)
            mask = (r <= clear_diameter / 2).astype(float)
        elif shape == ApertureShape.RECTANGULAR:
            width = self.specifications.width_mm or clear_diameter
            height = self.specifications.height_mm or clear_diameter
            mask = ((np.abs(x) <= width / 2) & (np.abs(y) <= height / 2)).astype(float)
        elif shape == ApertureShape.SQUARE:
            size = self.specifications.width_mm or clear_diameter
            mask = ((np.abs(x) <= size / 2) & (np.abs(y) <= size / 2)).astype(float)
        elif shape == ApertureShape.ELLIPTICAL:
            a = (self.specifications.width_mm or clear_diameter) / 2
            b = (self.specifications.height_mm or clear_diameter) / 2
            mask = ((x**2 / a**2) + (y**2 / b**2) <= 1.0).astype(float)
        elif shape == ApertureShape.HEXAGONAL:
            size = clear_diameter / 2
            mask = (np.abs(x) + np.abs(y) / np.sqrt(3) <= size * 2 / np.sqrt(3)).astype(float)
        elif shape == ApertureShape.OCTAGONAL:
            size = clear_diameter / 2
            mask = ((np.abs(x) + np.abs(y)) <= size * np.sqrt(2) * (1 + np.sqrt(2)) & 
                   (np.abs(x) <= size * (1 + np.sqrt(2))) & 
                   (np.abs(y) <= size * (1 + np.sqrt(2)))).astype(float)
        else:
            r = np.sqrt(x**2 + y**2)
            mask = (r <= clear_diameter / 2).astype(float)
        
        return mask
    
    def get_transmission_map(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray, 
                             angle_deg: float = 0.0, polarization: any = None) -> np.ndarray:
        if self.material is None:
            return self.get_aperture_mask(grid_x_mm, grid_y_mm)
        try:
            thickness_m = self.specifications.thickness_mm * 1e-3
            absorption = self.material._get_absorption_coefficient(self.wavelength_nm)
            transmission = np.exp(-absorption * thickness_m)
        except:
            transmission = 1.0
        return transmission * self.get_aperture_mask(grid_x_mm, grid_y_mm)
    
    def get_reflection_map(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray,
                           angle_deg: float = 0.0, polarization: any = None, n_medium: float = 1.0) -> np.ndarray:
        if self.material is None:
            return np.zeros_like(grid_x_mm)
        try:
            R = self.material.get_reflectance(self.wavelength_nm, angle_deg, polarization, n_medium)
        except:
            R = 0.0
        return R * self.get_aperture_mask(grid_x_mm, grid_y_mm)
    
    def get_full_phase_map(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray,
                           include_wfe: bool = True, include_tilt: bool = True, seed: Optional[int] = None) -> np.ndarray:
        phase_map = self.get_phase_map(grid_x_mm, grid_y_mm)
        if include_wfe and self.wfe is not None:
            phase_map += self.wfe.generate_phase_map(grid_x_mm, grid_y_mm, self.wavelength_nm, seed)
        if include_tilt and (self.tilt_deg[0] != 0 or self.tilt_deg[1] != 0):
            tilt_x_rad = np.deg2rad(self.tilt_deg[0])
            tilt_y_rad = np.deg2rad(self.tilt_deg[1])
            phase_map += (grid_x_mm * tilt_x_rad + grid_y_mm * tilt_y_rad) * self.wavelength_nm
        return phase_map * self.get_aperture_mask(grid_x_mm, grid_y_mm)
    
    def apply_to_beam(self, beam: any) -> any:
        if not BEAM_AVAILABLE:
            raise ImportError("Beam module is required for apply_to_beam().")
        
        if MATH_TOOLS_AVAILABLE:
            grid_x_mm, grid_y_mm = create_grid(beam.diameter_mm, beam.num_points)
        else:
            x = np.linspace(-beam.diameter_mm/2, beam.diameter_mm/2, beam.num_points)
            y = np.linspace(-beam.diameter_mm/2, beam.diameter_mm/2, beam.num_points)
            grid_x_mm, grid_y_mm = np.meshgrid(x, y)
        
        if self.display and VISUALIZATION_AVAILABLE:
            os.makedirs(self.display_dir, exist_ok=True)
            if hasattr(beam, 'intensity') and beam.intensity is not None:
                plot_intensity(beam.intensity, beam.diameter_mm, title=f"Before {self.name}",
                              save_path=os.path.join(self.display_dir, f"before_{self.name}_intensity.png"))
            if hasattr(beam, 'phase') and beam.phase is not None:
                plot_phase(beam.phase, beam.diameter_mm, title=f"Before {self.name}",
                          save_path=os.path.join(self.display_dir, f"before_{self.name}_phase.png"))
        
        phase_map = self.get_full_phase_map(grid_x_mm, grid_y_mm)
        new_beam = Beam(wavelength_nm=beam.wavelength_nm, diameter_mm=beam.diameter_mm,
                       energy=beam.energy, num_points=beam.num_points, coherence=beam.coherence)
        
        if beam.electric_field is not None:
            amplitude = np.abs(beam.electric_field)
            initial_phase = np.angle(beam.electric_field)
            phase_rad = phase_map * 2 * np.pi / self.wavelength_nm
            new_phase = initial_phase + phase_rad
            new_electric_field = amplitude * np.exp(1j * new_phase)
            new_beam.electric_field = new_electric_field
            new_beam.intensity = new_beam.compute_intensity_from_electric_field(new_electric_field)
            new_beam.phase = new_beam.extract_phase_from_electric_field(new_electric_field)
        
        if self.display and VISUALIZATION_AVAILABLE:
            if hasattr(new_beam, 'intensity') and new_beam.intensity is not None:
                plot_intensity(new_beam.intensity, new_beam.diameter_mm, title=f"After {self.name}",
                              save_path=os.path.join(self.display_dir, f"after_{self.name}_intensity.png"))
            if hasattr(new_beam, 'phase') and new_beam.phase is not None:
                plot_phase(new_beam.phase, new_beam.diameter_mm, title=f"After {self.name}",
                          save_path=os.path.join(self.display_dir, f"after_{self.name}_phase.png"))
        
        return new_beam
    
    def get_optical_path_difference(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        if self.material is None:
            return np.zeros_like(grid_x_mm)
        try:
            n = self.material.get_refractive_index(self.wavelength_nm, self.temperature_K)
        except:
            n = 1.5
        effective_thickness = self._get_effective_thickness(grid_x_mm, grid_y_mm)
        return (n - 1) * effective_thickness * 1e6
    
    @abstractmethod
    def _get_effective_thickness(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        pass


# =============================================================================
# 4. CLASSES SPÉCIFIQUES / SPECIFIC CLASSES
# =============================================================================

class IdealLens(OpticalElement):
    def __init__(self, name: str, focal_length_mm: float, diameter_mm: float,
                 material_name: str = "ideal", specifications: Optional[OpticSpecifications] = None, **kwargs):
        if specifications is None:
            specifications = OpticSpecifications(diameter_mm=diameter_mm, thickness_mm=0.0, material_name=material_name)
        self.focal_length_mm = focal_length_mm
        super().__init__(name=name, specifications=specifications, **kwargs)
    
    def get_phase_map(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        f_m = self.focal_length_mm * 1e-3
        x_m, y_m = grid_x_mm * 1e-3, grid_y_mm * 1e-3
        wavelength_m = self.wavelength_nm * 1e-9
        phase_rad = - (2 * np.pi / wavelength_m) * (x_m**2 + y_m**2) / (2 * f_m)
        return phase_rad * wavelength_m / (2 * np.pi) * 1e9
    
    def _get_effective_thickness(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        return np.full_like(grid_x_mm, 0.0)


class SimpleLens(OpticalElement):
    def __init__(self, name: str, radius_of_curvature_mm: float, diameter_mm: float,
                 thickness_mm: float, lens_type: LensType = LensType.PLAN_CONVEX,
                 material_name: str = "BK7", curved_face_position: str = "front",
                 specifications: Optional[OpticSpecifications] = None, **kwargs):
        if specifications is None:
            specifications = OpticSpecifications(diameter_mm=diameter_mm, thickness_mm=thickness_mm, material_name=material_name)
        self.radius_of_curvature_mm = radius_of_curvature_mm
        self.lens_type = lens_type
        self.curved_face_position = curved_face_position
        super().__init__(name=name, specifications=specifications, **kwargs)
    
    def get_phase_map(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        if self.material is None:
            return np.zeros_like(grid_x_mm)
        n = self.material.get_refractive_index(self.wavelength_nm, self.temperature_K)
        thickness = self._get_effective_thickness(grid_x_mm, grid_y_mm)
        return (n - 1) * thickness * 1e6
    
    def _get_effective_thickness(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        R = self.radius_of_curvature_mm
        d = self.specifications.thickness_mm
        r = np.sqrt(grid_x_mm**2 + grid_y_mm**2)
        if self.lens_type == LensType.PLAN_CONVEX:
            sign = -1 if self.curved_face_position == "front" else 1
        else:
            sign = 1 if self.curved_face_position == "front" else -1
        return np.maximum(d + sign * r**2 / (2 * abs(R)), 0.0)


class DoubleLens(OpticalElement):
    def __init__(self, name: str, radius_of_curvature_1_mm: float, radius_of_curvature_2_mm: float,
                 diameter_mm: float, thickness_mm: float, lens_type: LensType = LensType.BICONVEX,
                 material_name: str = "BK7", specifications: Optional[OpticSpecifications] = None, **kwargs):
        if specifications is None:
            specifications = OpticSpecifications(diameter_mm=diameter_mm, thickness_mm=thickness_mm, material_name=material_name)
        self.R1 = radius_of_curvature_1_mm
        self.R2 = radius_of_curvature_2_mm
        self.lens_type = lens_type
        super().__init__(name=name, specifications=specifications, **kwargs)
    
    def get_phase_map(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        if self.material is None:
            return np.zeros_like(grid_x_mm)
        n = self.material.get_refractive_index(self.wavelength_nm, self.temperature_K)
        thickness = self._get_effective_thickness(grid_x_mm, grid_y_mm)
        return (n - 1) * thickness * 1e6
    
    def _get_effective_thickness(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        R1, R2 = self.R1, self.R2
        d = self.specifications.thickness_mm
        r = np.sqrt(grid_x_mm**2 + grid_y_mm**2)
        if self.lens_type == LensType.BICONVEX:
            thickness = d - r**2 / (2 * R1) - r**2 / (2 * R2)
        elif self.lens_type == LensType.BICONCAVE:
            thickness = d + r**2 / (2 * abs(R1)) + r**2 / (2 * abs(R2))
        elif self.lens_type == LensType.MENISCUS:
            thickness = d - r**2 / (2 * R1) + r**2 / (2 * R2)
        else:
            thickness = np.full_like(grid_x_mm, d)
        return np.maximum(thickness, 0.0)


class DoubletLens(OpticalElement):
    def __init__(self, name: str, diameter_mm: float,
                 radius_of_curvature_1_mm: float, radius_of_curvature_2_mm: float,
                 thickness_1_mm: float, material_1_name: str = "BK7",
                 radius_of_curvature_3_mm: float, radius_of_curvature_4_mm: float,
                 thickness_2_mm: float, material_2_name: str = "SF5",
                 specifications: Optional[OpticSpecifications] = None, **kwargs):
        if specifications is None:
            specifications = OpticSpecifications(
                diameter_mm=diameter_mm,
                thickness_mm=thickness_1_mm + thickness_2_mm,
                material_name=f"{material_1_name}_{material_2_name}")
        super().__init__(name=name, specifications=specifications, **kwargs)
        self.R1 = radius_of_curvature_1_mm
        self.R2 = radius_of_curvature_2_mm
        self.R3 = radius_of_curvature_3_mm
        self.R4 = radius_of_curvature_4_mm
        self.d1 = thickness_1_mm
        self.d2 = thickness_2_mm
        
        if MATERIAL_BEHAVIOUR_AVAILABLE:
            try:
                self.material1 = MaterialBehaviour(material_1_name)
                self.material2 = MaterialBehaviour(material_2_name)
            except:
                self.material1 = None
                self.material2 = None
        else:
            self.material1 = None
            self.material2 = None
    
    def get_phase_map(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        if self.material1 is None or self.material2 is None:
            return np.zeros_like(grid_x_mm)
        n1 = self.material1.get_refractive_index(self.wavelength_nm, self.temperature_K)
        n2 = self.material2.get_refractive_index(self.wavelength_nm, self.temperature_K)
        thickness1 = self._get_effective_thickness_1(grid_x_mm, grid_y_mm)
        thickness2 = self._get_effective_thickness_2(grid_x_mm, grid_y_mm)
        return (n1 - 1) * thickness1 * 1e6 + (n2 - 1) * thickness2 * 1e6
    
    def _get_effective_thickness(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        return self._get_effective_thickness_1(grid_x_mm, grid_y_mm) + self._get_effective_thickness_2(grid_x_mm, grid_y_mm)
    
    def _get_effective_thickness_1(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        R1, R2 = self.R1, self.R2
        d1 = self.d1
        r = np.sqrt(grid_x_mm**2 + grid_y_mm**2)
        if R1 > 0 and R2 > 0:
            thickness = d1 - r**2 / (2 * R1) - r**2 / (2 * R2)
        elif R1 > 0 and R2 < 0:
            if abs(R2) > 1e6:
                thickness = d1 - r**2 / (2 * R1)
            else:
                thickness = d1 - r**2 / (2 * R1) + r**2 / (2 * abs(R2))
        else:
            thickness = np.full_like(grid_x_mm, d1)
        return np.maximum(thickness, 0.0)
    
    def _get_effective_thickness_2(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        R3, R4 = self.R3, self.R4
        d2 = self.d2
        r = np.sqrt(grid_x_mm**2 + grid_y_mm**2)
        if R3 < 0 and R4 < 0:
            thickness = d2 + r**2 / (2 * abs(R3)) + r**2 / (2 * abs(R4))
        elif R3 < 0 and abs(R4) > 1e6:
            thickness = d2 + r**2 / (2 * abs(R3))
        else:
            thickness = np.full_like(grid_x_mm, d2)
        return np.maximum(thickness, 0.0)
    
    @staticmethod
    def create_achromatic_doublet(focal_length_mm: float, diameter_mm: float,
                                  material1_name: str = "BK7", material2_name: str = "SF5",
                                  wavelength_nm: float = 633.0, **kwargs) -> 'DoubletLens':
        if not MATERIAL_BEHAVIOUR_AVAILABLE:
            raise ImportError("Material_Behaviour required for achromatic doublet")
        mat1 = MaterialBehaviour(material1_name)
        mat2 = MaterialBehaviour(material2_name)
        n1 = mat1.get_refractive_index(wavelength_nm)
        n2 = mat2.get_refractive_index(wavelength_nm)
        V1, V2 = 64.0, 32.0
        P = 1 / (focal_length_mm * 1e-3)
        P1 = P * V1 / (V1 - V2)
        P2 = P * V2 / (V2 - V1)
        R1 = 2 * (n1 - 1) / P1
        R2 = -R1
        R3 = -R2
        R4 = 2 * (n2 - 1) / P2
        return DoubletLens(
            name=f"Achromatic_Doublet_f{focal_length_mm}mm", diameter_mm=diameter_mm,
            radius_of_curvature_1_mm=R1, radius_of_curvature_2_mm=R2, thickness_1_mm=3.0,
            material_1_name=material1_name, radius_of_curvature_3_mm=R3, radius_of_curvature_4_mm=R4,
            thickness_2_mm=2.0, material_2_name=material2_name, wavelength_nm=wavelength_nm, **kwargs)


class Mirror(OpticalElement):
    def __init__(self, name: str, diameter_mm: float, mirror_type: MirrorType = MirrorType.FLAT,
                 radius_of_curvature_mm: Optional[float] = None, focal_length_mm: Optional[float] = None,
                 material_name: str = "Aluminum", coating_reflectivity: float = 0.95,
                 specifications: Optional[OpticSpecifications] = None, **kwargs):
        if specifications is None:
            specifications = OpticSpecifications(diameter_mm=diameter_mm, thickness_mm=10.0, material_name=material_name)
        super().__init__(name=name, specifications=specifications, **kwargs)
        self.mirror_type = mirror_type
        self.radius_of_curvature_mm = radius_of_curvature_mm
        self.focal_length_mm = focal_length_mm
        self.coating_reflectivity = coating_reflectivity
        if self.focal_length_mm is not None and self.radius_of_curvature_mm is None:
            if mirror_type == MirrorType.SPHERICAL:
                self.radius_of_curvature_mm = 2 * self.focal_length_mm
            elif mirror_type == MirrorType.PARABOLIC:
                self.radius_of_curvature_mm = 2 * self.focal_length_mm
    
    def get_phase_map(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        if self.mirror_type == MirrorType.FLAT:
            return np.zeros_like(grid_x_mm)
        elif self.mirror_type == MirrorType.SPHERICAL:
            R = self.radius_of_curvature_mm
            if R is None:
                return np.zeros_like(grid_x_mm)
            return -2 * (grid_x_mm**2 + grid_y_mm**2) / R * 1e6
        elif self.mirror_type == MirrorType.PARABOLIC:
            f = self.focal_length_mm
            if f is None:
                return np.zeros_like(grid_x_mm)
            return - (grid_x_mm**2 + grid_y_mm**2) / f * 1e6
        return np.zeros_like(grid_x_mm)
    
    def get_reflection_map(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray,
                           angle_deg: float = 0.0, polarization: any = None, n_medium: float = 1.0) -> np.ndarray:
        return self.coating_reflectivity * self.get_aperture_mask(grid_x_mm, grid_y_mm)
    
    def _get_effective_thickness(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        return np.full_like(grid_x_mm, self.specifications.thickness_mm)


class Beamsplitter(OpticalElement):
    def __init__(self, name: str, diameter_mm: float,
                 beamsplitter_type: BeamsplitterType = BeamsplitterType.PLATE,
                 transmission_ratio: float = 0.5, reflection_ratio: float = 0.5,
                 material_name: str = "BK7", polarization_axis: Optional[str] = None,
                 specifications: Optional[OpticSpecifications] = None, **kwargs):
        if specifications is None:
            specifications = OpticSpecifications(diameter_mm=diameter_mm, thickness_mm=1.0, material_name=material_name)
        super().__init__(name=name, specifications=specifications, **kwargs)
        self.beamsplitter_type = beamsplitter_type
        total = transmission_ratio + reflection_ratio
        self.transmission_ratio = transmission_ratio / total if total > 0 else 0.5
        self.reflection_ratio = reflection_ratio / total if total > 0 else 0.5
        self.polarization_axis = polarization_axis
    
    def get_phase_map(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        if self.material is None:
            return np.zeros_like(grid_x_mm)
        n = self.material.get_refractive_index(self.wavelength_nm, self.temperature_K)
        thickness = self._get_effective_thickness(grid_x_mm, grid_y_mm)
        return (n - 1) * thickness * 1e6
    
    def get_transmission_map(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray,
                             angle_deg: float = 0.0, polarization: any = None) -> np.ndarray:
        mask = self.get_aperture_mask(grid_x_mm, grid_y_mm)
        if self.beamsplitter_type == BeamsplitterType.POLARIZING and self.polarization_axis:
            return mask * (0.5 if polarization is None else 1.0)
        return self.transmission_ratio * mask
    
    def get_reflection_map(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray,
                           angle_deg: float = 0.0, polarization: any = None, n_medium: float = 1.0) -> np.ndarray:
        mask = self.get_aperture_mask(grid_x_mm, grid_y_mm)
        if self.beamsplitter_type == BeamsplitterType.POLARIZING and self.polarization_axis:
            return mask * (0.5 if polarization is None else 1.0)
        return self.reflection_ratio * mask
    
    def _get_effective_thickness(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        return np.full_like(grid_x_mm, self.specifications.thickness_mm)


class Window(OpticalElement):
    def __init__(self, name: str, diameter_mm: float, thickness_mm: float,
                 material_name: str = "Fused_Silica", specifications: Optional[OpticSpecifications] = None, **kwargs):
        if specifications is None:
            specifications = OpticSpecifications(diameter_mm=diameter_mm, thickness_mm=thickness_mm, material_name=material_name)
        super().__init__(name=name, specifications=specifications, **kwargs)
    
    def get_phase_map(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        if self.material is None:
            return np.zeros_like(grid_x_mm)
        n = self.material.get_refractive_index(self.wavelength_nm, self.temperature_K)
        thickness = self.specifications.thickness_mm
        phase = (n - 1) * thickness * 1e6
        if self.tilt_deg[0] != 0 or self.tilt_deg[1] != 0:
            tilt_x_rad = np.deg2rad(self.tilt_deg[0])
            tilt_y_rad = np.deg2rad(self.tilt_deg[1])
            phase += (n - 1) * (grid_x_mm * tilt_x_rad + grid_y_mm * tilt_y_rad) * 1e3
        return phase
    
    def _get_effective_thickness(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        return np.full_like(grid_x_mm, self.specifications.thickness_mm)


class Prism(OpticalElement):
    def __init__(self, name: str, apex_angle_deg: float, base_length_mm: float, height_mm: float,
                 material_name: str = "BK7", orientation_deg: float = 0.0,
                 specifications: Optional[OpticSpecifications] = None, **kwargs):
        diameter_mm = np.sqrt(base_length_mm**2 + height_mm**2)
        if specifications is None:
            specifications = OpticSpecifications(
                diameter_mm=diameter_mm, thickness_mm=height_mm, material_name=material_name,
                aperture_shape=ApertureShape.RECTANGULAR, width_mm=base_length_mm, height_mm=height_mm)
        super().__init__(name=name, specifications=specifications, **kwargs)
        self.apex_angle_deg = apex_angle_deg
        self.apex_angle_rad = np.deg2rad(apex_angle_deg)
        self.base_length_mm = base_length_mm
        self.height_mm = height_mm
        self.orientation_deg = orientation_deg
        self.orientation_rad = np.deg2rad(orientation_deg)
    
    def get_phase_map(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        if self.material is None:
            return np.zeros_like(grid_x_mm)
        n = self.material.get_refractive_index(self.wavelength_nm, self.temperature_K)
        thickness = self._get_effective_thickness(grid_x_mm, grid_y_mm)
        return (n - 1) * thickness * 1e6
    
    def get_deviation_angle(self, wavelength_nm: Optional[float] = None) -> float:
        if wavelength_nm is None:
            wavelength_nm = self.wavelength_nm
        if self.material is None:
            return 0.0
        n = self.material.get_refractive_index(wavelength_nm, self.temperature_K)
        return np.rad2deg((n - 1) * self.apex_angle_rad)
    
    def _get_effective_thickness(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        angle = self.orientation_rad
        y_prism = grid_x_mm * np.sin(angle) + grid_y_mm * np.cos(angle)
        e0 = self.height_mm / 2
        e_mm = e0 - y_prism * np.tan(self.apex_angle_rad / 2)
        return np.maximum(e_mm, 0.0)


class AsphericLens(OpticalElement):
    def __init__(self, name: str, radius_of_curvature_mm: float, diameter_mm: float,
                 thickness_mm: float, conic_constant: float = 0.0,
                 aspheric_coefficients: Optional[Dict[int, float]] = None,
                 material_name: str = "BK7", specifications: Optional[OpticSpecifications] = None, **kwargs):
        if specifications is None:
            specifications = OpticSpecifications(diameter_mm=diameter_mm, thickness_mm=thickness_mm, material_name=material_name)
        super().__init__(name=name, specifications=specifications, **kwargs)
        self.radius_of_curvature_mm = radius_of_curvature_mm
        self.conic_constant = conic_constant
        self.aspheric_coefficients = aspheric_coefficients or {}
    
    def get_phase_map(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        if self.material is None:
            return np.zeros_like(grid_x_mm)
        n = self.material.get_refractive_index(self.wavelength_nm, self.temperature_K)
        thickness = self._get_effective_thickness(grid_x_mm, grid_y_mm)
        return (n - 1) * thickness * 1e6
    
    def _get_effective_thickness(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        r = np.sqrt(grid_x_mm**2 + grid_y_mm**2)
        R = self.radius_of_curvature_mm
        with np.errstate(divide='ignore', invalid='ignore'):
            term1 = (r**2 / R) / (1 + np.sqrt(1 - (1 + self.conic_constant) * (r**2 / R**2)))
            sag = term1.copy()
            for i, a_i in self.aspheric_coefficients.items():
                sag += a_i * (r ** i)
            sag = np.nan_to_num(sag, nan=0.0)
        d = self.specifications.thickness_mm
        return np.maximum(d - sag, 0.0)


class ApertureStop(OpticalElement):
    """FR: Diaphragme (stop d'ouverture). Bloque la lumière en dehors de l'aperture."""
    
    def __init__(self, name: str, diameter_mm: float,
                 material_name: str = "opaque",
                 specifications: Optional[OpticSpecifications] = None, **kwargs):
        if specifications is None:
            specifications = OpticSpecifications(
                diameter_mm=diameter_mm,
                thickness_mm=0.0,
                material_name=material_name,
                aperture_shape=kwargs.get('aperture_shape', ApertureShape.CIRCULAR),
                width_mm=kwargs.get('width_mm'),
                height_mm=kwargs.get('height_mm'),
            )
        super().__init__(name=name, specifications=specifications, **kwargs)
    
    def get_phase_map(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        return np.zeros_like(grid_x_mm)
    
    def get_transmission_map(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray,
                             angle_deg: float = 0.0, polarization: any = None) -> np.ndarray:
        return self.get_aperture_mask(grid_x_mm, grid_y_mm)
    
    def _get_effective_thickness(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        return np.zeros_like(grid_x_mm)


class DiffractionGrating(OpticalElement):
    """FR: Réseau de diffraction. Crée une modulation de phase périodique."""
    
    def __init__(self, name: str, lines_per_mm: float, diameter_mm: float,
                 orientation_deg: float = 0.0,
                 grating_type: GratingType = GratingType.TRANSMISSION,
                 material_name: str = "Fused_Silica",
                 specifications: Optional[OpticSpecifications] = None, **kwargs):
        if specifications is None:
            specifications = OpticSpecifications(
                diameter_mm=diameter_mm,
                thickness_mm=0.1,
                material_name=material_name,
            )
        super().__init__(name=name, specifications=specifications, **kwargs)
        self.lines_per_mm = lines_per_mm
        self.orientation_deg = orientation_deg
        self.orientation_rad = np.deg2rad(orientation_deg)
        self.grating_type = grating_type
        self.pitch_mm = 1.0 / lines_per_mm
    
    def get_phase_map(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        if self.material is None:
            return np.zeros_like(grid_x_mm)
        n = self.material.get_refractive_index(self.wavelength_nm, self.temperature_K)
        x_rot = grid_x_mm * np.cos(self.orientation_rad) - grid_y_mm * np.sin(self.orientation_rad)
        phase_nm = (n - 1) * self.specifications.thickness_mm * 1e6 * np.sin(2 * np.pi * x_rot / self.pitch_mm)
        return phase_nm
    
    def get_diffraction_orders(self, max_order: int = 5) -> List[Optional[float]]:
        """FR: Calcule les angles de diffraction pour différents ordres."""
        wavelength_mm = self.wavelength_nm * 1e-6
        orders = []
        for m in range(-max_order, max_order + 1):
            if m == 0:
                orders.append(0.0)
            else:
                sin_theta = m * wavelength_mm / self.pitch_mm
                if abs(sin_theta) <= 1.0:
                    orders.append(np.rad2deg(np.arcsin(sin_theta)))
                else:
                    orders.append(None)
        return orders
    
    def get_diffraction_efficiency(self, order: int) -> float:
        """FR: Calcule l'efficacité de diffraction pour un ordre donné."""
        if self.material is None:
            return 0.0
        try:
            n = self.material.get_refractive_index(self.wavelength_nm, self.temperature_K)
            d = self.specifications.thickness_mm * 1e-3
            wavelength_m = self.wavelength_nm * 1e-9
            phi_0 = (2 * np.pi / wavelength_m) * (n - 1) * d
            from scipy.special import jv
            return jv(order, phi_0)**2
        except:
            return 0.0
    
    def _get_effective_thickness(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        return np.full_like(grid_x_mm, self.specifications.thickness_mm)


# =============================================================================
# 5. FONCTIONS UTILITAIRES
# =============================================================================

def create_optic(optic_type: OpticType, name: str = "Optic", **kwargs) -> OpticalElement:
    """FR: Fabrique une optique de type spécifié."""
    optic_classes = {
        OpticType.IDEAL_LENS: IdealLens,
        OpticType.SIMPLE_LENS: SimpleLens,
        OpticType.DOUBLE_LENS: DoubleLens,
        OpticType.DOUBLET_LENS: DoubletLens,
        OpticType.MIRROR: Mirror,
        OpticType.BEAMSPLITTER: Beamsplitter,
        OpticType.WINDOW: Window,
        OpticType.PRISM: Prism,
        OpticType.ASPHERIC_LENS: AsphericLens,
        OpticType.APERTURE_STOP: ApertureStop,
        OpticType.DIFFRACTION_GRATING: DiffractionGrating,
    }
    if optic_type not in optic_classes:
        raise ValueError(f"Type d'optique inconnu: {optic_type}")
    return optic_classes[optic_type](name=name, **kwargs)


def create_lens_from_preset(preset_name: str, diameter_mm: float, focal_length_mm: float,
                              material_name: str = "BK7", **kwargs) -> OpticalElement:
    """FR: Crée une lentille à partir d'une configuration prédéfinie."""
    presets = {
        "ideal": {"type": OpticType.IDEAL_LENS},
        "plan_convex": {"type": OpticType.SIMPLE_LENS, "lens_type": LensType.PLAN_CONVEX, "curved_face_position": "front"},
        "biconvex": {"type": OpticType.DOUBLE_LENS, "lens_type": LensType.BICONVEX},
        "achromatic_doublet": {"type": OpticType.DOUBLET_LENS, "factory": DoubletLens.create_achromatic_doublet},
        "window": {"type": OpticType.WINDOW, "thickness_mm": 2.0},
        "prism_45": {"type": OpticType.PRISM, "apex_angle_deg": 45.0, "base_length_mm": 10.0, "height_mm": 10.0},
        "prism_60": {"type": OpticType.PRISM, "apex_angle_deg": 60.0, "base_length_mm": 10.0, "height_mm": 10.0},
        "mirror_flat": {"type": OpticType.MIRROR, "mirror_type": MirrorType.FLAT},
        "beamsplitter": {"type": OpticType.BEAMSPLITTER, "beamsplitter_type": BeamsplitterType.PLATE},
        "aperture_stop": {"type": OpticType.APERTURE_STOP},
        "grating_100": {"type": OpticType.DIFFRACTION_GRATING, "lines_per_mm": 100.0},
        "grating_500": {"type": OpticType.DIFFRACTION_GRATING, "lines_per_mm": 500.0},
    }
    if preset_name not in presets:
        raise ValueError(f"Preset inconnu: {preset_name}")
    preset = presets[preset_name]
    optic_type = preset["type"]
    optic_kwargs = {"name": f"{preset_name}_{diameter_mm}mm", "diameter_mm": diameter_mm, **kwargs}
    for key, value in preset.items():
        if key not in ["type", "factory"]:
            optic_kwargs[key] = value
    if "factory" in preset:
        return preset["factory"](focal_length_mm=focal_length_mm, diameter_mm=diameter_mm, **optic_kwargs)
    if optic_type == OpticType.IDEAL_LENS:
        optic_kwargs["focal_length_mm"] = focal_length_mm
    elif optic_type in [OpticType.SIMPLE_LENS, OpticType.DOUBLE_LENS]:
        n = 1.5
        if MATERIAL_BEHAVIOUR_AVAILABLE:
            try:
                mat = MaterialBehaviour(material_name)
                n = mat.get_refractive_index(633.0)
            except:
                pass
        if optic_type == OpticType.SIMPLE_LENS:
            optic_kwargs["radius_of_curvature_mm"] = (n - 1) * focal_length_mm
        else:
            R = 2 * (n - 1) * focal_length_mm
            optic_kwargs["radius_of_curvature_1_mm"] = R
            optic_kwargs["radius_of_curvature_2_mm"] = -R
    return create_optic(optic_type, **optic_kwargs)


# =============================================================================
# 6. CLASSE OpticalSystem
# =============================================================================

class OpticalSystem:
    """FR: Système optique composé de plusieurs éléments."""
    
    def __init__(self, display: bool = False, display_dir: str = "output"):
        self.elements: List[OpticalElement] = []
        self.display = display
        self.display_dir = display_dir
        if self.display:
            os.makedirs(self.display_dir, exist_ok=True)
    
    def add_element(self, element: OpticalElement) -> None:
        self.elements.append(element)
    
    def add_lens(self, name: str, focal_length_mm: float, diameter_mm: float,
                 material_name: str = "BK7", position_z_mm: float = 0.0, **kwargs) -> OpticalElement:
        lens = IdealLens(name=name, focal_length_mm=focal_length_mm, diameter_mm=diameter_mm,
                        material_name=material_name, position_mm=(0.0, 0.0, position_z_mm), **kwargs)
        self.add_element(lens)
        return lens
    
    def add_mirror(self, name: str, diameter_mm: float, mirror_type: MirrorType = MirrorType.FLAT,
                   radius_of_curvature_mm: Optional[float] = None, focal_length_mm: Optional[float] = None,
                   position_z_mm: float = 0.0, tilt_deg: Tuple[float, float] = (0.0, 0.0), **kwargs) -> OpticalElement:
        mirror = Mirror(name=name, diameter_mm=diameter_mm, mirror_type=mirror_type,
                       radius_of_curvature_mm=radius_of_curvature_mm, focal_length_mm=focal_length_mm,
                       position_mm=(0.0, 0.0, position_z_mm), tilt_deg=tilt_deg, **kwargs)
        self.add_element(mirror)
        return mirror
    
    def add_prism(self, name: str, apex_angle_deg: float, base_length_mm: float, height_mm: float,
                  material_name: str = "BK7", position_z_mm: float = 0.0, orientation_deg: float = 0.0, **kwargs) -> OpticalElement:
        prism = Prism(name=name, apex_angle_deg=apex_angle_deg, base_length_mm=base_length_mm,
                     height_mm=height_mm, material_name=material_name, position_mm=(0.0, 0.0, position_z_mm),
                     orientation_deg=orientation_deg, **kwargs)
        self.add_element(prism)
        return prism
    
    def add_aperture_stop(self, name: str, diameter_mm: float, position_z_mm: float = 0.0,
                          aperture_shape: ApertureShape = ApertureShape.CIRCULAR,
                          width_mm: Optional[float] = None, height_mm: Optional[float] = None, **kwargs) -> OpticalElement:
        specs = OpticSpecifications(
            diameter_mm=diameter_mm, thickness_mm=0.0, material_name="opaque",
            aperture_shape=aperture_shape, width_mm=width_mm, height_mm=height_mm)
        aperture = ApertureStop(name=name, diameter_mm=diameter_mm, specifications=specs,
                                position_mm=(0.0, 0.0, position_z_mm), **kwargs)
        self.add_element(aperture)
        return aperture
    
    def add_diffraction_grating(self, name: str, lines_per_mm: float, diameter_mm: float,
                                position_z_mm: float = 0.0, orientation_deg: float = 0.0,
                                grating_type: GratingType = GratingType.TRANSMISSION, **kwargs) -> OpticalElement:
        grating = DiffractionGrating(name=name, lines_per_mm=lines_per_mm, diameter_mm=diameter_mm,
                                     orientation_deg=orientation_deg, grating_type=grating_type,
                                     position_mm=(0.0, 0.0, position_z_mm), **kwargs)
        self.add_element(grating)
        return grating
    
    def sort_elements_by_position(self) -> None:
        self.elements.sort(key=lambda x: x.position_mm[2])
    
    def propagate_beam(self, beam: any, initial_position_mm: float = 0.0, use_propagation: bool = True) -> any:
        """FR: Propage un faisceau à travers le système optique."""
        if not BEAM_AVAILABLE:
            raise ImportError("Beam module is required for propagate_beam().")
        self.sort_elements_by_position()
        current_position_z = initial_position_mm
        current_beam = beam
        if self.display and VISUALIZATION_AVAILABLE:
            os.makedirs(self.display_dir, exist_ok=True)
            if hasattr(current_beam, 'intensity') and current_beam.intensity is not None:
                plot_intensity(current_beam.intensity, current_beam.diameter_mm, title="Initial Beam",
                              save_path=os.path.join(self.display_dir, "initial_beam.png"))
        for idx, element in enumerate(self.elements):
            element_z = element.position_mm[2]
            distance_mm = element_z - current_position_z
            if distance_mm > 0 and use_propagation and PROPAGATION_AVAILABLE:
                propagator = Propagation(
                    wavelength_nm=current_beam.wavelength_nm, propagation_distance_mm=distance_mm,
                    input_diameter_mm=current_beam.diameter_mm, output_diameter_mm=current_beam.diameter_mm,
                    num_points=current_beam.num_points, method="angular_spectrum", coherence=current_beam.coherence)
                propagated_field = propagator.propagate(current_beam.electric_field)
                current_beam.electric_field = propagated_field
                current_beam.intensity = current_beam.compute_intensity_from_electric_field(propagated_field)
                current_beam.phase = current_beam.extract_phase_from_electric_field(propagated_field)
                if self.display and VISUALIZATION_AVAILABLE:
                    if hasattr(current_beam, 'intensity') and current_beam.intensity is not None:
                        plot_intensity(current_beam.intensity, current_beam.diameter_mm,
                                      title=f"After Propagation to {element.name}",
                                      save_path=os.path.join(self.display_dir, f"after_prop_to_{element.name}.png"))
            current_beam = element.apply_to_beam(current_beam)
            current_position_z = element_z
        if self.display and VISUALIZATION_AVAILABLE:
            if hasattr(current_beam, 'intensity') and current_beam.intensity is not None:
                plot_intensity(current_beam.intensity, current_beam.diameter_mm, title="Final Beam",
                              save_path=os.path.join(self.display_dir, "final_beam.png"))
        return current_beam
    
    def get_total_phase_map(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        """FR: Calcule la phase totale du système."""
        total_phase = np.zeros_like(grid_x_mm)
        for element in self.elements:
            total_phase += element.get_full_phase_map(grid_x_mm, grid_y_mm)
        return total_phase


# =============================================================================
# 7. TESTS UNITAIRES
# =============================================================================

class TestOptiques:
    """FR: Tests unitaires pour Optiques.py."""
    
    def test_ideal_lens(self):
        lens = IdealLens(name="Test", focal_length_mm=100.0, diameter_mm=10.0)
        grid = self._create_grid(10.0, 256)
        phase = lens.get_phase_map(*grid)
        assert np.all(phase <= 0), "Ideal lens phase should be negative"
        center = phase.shape[0] // 2, phase.shape[1] // 2
        assert abs(phase[center, center]) < 1e-6, "Center phase should be zero"
    
    def test_simple_lens(self):
        lens = SimpleLens(name="Test", radius_of_curvature_mm=100.0, diameter_mm=10.0,
                         thickness_mm=5.0, lens_type=LensType.PLAN_CONVEX, material_name="BK7")
        grid = self._create_grid(10.0, 256)
        phase = lens.get_phase_map(*grid)
        assert np.all(phase >= 0), "Simple lens phase should be positive"
    
    def test_doublet_lens(self):
        doublet = DoubletLens(name="Test", diameter_mm=10.0,
                              radius_of_curvature_1_mm=100.0, radius_of_curvature_2_mm=-50.0,
                              thickness_1_mm=3.0, material_1_name="BK7",
                              radius_of_curvature_3_mm=50.0, radius_of_curvature_4_mm=-200.0,
                              thickness_2_mm=2.0, material_2_name="SF5")
        grid = self._create_grid(10.0, 256)
        phase = doublet.get_phase_map(*grid)
        assert phase.shape == grid[0].shape
        assert not np.all(phase == 0)
    
    def test_prism(self):
        prism = Prism(name="Test", apex_angle_deg=45.0, base_length_mm=10.0, height_mm=10.0, material_name="BK7")
        deviation = prism.get_deviation_angle()
        assert 40.0 < deviation < 50.0, f"Expected ~45°, got {deviation:.2f}°"
    
    def test_aperture_stop(self):
        aperture = ApertureStop(name="Test", diameter_mm=5.0, aperture_shape=ApertureShape.CIRCULAR)
        grid = self._create_grid(10.0, 256)
        phase = aperture.get_phase_map(*grid)
        assert np.all(phase == 0), "Aperture stop phase should be zero"
        transmission = aperture.get_transmission_map(*grid)
        assert np.all((transmission == 0) | (transmission == 1)), "Transmission should be binary"
    
    def test_diffraction_grating(self):
        grating = DiffractionGrating(name="Test", lines_per_mm=100.0, diameter_mm=10.0)
        assert abs(grating.pitch_mm - 0.01) < 1e-6, "Pitch should be 0.01 mm for 100 lines/mm"
        orders = grating.get_diffraction_orders(max_order=2)
        assert len(orders) == 5
        assert orders[2] == 0.0
    
    def test_aperture_shapes(self):
        shapes = [ApertureShape.CIRCULAR, ApertureShape.RECTANGULAR, ApertureShape.SQUARE,
                  ApertureShape.ELLIPTICAL, ApertureShape.HEXAGONAL, ApertureShape.OCTAGONAL]
        for shape in shapes:
            specs = OpticSpecifications(diameter_mm=10.0, thickness_mm=2.0, aperture_shape=shape,
                                         width_mm=8.0, height_mm=6.0)
            optic = Window(name=f"Test_{shape.value}", diameter_mm=10.0, thickness_mm=2.0, specifications=specs)
            grid = self._create_grid(20.0, 256)
            mask = optic.get_aperture_mask(*grid)
            assert mask.shape == grid[0].shape
            assert np.any(mask > 0), f"Mask for {shape.value} should have non-zero values"
    
    def test_achromatic_doublet(self):
        if not MATERIAL_BEHAVIOUR_AVAILABLE:
            return
        doublet = DoubletLens.create_achromatic_doublet(focal_length_mm=100.0, diameter_mm=25.0)
        assert isinstance(doublet, DoubletLens)
        assert doublet.R1 is not None
    
    def test_optical_system_propagation(self):
        if not BEAM_AVAILABLE:
            return
        system = OpticalSystem()
        system.add_lens(name="Lens", focal_length_mm=100.0, diameter_mm=10.0, position_z_mm=50.0)
        beam = Beam(wavelength_nm=633.0, diameter_mm=10.0, num_points=256)
        beam.electric_field = beam.generate_electric_field(method="gaussian")
        propagated_beam = system.propagate_beam(beam, initial_position_mm=0.0, use_propagation=False)
        assert propagated_beam is not beam
    
    def _create_grid(self, size_mm: float, num_points: int) -> Tuple[np.ndarray, np.ndarray]:
        if MATH_TOOLS_AVAILABLE:
            return create_grid(size_mm, num_points)
        x = y = np.linspace(-size_mm/2, size_mm/2, num_points)
        return np.meshgrid(x, y)


if __name__ == "__main__":
    import unittest
    unittest.main()

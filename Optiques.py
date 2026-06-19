"""
Optiques.py
FR: Module pour la génération et la gestion d'optiques diverses (réelles ou parfaites).
    Permet de générer des cartes de phase pour :
    - Lentilles parfaites (paraxiales)
    - Lentilles réelles (plan-convexe, biconvexe, biconcave, plan-concave, ménisque, doublets collés)
    - Miroirs (plats, sphériques, paraboliques)
    - Beamsplitters (cubes, lames)
    - Fenêtres optiques (plates, inclinées)
    - Prismes (déviation du faisceau)
    
    Fonctionnalités clés :
    - Intégration avec Material_Behaviour.py pour les propriétés des matériaux
    - Ajout d'erreurs de front d'onde (WFE) : qualité de surface, parallélisme, aberrations
    - Positionnement : tilt (inclinaison) et décentrement
    - Masquage des zones non couvertes par l'optique
    - Caractéristiques mécaniques complètes (forme, épaisseur, rayons de courbure)
    - Formes géométriques variées (circulaire, rectangulaire, elliptique, hexagonale, etc.)
    - Affichage automatique des cartes d'intensité et de phase

EN: Module for generating and managing various optical elements (ideal or real).
    Allows generating phase maps for:
    - Ideal lenses (paraxial)
    - Real lenses (plan-convex, biconvex, biconcave, plan-concave, meniscus, cemented doublets)
    - Mirrors (flat, spherical, parabolic)
    - Beamsplitters (cube, plate)
    - Optical windows (flat, tilted)
    - Prisms (beam deviation)
    
    Key features:
    - Integration with Material_Behaviour.py for material properties
    - Wavefront error (WFE) addition: surface roughness, parallelism, aberrations
    - Positioning: tilt and decentering
    - Masking of areas not covered by the optic
    - Complete mechanical characteristics (shape, thickness, curvature radii)
    - Various geometric shapes (circular, rectangular, elliptical, hexagonal, etc.)
    - Automatic display of intensity and phase maps

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch

Sources:
    - Optical design principles from "Optics" by E. Hecht
    - Lensmaker's equation and paraxial optics
    - Zernike polynomials for aberration representation
"""

import numpy as np
import logging
from typing import Optional, Tuple, Dict, Union, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

# Importer les dépendances locales
try:
    from Material_Behaviour import MaterialBehaviour, STANDARD_TEMPERATURE_K, Polarization
    from Beam import Beam
    from MathAndPhysicsTools import create_grid, zernike_polynomial, normalize_zernike_coefficients
    MATERIAL_BEHAVIOUR_AVAILABLE = True
except ImportError:
    MATERIAL_BEHAVIOUR_AVAILABLE = False
    logging.warning("Material_Behaviour module not available. Some features will be limited.")

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Optiques")


# =============================================================================
# 1. ENUMS ET CONSTANTES / ENUMS AND CONSTANTS
# =============================================================================

class OpticType(Enum):
    """FR: Type d'optique."""
    IDEAL_LENS = "ideal_lens"           # Lentille paraxiale parfaite
    SIMPLE_LENS = "simple_lens"         # Lentille simple (1 face courbe)
    DOUBLE_LENS = "double_lens"        # Lentille à 2 faces courbes
    CEMENTED_DOUBLET = "cemented_doublet"  # Doublet collé
    MIRROR = "mirror"                   # Miroir
    BEAMSPLITTER = "beamsplitter"       # Séparateur de faisceau
    WINDOW = "window"                   # Fenêtre optique
    PRISM = "prism"                     # Prisme
    ASPHERIC_LENS = "aspheric_lens"    # Lentille asphérique


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


class PrismType(Enum):
    """FR: Type de prisme."""
    RIGHT_ANGLE = "right_angle"         # Prisme à angle droit
    EQUILATERAL = "equilateral"         # Prisme équilatéral
    DISPERSING = "dispersing"           # Prisme dispersif
    ROOF = "roof"                       # Prisme en toit


class ShapeType(Enum):
    """FR: Type de forme géométrique."""
    CIRCULAR = "circular"               # Circulaire
    RECTANGULAR = "rectangular"         # Rectangulaire
    SQUARE = "square"                   # Carré
    ELLIPTICAL = "elliptical"           # Elliptique
    HEXAGONAL = "hexagonal"             # Hexagonal
    OCTAGONAL = "octagonal"             # Octogonal
    CUSTOM = "custom"                   # Forme personnalisée


class WFESource(Enum):
    """FR: Source des erreurs de front d'onde."""
    SURFACE_ROUGHNESS = "surface_roughness"  # Rugosité de surface
    PARALLELISM = "parallelism"              # Parallélisme
    ZERNIKE = "zernike"                      # Polynômes de Zernike
    CUSTOM = "custom"                        # Carte personnalisée
    FILE = "file"                          # Depuis un fichier


# Constantes physiques
C_LIGHT_M_S = 299792458  # Vitesse de la lumière en m/s


# =============================================================================
# 2. CLASSES POUR LES ERREURS DE FRONT D'ONDE / WAVEFRONT ERROR CLASSES
# =============================================================================

@dataclass
class WaveFrontError:
    """
    FR: Classe représentant les erreurs de front d'onde (WFE) d'une optique.
        Permet de définir des erreurs de surface, de parallélisme, des aberrations,
        ou une carte de phase personnalisée.

    EN: Class representing wavefront errors (WFE) of an optical element.
        Allows defining surface roughness, parallelism errors, aberrations,
        or a custom phase map.

    Attributes:
        surface_roughness_nm (float): Rugosité de surface en nm (RMS).
        parallelism_arcsec (float): Erreur de parallélisme en secondes d'arc.
        zernike_coefficients (Dict): Coefficients des polynômes de Zernike (format: {(n,m): coeff_nm}).
        custom_phase_map (np.ndarray): Carte de phase personnalisée en nm.
        wfe_source (WFESource): Source principale des erreurs.
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
        """
        FR: Génère une carte de phase représentant les erreurs de front d'onde.

        EN: Generates a phase map representing wavefront errors.

        Args:
            grid_x_mm (np.ndarray): Grille de positions en x (mm).
            grid_y_mm (np.ndarray): Grille de positions en y (mm).
            wavelength_nm (float): Longueur d'onde en nm.
            seed (int, optional): Graine pour la génération aléatoire.

        Returns:
            np.ndarray: Carte de phase en nm.
        """
        if seed is not None:
            np.random.seed(seed)
        
        phase_map = np.zeros_like(grid_x_mm)
        
        # 1. Ajouter la rugosité de surface (phase aléatoire haute fréquence)
        if self.surface_roughness_nm > 0:
            # Générer une carte de phase aléatoire avec une rugosité RMS donnée
            # Utiliser un bruit blanc filtré pour simuler une rugosité réaliste
            phase_rms = self.surface_roughness_nm
            
            # Générer un bruit avec une distribution normale
            noise = np.random.normal(0, phase_rms, grid_x_mm.shape)
            
            # Appliquer un filtre passe-haut pour simuler une rugosité de surface
            # (Les basses fréquences spatiales correspondent à des défauts de forme, pas à de la rugosité)
            from scipy.fft import fft2, ifft2, fftshift
            noise_fft = fft2(noise)
            rows, cols = noise_fft.shape
            crow, ccol = rows // 2, cols // 2
            
            # Créer un filtre passe-haut (supprimer les basses fréquences)
            mask = np.ones_like(noise)
            radius = min(rows, cols) // 10  # Rayon du filtre
            y, x = np.ogrid[:rows, :cols]
            mask_area = (x - ccol)**2 + (y - crow)**2 <= radius**2
            mask[mask_area] = 0
            
            noise_fft_filtered = noise_fft * mask
            noise_filtered = np.real(ifft2(noise_fft_filtered))
            
            # Normaliser pour avoir la bonne RMS
            current_rms = np.std(noise_filtered)
            if current_rms > 0:
                noise_filtered = noise_filtered * (phase_rms / current_rms)
            
            phase_map += noise_filtered
        
        # 2. Ajouter l'erreur de parallélisme (tilt)
        if self.parallelism_arcsec > 0:
            # Convertir les secondes d'arc en radians
            parallelism_rad = self.parallelism_arcsec * np.pi / (180 * 3600)
            
            # Direction du tilt (aléatoire ou fixe)
            # Pour simplifier, on applique un tilt en x
            tilt_direction = np.pi / 4  # 45 degrés (diagonale)
            
            # Calculer la phase de tilt : φ = (2π/λ) * (x*cosθ + y*sinθ) * tilt_angle
            # Mais on veut φ en nm, donc :
            # φ_nm = (2π * (x_mm * 1e-3 * cosθ + y_mm * 1e-3 * sinθ) * tilt_rad) * (wavelength_nm * 1e-9) / (2π) * 1e9
            # Simplification : φ_nm = (x_mm * cosθ + y_mm * sinθ) * tilt_rad * wavelength_nm / 1e3
            phase_map += (grid_x_mm * np.cos(tilt_direction) + grid_y_mm * np.sin(tilt_direction)) * parallelism_rad * wavelength_nm / 1e3
        
        # 3. Ajouter les coefficients de Zernike
        if self.zernike_coefficients:
            for (n, m), coeff_nm in self.zernike_coefficients.items():
                # Générer le polynôme de Zernike
                zernike = zernike_polynomial(n, m, grid_x_mm, grid_y_mm)
                
                # Normalisation : les coefficients sont en nm RMS
                # On suppose que le polynôme est déjà normalisé
                phase_map += coeff_nm * zernike
        
        # 4. Ajouter la carte de phase personnalisée
        if self.custom_phase_map is not None:
            # Redimensionner si nécessaire
            if self.custom_phase_map.shape != grid_x_mm.shape:
                from scipy.interpolate import interp2d
                x = grid_x_mm[0, :]
                y = grid_y_mm[:, 0]
                f = interp2d(x, y, self.custom_phase_map, kind='cubic')
                new_x = np.linspace(x.min(), x.max(), grid_x_mm.shape[1])
                new_y = np.linspace(y.min(), y.max(), grid_x_mm.shape[0])
                phase_map += f(new_x, new_y)
            else:
                phase_map += self.custom_phase_map
        
        return phase_map


@dataclass
class OpticSpecifications:
    """
    FR: Spécifications techniques d'une optique.
        Contient les caractéristiques mécaniques et optiques.

    EN: Technical specifications of an optical element.
        Contains mechanical and optical characteristics.

    Attributes:
        diameter_mm (float): Diamètre de l'optique en mm.
        thickness_mm (float): Épaisseur de l'optique en mm.
        material_name (str): Nom du matériau.
        surface_roughness_nm (float): Rugosité de surface en nm (RMS).
        parallelism_arcsec (float): Parallélisme en secondes d'arc.
        clear_aperture_ratio (float): Ratio d'ouverture utile (0-1).
        edge_thickness_mm (float): Épaisseur au bord pour les lentilles.
        shape_type (ShapeType): Type de forme géométrique.
        shape_dimensions (Dict): Dimensions spécifiques à la forme.
    """
    diameter_mm: float
    thickness_mm: float
    material_name: str = "Fused_Silica"
    surface_roughness_nm: float = 1.0  # Rugosité typique en nm RMS
    parallelism_arcsec: float = 10.0  # Parallélisme typique en arcsec
    clear_aperture_ratio: float = 0.9  # 90% du diamètre est utilisable
    edge_thickness_mm: Optional[float] = None  # Pour les lentilles
    shape_type: ShapeType = ShapeType.CIRCULAR
    shape_dimensions: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# 3. CLASSE DE BASE POUR LES ÉLÉMENTS OPTIQUES / BASE CLASS FOR OPTICAL ELEMENTS
# =============================================================================

@dataclass
class OpticalElement(ABC):
    """
    FR: Classe de base abstraite pour tous les éléments optiques.
        Contient les propriétés communes à toutes les optiques.

    EN: Abstract base class for all optical elements.
        Contains common properties for all optics.

    Attributes:
        name (str): Nom de l'optique.
        specifications (OpticSpecifications): Spécifications techniques.
        position_mm (Tuple[float, float, float]): Position (x, y, z) en mm.
        tilt_deg (Tuple[float, float]): Inclinaison (θx, θy) en degrés.
        decentering_mm (Tuple[float, float]): Décentrement (dx, dy) en mm.
        temperature_K (float): Température en Kelvin.
        wavelength_nm (float): Longueur d'onde de travail en nm.
    """
    name: str
    specifications: OpticSpecifications
    position_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    tilt_deg: Tuple[float, float] = (0.0, 0.0)
    decentering_mm: Tuple[float, float] = (0.0, 0.0)
    temperature_K: float = STANDARD_TEMPERATURE_K
    wavelength_nm: float = 633.0
    
    # Propriétés dérivées
    material: Optional[MaterialBehaviour] = None
    wfe: Optional[WaveFrontError] = None
    
    def __post_init__(self):
        """FR: Initialisation après la création de l'objet."""
        # Charger le matériau
        if MATERIAL_BEHAVIOUR_AVAILABLE:
            try:
                self.material = MaterialBehaviour(self.specifications.material_name)
            except ValueError as e:
                logger.warning(f"Matériau inconnu: {self.specifications.material_name}. {e}")
                self.material = None
        else:
            self.material = None
        
        # Initialiser les erreurs de front d'onde
        self.wfe = WaveFrontError(
            surface_roughness_nm=self.specifications.surface_roughness_nm,
            parallelism_arcsec=self.specifications.parallelism_arcsec
        )
    
    @abstractmethod
    def get_phase_map(
        self,
        grid_x_mm: np.ndarray,
        grid_y_mm: np.ndarray,
    ) -> np.ndarray:
        """
        FR: Méthode abstraite pour calculer la carte de phase introduite par l'optique.

        EN: Abstract method to calculate the phase map introduced by the optic.

        Args:
            grid_x_mm (np.ndarray): Grille de positions en x (mm).
            grid_y_mm (np.ndarray): Grille de positions en y (mm).

        Returns:
            np.ndarray: Carte de phase en nm.
        """
        pass
    
    def get_aperture_mask(
        self,
        grid_x_mm: np.ndarray,
        grid_y_mm: np.ndarray,
    ) -> np.ndarray:
        """
        FR: Génère un masque binaire pour l'ouverture de l'optique.
            Masque = 1 à l'intérieur de l'optique, 0 à l'extérieur.

        EN: Generates a binary mask for the optic aperture.
            Mask = 1 inside the optic, 0 outside.

        Args:
            grid_x_mm (np.ndarray): Grille de positions en x (mm).
            grid_y_mm (np.ndarray): Grille de positions en y (mm).

        Returns:
            np.ndarray: Masque binaire (0 ou 1).
        """
        # Calculer la distance par rapport au centre (en tenant compte du décentrement)
        x_center = self.decentering_mm[0]
        y_center = self.decentering_mm[1]
        
        # Créer une grille centrée
        x = grid_x_mm - x_center
        y = grid_y_mm - y_center
        
        # Déterminer la forme de l'ouverture
        shape_type = self.specifications.shape_type
        
        if shape_type == ShapeType.CIRCULAR:
            # Ouverture circulaire
            r_mm = np.sqrt(x**2 + y**2)
            effective_diameter = self.specifications.diameter_mm * self.specifications.clear_aperture_ratio
            mask = (r_mm <= effective_diameter / 2).astype(float)
        
        elif shape_type == ShapeType.RECTANGULAR:
            # Ouverture rectangulaire
            width = self.specifications.shape_dimensions.get("width_mm", self.specifications.diameter_mm)
            height = self.specifications.shape_dimensions.get("height_mm", self.specifications.diameter_mm)
            effective_width = width * self.specifications.clear_aperture_ratio
            effective_height = height * self.specifications.clear_aperture_ratio
            mask = ((np.abs(x) <= effective_width / 2) & (np.abs(y) <= effective_height / 2)).astype(float)
        
        elif shape_type == ShapeType.SQUARE:
            # Ouverture carrée
            side = self.specifications.shape_dimensions.get("side_mm", self.specifications.diameter_mm)
            effective_side = side * self.specifications.clear_aperture_ratio
            mask = ((np.abs(x) <= effective_side / 2) & (np.abs(y) <= effective_side / 2)).astype(float)
        
        elif shape_type == ShapeType.ELLIPTICAL:
            # Ouverture elliptique
            a = self.specifications.shape_dimensions.get("semi_major_axis_mm", self.specifications.diameter_mm / 2)
            b = self.specifications.shape_dimensions.get("semi_minor_axis_mm", self.specifications.diameter_mm / 2)
            effective_a = a * self.specifications.clear_aperture_ratio
            effective_b = b * self.specifications.clear_aperture_ratio
            mask = ((x / effective_a)**2 + (y / effective_b)**2 <= 1).astype(float)
        
        elif shape_type == ShapeType.HEXAGONAL:
            # Ouverture hexagonale
            radius = self.specifications.shape_dimensions.get("radius_mm", self.specifications.diameter_mm / 2)
            effective_radius = radius * self.specifications.clear_aperture_ratio
            # Hexagone régulier : |x| <= R, |x/2 + y*sqrt(3)/2| <= R, |x/2 - y*sqrt(3)/2| <= R
            mask = ((np.abs(x) <= effective_radius) & 
                    (np.abs(x/2 + y*np.sqrt(3)/2) <= effective_radius) & 
                    (np.abs(x/2 - y*np.sqrt(3)/2) <= effective_radius)).astype(float)
        
        elif shape_type == ShapeType.OCTAGONAL:
            # Ouverture octogonale
            radius = self.specifications.shape_dimensions.get("radius_mm", self.specifications.diameter_mm / 2)
            effective_radius = radius * self.specifications.clear_aperture_ratio
            # Octogone : combinaison de carré et de losange
            mask = ((np.abs(x) <= effective_radius) & 
                    (np.abs(y) <= effective_radius) & 
                    (np.abs(x) + np.abs(y) <= effective_radius * np.sqrt(2))).astype(float)
        
        elif shape_type == ShapeType.CUSTOM:
            # Forme personnalisée (masque fourni)
            custom_mask = self.specifications.shape_dimensions.get("custom_mask")
            if custom_mask is not None:
                mask = custom_mask.astype(float)
            else:
                mask = np.ones_like(grid_x_mm)
        
        else:
            # Par défaut : circulaire
            r_mm = np.sqrt(x**2 + y**2)
            effective_diameter = self.specifications.diameter_mm * self.specifications.clear_aperture_ratio
            mask = (r_mm <= effective_diameter / 2).astype(float)
        
        return mask
    
    def get_transmission_map(
        self,
        grid_x_mm: np.ndarray,
        grid_y_mm: np.ndarray,
        angle_deg: float = 0.0,
        polarization: Polarization = Polarization.NONE,
    ) -> np.ndarray:
        """
        FR: Calcule la carte de transmission de l'optique.
            Pour les lentilles : transmission = exp(-α * épaisseur)
            où α est le coefficient d'absorption du matériau.

        EN: Calculates the transmission map of the optic.
            For lenses: transmission = exp(-α * thickness)
            where α is the material's absorption coefficient.

        Args:
            grid_x_mm (np.ndarray): Grille de positions en x (mm).
            grid_y_mm (np.ndarray): Grille de positions en y (mm).
            angle_deg (float): Angle d'incidence en degrés.
            polarization (Polarization): Polarisation de la lumière.

        Returns:
            np.ndarray: Carte de transmission (0-1).
        """
        if self.material is None:
            # Transmission parfaite si matériau inconnu
            return np.ones_like(grid_x_mm)
        
        # Calculer la transmission (sans prendre en compte l'angle pour l'instant)
        thickness_m = self.specifications.thickness_mm * 1e-3
        absorption = self.material._get_absorption_coefficient(self.wavelength_nm)
        transmission = np.exp(-absorption * thickness_m)
        
        # Appliquer le masque d'aperture
        mask = self.get_aperture_mask(grid_x_mm, grid_y_mm)
        
        return transmission * mask
    
    def get_reflection_map(
        self,
        grid_x_mm: np.ndarray,
        grid_y_mm: np.ndarray,
        angle_deg: float = 0.0,
        polarization: Polarization = Polarization.NONE,
        n_medium: float = 1.0,
    ) -> np.ndarray:
        """
        FR: Calcule la carte de réflexion de l'optique.
            Pour les miroirs : réflexion = 1 (parfait)
            Pour les lentilles : réflexion = R (calculée par Fresnel)

        EN: Calculates the reflection map of the optic.
            For mirrors: reflection = 1 (perfect)
            For lenses: reflection = R (calculated by Fresnel)

        Args:
            grid_x_mm (np.ndarray): Grille de positions en x (mm).
            grid_y_mm (np.ndarray): Grille de positions en y (mm).
            angle_deg (float): Angle d'incidence en degrés.
            polarization (Polarization): Polarisation de la lumière.
            n_medium (float): Indice du milieu incident.

        Returns:
            np.ndarray: Carte de réflexion (0-1).
        """
        if self.material is None:
            # Réflexion nulle si matériau inconnu
            return np.zeros_like(grid_x_mm)
        
        # Calculer la réflectance
        R = self.material.get_reflectance(
            self.wavelength_nm,
            angle_deg=angle_deg,
            polarization=polarization,
            n_medium=n_medium
        )
        
        # Appliquer le masque d'aperture
        mask = self.get_aperture_mask(grid_x_mm, grid_y_mm)
        
        return R * mask
    
    def get_full_phase_map(
        self,
        grid_x_mm: np.ndarray,
        grid_y_mm: np.ndarray,
        include_wfe: bool = True,
        include_tilt: bool = True,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        FR: Génère la carte de phase complète de l'optique, incluant :
            - La phase optique (lentille, miroir, etc.)
            - Les erreurs de front d'onde (WFE)
            - L'effet du tilt

        EN: Generates the complete phase map of the optic, including:
            - The optical phase (lens, mirror, etc.)
            - Wavefront errors (WFE)
            - Tilt effect

        Args:
            grid_x_mm (np.ndarray): Grille de positions en x (mm).
            grid_y_mm (np.ndarray): Grille de positions en y (mm).
            include_wfe (bool): Inclure les erreurs de front d'onde.
            include_tilt (bool): Inclure l'effet du tilt.
            seed (int, optional): Graine pour la génération aléatoire.

        Returns:
            np.ndarray: Carte de phase complète en nm.
        """
        # 1. Calculer la phase optique de base
        phase_map = self.get_phase_map(grid_x_mm, grid_y_mm)
        
        # 2. Ajouter les erreurs de front d'onde
        if include_wfe and self.wfe is not None:
            wfe_phase = self.wfe.generate_phase_map(grid_x_mm, grid_y_mm, self.wavelength_nm, seed=seed)
            phase_map += wfe_phase
        
        # 3. Ajouter l'effet du tilt
        if include_tilt and (self.tilt_deg[0] != 0 or self.tilt_deg[1] != 0):
            # Convertir le tilt en radians
            tilt_x_rad = np.deg2rad(self.tilt_deg[0])
            tilt_y_rad = np.deg2rad(self.tilt_deg[1])
            
            # Calculer la phase de tilt : φ = (2π/λ) * (x*tilt_x + y*tilt_y) * d
            # Mais on veut φ en nm, donc :
            # φ_nm = (2π * (x_mm * 1e-3 * cosθ + y_mm * 1e-3 * sinθ) * tilt_rad) * (wavelength_nm * 1e-9) / (2π) * 1e9
            # Simplification : φ_nm = (x_mm * cosθ + y_mm * sinθ) * tilt_rad * wavelength_nm / 1e3
            phase_map += (grid_x_mm * tilt_x_rad + grid_y_mm * tilt_y_rad) * self.wavelength_nm
        
        # 4. Appliquer le masque d'aperture (la phase est nulle à l'extérieur)
        mask = self.get_aperture_mask(grid_x_mm, grid_y_mm)
        phase_map = phase_map * mask
        
        return phase_map
    
    def apply_to_beam(self, beam: 'Beam') -> 'Beam':
        """
        FR: Applique l'optique à un faisceau.
            Met à jour la phase du faisceau avec la phase de l'optique.

        EN: Applies the optic to a beam.
            Updates the beam's phase with the optic's phase.

        Args:
            beam (Beam): Faisceau incident.

        Returns:
            Beam: Faisceau après passage à travers l'optique.
        """
        # Créer une grille correspondant au faisceau
        grid_x_mm, grid_y_mm = create_grid(beam.diameter_mm, beam.num_points)
        
        # Calculer la carte de phase complète
        phase_map = self.get_full_phase_map(grid_x_mm, grid_y_mm)
        
        # Créer une nouvelle instance de Beam avec la phase modifiée
        # (On suppose que Beam a une méthode pour appliquer une phase)
        new_beam = Beam(
            wavelength_nm=beam.wavelength_nm,
            diameter_mm=beam.diameter_mm,
            energy=beam.energy,
            num_points=beam.num_points,
            coherence=beam.coherence,
        )
        
        # Appliquer la phase à un champ électrique existant
        if beam.electric_field is not None:
            # Extraire l'amplitude et la phase initiale
            amplitude = np.abs(beam.electric_field)
            initial_phase = np.angle(beam.electric_field)
            
            # Convertir la phase_map en radians
            phase_rad = phase_map * 2 * np.pi / self.wavelength_nm
            
            # Nouvelle phase = phase initiale + phase de l'optique
            new_phase = initial_phase + phase_rad
            
            # Nouveau champ électrique
            new_electric_field = amplitude * np.exp(1j * new_phase)
            new_beam.electric_field = new_electric_field
            
            # Mettre à jour l'intensité et la phase
            new_beam.intensity = new_beam.compute_intensity_from_electric_field(new_electric_field)
            new_beam.phase = new_beam.extract_phase_from_electric_field(new_electric_field)
        
        return new_beam
    
    def get_optical_path_difference(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        """
        FR: Calcule la différence de marche optique (OPD) introduite par l'optique.
            OPD = (n - 1) * épaisseur_effective

        EN: Calculates the optical path difference (OPD) introduced by the optic.
            OPD = (n - 1) * effective_thickness

        Args:
            grid_x_mm (np.ndarray): Grille de positions en x (mm).
            grid_y_mm (np.ndarray): Grille de positions en y (mm).

        Returns:
            np.ndarray: Carte de OPD en nm.
        """
        if self.material is None:
            return np.zeros_like(grid_x_mm)
        
        # Calculer l'indice de réfraction
        n = self.material.get_refractive_index(self.wavelength_nm, self.temperature_K)
        
        # Calculer l'épaisseur effective (à implémenter dans les classes dérivées)
        effective_thickness_mm = self._get_effective_thickness(grid_x_mm, grid_y_mm)
        
        # OPD = (n - 1) * épaisseur * 1e6 (pour convertir mm en nm)
        opd_nm = (n - 1) * effective_thickness_mm * 1e6
        
        return opd_nm
    
    @abstractmethod
    def _get_effective_thickness(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        """
        FR: Méthode abstraite pour calculer l'épaisseur effective de l'optique.

        EN: Abstract method to calculate the effective thickness of the optic.

        Args:
            grid_x_mm (np.ndarray): Grille de positions en x (mm).
            grid_y_mm (np.ndarray): Grille de positions en y (mm).

        Returns:
            np.ndarray: Épaisseur effective en mm.
        """
        pass


# =============================================================================
# 4. CLASSES POUR LES DIFFÉRENTS TYPES D'OPTIQUES / CLASSES FOR DIFFERENT OPTIC TYPES
# =============================================================================

class IdealLens(OpticalElement):
    """
    FR: Lentille paraxiale parfaite.
        Génère une carte de phase quadratique : φ = -π * (x² + y²) / (λ * f)
        où f est la distance focale et λ la longueur d'onde.

    EN: Perfect paraxial lens.
        Generates a quadratic phase map: φ = -π * (x² + y²) / (λ * f)
        where f is the focal length and λ is the wavelength.

    Attributes:
        focal_length_mm (float): Distance focale en mm.
    """
    
    def __init__(
        self,
        name: str,
        focal_length_mm: float,
        diameter_mm: float,
        material_name: str = "ideal",
        specifications: Optional[OpticSpecifications] = None,
        **kwargs,
    ):
        """
        FR: Initialise une lentille paraxiale parfaite.

        EN: Initializes a perfect paraxial lens.

        Args:
            name (str): Nom de la lentille.
            focal_length_mm (float): Distance focale en mm.
            diameter_mm (float): Diamètre de la lentille en mm.
            material_name (str): Nom du matériau (par défaut "ideal").
            specifications (OpticSpecifications, optional): Spécifications techniques.
            **kwargs: Arguments supplémentaires pour OpticalElement.
        """
        if specifications is None:
            specifications = OpticSpecifications(
                diameter_mm=diameter_mm,
                thickness_mm=0.0,  # Épaisseur nulle pour une lentille idéale
                material_name=material_name,
            )
        
        self.focal_length_mm = focal_length_mm
        
        super().__init__(
            name=name,
            specifications=specifications,
            **kwargs,
        )
    
    def get_phase_map(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        """
        FR: Calcule la carte de phase quadratique pour une lentille paraxiale.
            φ = -π * (x² + y²) / (λ * f) * λ/(2π) * 2π = -π * (x² + y²) / (λ * f) * λ
            Simplification : φ = -π * (x² + y²) / (f * λ) * λ = -π * (x² + y²) / f
            Mais on veut φ en nm, donc :
            φ_nm = -π * (x² + y²) / (f * λ) * λ * 1e9 / (2π)
            = - (x² + y²) / (2 * f) * 1e9
            
            Correction : φ = - (2π / λ) * (x² + y²) / (2f)
            φ_nm = φ * λ / (2π) * 1e9 = - (x² + y²) / (2f) * 1e9

        EN: Calculates the quadratic phase map for a paraxial lens.

        Args:
            grid_x_mm (np.ndarray): Grille de positions en x (mm).
            grid_y_mm (np.ndarray): Grille de positions en y (mm).

        Returns:
            np.ndarray: Carte de phase en nm.
        """
        # Convertir la distance focale en mètres
        f_m = self.focal_length_mm * 1e-3
        
        # Convertir les positions en mètres
        x_m = grid_x_mm * 1e-3
        y_m = grid_y_mm * 1e-3
        
        # Calculer la phase en radians : φ = - (2π / λ) * (x² + y²) / (2f)
        wavelength_m = self.wavelength_nm * 1e-9
        phase_rad = - (2 * np.pi / wavelength_m) * (x_m**2 + y_m**2) / (2 * f_m)
        
        # Convertir en nm : φ_nm = φ_rad * λ / (2π) * 1e9
        phase_nm = phase_rad * wavelength_m / (2 * np.pi) * 1e9
        
        return phase_nm
    
    def _get_effective_thickness(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        """
        FR: Pour une lentille idéale, l'épaisseur effective est constante.

        EN: For an ideal lens, the effective thickness is constant.
        """
        return np.full_like(grid_x_mm, 0.0)  # Épaisseur nulle pour une lentille idéale


class SimpleLens(OpticalElement):
    """
    FR: Lentille simple avec une face courbe et une face plane.
        Types supportés : plan-convexe, plan-concave.

    EN: Simple lens with one curved surface and one flat surface.
        Supported types: plan-convex, plan-concave.

    Attributes:
        radius_of_curvature_mm (float): Rayon de courbure de la face courbe en mm.
        lens_type (LensType): Type de lentille (PLAN_CONVEX ou PLAN_CONCAVE).
        curved_face_position (str): Position de la face courbe ('front' ou 'back').
    """
    
    def __init__(
        self,
        name: str,
        radius_of_curvature_mm: float,
        diameter_mm: float,
        thickness_mm: float,
        lens_type: LensType = LensType.PLAN_CONVEX,
        material_name: str = "BK7",
        curved_face_position: str = "front",
        specifications: Optional[OpticSpecifications] = None,
        **kwargs,
    ):
        """
        FR: Initialise une lentille simple.

        EN: Initializes a simple lens.

        Args:
            name (str): Nom de la lentille.
            radius_of_curvature_mm (float): Rayon de courbure en mm (positif pour convexe).
            diameter_mm (float): Diamètre de la lentille en mm.
            thickness_mm (float): Épaisseur de la lentille en mm.
            lens_type (LensType): Type de lentille.
            material_name (str): Nom du matériau.
            curved_face_position (str): Position de la face courbe ('front' ou 'back').
            specifications (OpticSpecifications, optional): Spécifications techniques.
            **kwargs: Arguments supplémentaires pour OpticalElement.
        """
        if specifications is None:
            specifications = OpticSpecifications(
                diameter_mm=diameter_mm,
                thickness_mm=thickness_mm,
                material_name=material_name,
            )
        
        self.radius_of_curvature_mm = radius_of_curvature_mm
        self.lens_type = lens_type
        self.curved_face_position = curved_face_position
        
        super().__init__(
            name=name,
            specifications=specifications,
            **kwargs,
        )
    
    def get_phase_map(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        """
        FR: Calcule la carte de phase pour une lentille simple.
            Pour une lentille plan-convexe :
            φ = (n - 1) * [d - (x² + y²)/(2R)] * 1e6 (pour convertir en nm)
            où d est l'épaisseur au centre et R le rayon de courbure.

        EN: Calculates the phase map for a simple lens.
            For a plan-convex lens:
            φ = (n - 1) * [d - (x² + y²)/(2R)] * 1e6 (to convert to nm)
            where d is the center thickness and R is the radius of curvature.

        Args:
            grid_x_mm (np.ndarray): Grille de positions en x (mm).
            grid_y_mm (np.ndarray): Grille de positions en y (mm).

        Returns:
            np.ndarray: Carte de phase en nm.
        """
        if self.material is None:
            return np.zeros_like(grid_x_mm)
        
        # Calculer l'indice de réfraction
        n = self.material.get_refractive_index(self.wavelength_nm, self.temperature_K)
        
        # Calculer l'épaisseur effective
        effective_thickness_mm = self._get_effective_thickness(grid_x_mm, grid_y_mm)
        
        # Calculer la phase : φ = (n - 1) * épaisseur_effective * 1e6 (mm → nm)
        phase_nm = (n - 1) * effective_thickness_mm * 1e6
        
        return phase_nm
    
    def _get_effective_thickness(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        """
        FR: Calcule l'épaisseur effective pour une lentille simple.
            Pour une lentille plan-convexe :
            épaisseur(x,y) = d - (x² + y²)/(2R) + sqrt(R² - (x² + y²) + (x² + y²)²/(4R²))
            Approximation paraxiale : épaisseur(x,y) ≈ d - (x² + y²)/(2R)

        EN: Calculates the effective thickness for a simple lens.
            For a plan-convex lens:
            thickness(x,y) = d - (x² + y²)/(2R) + sqrt(R² - (x² + y²) + (x² + y²)²/(4R²))
            Paraxial approximation: thickness(x,y) ≈ d - (x² + y²)/(2R)

        Args:
            grid_x_mm (np.ndarray): Grille de positions en x (mm).
            grid_y_mm (np.ndarray): Grille de positions en y (mm).

        Returns:
            np.ndarray: Épaisseur effective en mm.
        """
        R_mm = self.radius_of_curvature_mm
        d_mm = self.specifications.thickness_mm
        
        # Calculer la distance radiale
        r_mm = np.sqrt(grid_x_mm**2 + grid_y_mm**2)
        
        # Approximation paraxiale : épaisseur ≈ d - r²/(2R)
        # Le signe dépend du type de lentille et de la position de la face courbe
        if self.lens_type == LensType.PLAN_CONVEX:
            if self.curved_face_position == "front":
                # Face avant convexe : l'épaisseur diminue vers les bords
                effective_thickness = d_mm - r_mm**2 / (2 * R_mm)
            else:  # back
                # Face arrière convexe : l'épaisseur augmente vers les bords
                effective_thickness = d_mm + r_mm**2 / (2 * R_mm)
        elif self.lens_type == LensType.PLAN_CONCAVE:
            if self.curved_face_position == "front":
                # Face avant concave : l'épaisseur augmente vers les bords
                effective_thickness = d_mm + r_mm**2 / (2 * abs(R_mm))
            else:  # back
                # Face arrière concave : l'épaisseur diminue vers les bords
                effective_thickness = d_mm - r_mm**2 / (2 * abs(R_mm))
        else:
            effective_thickness = np.full_like(grid_x_mm, d_mm)
        
        # Limiter l'épaisseur à des valeurs positives
        effective_thickness = np.maximum(effective_thickness, 0.0)
        
        return effective_thickness


class DoubleLens(OpticalElement):
    """
    FR: Lentille avec deux faces courbes (biconvexe, biconcave, ménisque).

    EN: Lens with two curved surfaces (biconvex, biconcave, meniscus).

    Attributes:
        radius_of_curvature_1_mm (float): Rayon de courbure de la première face en mm.
        radius_of_curvature_2_mm (float): Rayon de courbure de la deuxième face en mm.
        lens_type (LensType): Type de lentille.
    """
    
    def __init__(
        self,
        name: str,
        radius_of_curvature_1_mm: float,
        radius_of_curvature_2_mm: float,
        diameter_mm: float,
        thickness_mm: float,
        lens_type: LensType = LensType.BICONVEX,
        material_name: str = "BK7",
        specifications: Optional[OpticSpecifications] = None,
        **kwargs,
    ):
        """
        FR: Initialise une lentille à deux faces courbes.

        EN: Initializes a lens with two curved surfaces.

        Args:
            name (str): Nom de la lentille.
            radius_of_curvature_1_mm (float): Rayon de courbure de la première face en mm.
            radius_of_curvature_2_mm (float): Rayon de courbure de la deuxième face en mm.
            diameter_mm (float): Diamètre de la lentille en mm.
            thickness_mm (float): Épaisseur de la lentille en mm.
            lens_type (LensType): Type de lentille.
            material_name (str): Nom du matériau.
            specifications (OpticSpecifications, optional): Spécifications techniques.
            **kwargs: Arguments supplémentaires pour OpticalElement.
        """
        if specifications is None:
            specifications = OpticSpecifications(
                diameter_mm=diameter_mm,
                thickness_mm=thickness_mm,
                material_name=material_name,
            )
        
        self.radius_of_curvature_1_mm = radius_of_curvature_1_mm
        self.radius_of_curvature_2_mm = radius_of_curvature_2_mm
        self.lens_type = lens_type
        
        super().__init__(
            name=name,
            specifications=specifications,
            **kwargs,
        )
    
    def get_phase_map(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        """
        FR: Calcule la carte de phase pour une lentille à deux faces courbes.
            φ = (n - 1) * [d - (x² + y²)/(2R₁) + (x² + y²)/(2R₂)] * 1e6

        EN: Calculates the phase map for a lens with two curved surfaces.

        Args:
            grid_x_mm (np.ndarray): Grille de positions en x (mm).
            grid_y_mm (np.ndarray): Grille de positions en y (mm).

        Returns:
            np.ndarray: Carte de phase en nm.
        """
        if self.material is None:
            return np.zeros_like(grid_x_mm)
        
        # Calculer l'indice de réfraction
        n = self.material.get_refractive_index(self.wavelength_nm, self.temperature_K)
        
        # Calculer l'épaisseur effective
        effective_thickness_mm = self._get_effective_thickness(grid_x_mm, grid_y_mm)
        
        # Calculer la phase
        phase_nm = (n - 1) * effective_thickness_mm * 1e6
        
        return phase_nm
    
    def _get_effective_thickness(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        """
        FR: Calcule l'épaisseur effective pour une lentille à deux faces courbes.
            épaisseur(x,y) ≈ d - r²/(2R₁) + r²/(2R₂)

        EN: Calculates the effective thickness for a lens with two curved surfaces.

        Args:
            grid_x_mm (np.ndarray): Grille de positions en x (mm).
            grid_y_mm (np.ndarray): Grille de positions en y (mm).

        Returns:
            np.ndarray: Épaisseur effective en mm.
        """
        R1_mm = self.radius_of_curvature_1_mm
        R2_mm = self.radius_of_curvature_2_mm
        d_mm = self.specifications.thickness_mm
        
        # Calculer la distance radiale
        r_mm = np.sqrt(grid_x_mm**2 + grid_y_mm**2)
        
        # Approximation paraxiale
        if self.lens_type == LensType.BICONVEX:
            effective_thickness = d_mm - r_mm**2 / (2 * R1_mm) - r_mm**2 / (2 * R2_mm)
        elif self.lens_type == LensType.BICONCAVE:
            effective_thickness = d_mm + r_mm**2 / (2 * abs(R1_mm)) + r_mm**2 / (2 * abs(R2_mm))
        elif self.lens_type == LensType.MENISCUS:
            # Pour un ménisque, un rayon est positif et l'autre négatif
            effective_thickness = d_mm - r_mm**2 / (2 * R1_mm) + r_mm**2 / (2 * R2_mm)
        else:
            effective_thickness = np.full_like(grid_x_mm, d_mm)
        
        # Limiter l'épaisseur à des valeurs positives
        effective_thickness = np.maximum(effective_thickness, 0.0)
        
        return effective_thickness


class CementedDoublet(OpticalElement):
    """
    FR: Doublet collé (lentille achromatique).
        Combine deux lentilles de matériaux différents collées ensemble.

    EN: Cemented doublet (achromatic lens).
        Combines two lenses of different materials cemented together.

    Attributes:
        lens1 (OpticalElement): Première lentille.
        lens2 (OpticalElement): Deuxième lentille.
        interface_radius_mm (float): Rayon de courbure de l'interface en mm.
    """
    
    def __init__(
        self,
        name: str,
        lens1: OpticalElement,
        lens2: OpticalElement,
        interface_radius_mm: float,
        specifications: Optional[OpticSpecifications] = None,
        **kwargs,
    ):
        """
        FR: Initialise un doublet collé.

        EN: Initializes a cemented doublet.

        Args:
            name (str): Nom du doublet.
            lens1 (OpticalElement): Première lentille.
            lens2 (OpticalElement): Deuxième lentille.
            interface_radius_mm (float): Rayon de courbure de l'interface en mm.
            specifications (OpticSpecifications, optional): Spécifications techniques.
            **kwargs: Arguments supplémentaires pour OpticalElement.
        """
        if specifications is None:
            specifications = OpticSpecifications(
                diameter_mm=min(lens1.specifications.diameter_mm, lens2.specifications.diameter_mm),
                thickness_mm=lens1.specifications.thickness_mm + lens2.specifications.thickness_mm,
                material_name=f"{lens1.specifications.material_name}+{lens2.specifications.material_name}",
            )
        
        self.lens1 = lens1
        self.lens2 = lens2
        self.interface_radius_mm = interface_radius_mm
        
        super().__init__(
            name=name,
            specifications=specifications,
            **kwargs,
        )
    
    def get_phase_map(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        """
        FR: Calcule la carte de phase pour un doublet collé.
            Somme les phases des deux lentilles.

        EN: Calculates the phase map for a cemented doublet.
            Sums the phases of both lenses.

        Args:
            grid_x_mm (np.ndarray): Grille de positions en x (mm).
            grid_y_mm (np.ndarray): Grille de positions en y (mm).

        Returns:
            np.ndarray: Carte de phase en nm.
        """
        # Calculer la phase de chaque lentille
        phase1 = self.lens1.get_phase_map(grid_x_mm, grid_y_mm)
        phase2 = self.lens2.get_phase_map(grid_x_mm, grid_y_mm)
        
        # Somme des phases
        return phase1 + phase2
    
    def get_full_phase_map(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray, include_wfe: bool = True, include_tilt: bool = True, seed: Optional[int] = None) -> np.ndarray:
        """
        FR: Génère la carte de phase complète du doublet.

        EN: Generates the complete phase map of the doublet.
        """
        phase1 = self.lens1.get_full_phase_map(grid_x_mm, grid_y_mm, include_wfe, include_tilt, seed)
        phase2 = self.lens2.get_full_phase_map(grid_x_mm, grid_y_mm, include_wfe, include_tilt, seed)
        return phase1 + phase2


class Mirror(OpticalElement):
    """
    FR: Miroir optique (plat, sphérique, parabolique).

    EN: Optical mirror (flat, spherical, parabolic).

    Attributes:
        mirror_type (MirrorType): Type de miroir.
        radius_of_curvature_mm (float): Rayon de courbure pour les miroirs courbes (en mm).
        focal_length_mm (float): Distance focale (pour les miroirs courbes).
    """
    
    def __init__(
        self,
        name: str,
        diameter_mm: float,
        mirror_type: MirrorType = MirrorType.FLAT,
        radius_of_curvature_mm: Optional[float] = None,
        focal_length_mm: Optional[float] = None,
        material_name: str = "Aluminum",  # Matériau du substrat
        coating_reflectivity: float = 0.95,  # Réflectivité du revêtement
        specifications: Optional[OpticSpecifications] = None,
        **kwargs,
    ):
        """
        FR: Initialise un miroir optique.

        EN: Initializes an optical mirror.

        Args:
            name (str): Nom du miroir.
            diameter_mm (float): Diamètre du miroir en mm.
            mirror_type (MirrorType): Type de miroir.
            radius_of_curvature_mm (float): Rayon de courbure en mm (pour les miroirs courbes).
            focal_length_mm (float): Distance focale en mm (pour les miroirs paraboliques).
            material_name (str): Nom du matériau du substrat.
            coating_reflectivity (float): Réflectivité du revêtement (0-1).
            specifications (OpticSpecifications, optional): Spécifications techniques.
            **kwargs: Arguments supplémentaires pour OpticalElement.
        """
        if specifications is None:
            specifications = OpticSpecifications(
                diameter_mm=diameter_mm,
                thickness_mm=10.0,  # Épaisseur typique pour un miroir
                material_name=material_name,
            )
        
        self.mirror_type = mirror_type
        self.radius_of_curvature_mm = radius_of_curvature_mm
        self.focal_length_mm = focal_length_mm
        self.coating_reflectivity = coating_reflectivity
        
        # Calculer le rayon de courbure à partir de la distance focale si nécessaire
        if self.focal_length_mm is not None and self.radius_of_curvature_mm is None:
            if mirror_type == MirrorType.SPHERICAL:
                self.radius_of_curvature_mm = 2 * self.focal_length_mm
            elif mirror_type == MirrorType.PARABOLIC:
                # Pour un miroir parabolique, R ≈ 2f (approximation)
                self.radius_of_curvature_mm = 2 * self.focal_length_mm
        
        super().__init__(
            name=name,
            specifications=specifications,
            **kwargs,
        )
    
    def get_phase_map(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        """
        FR: Calcule la carte de phase pour un miroir.
            Pour un miroir plat : phase = 0 (ou constante).
            Pour un miroir sphérique : phase = -2 * (x² + y²) / R * 1e6
            Pour un miroir parabolique : phase ≈ - (x² + y²) / f * 1e6

        EN: Calculates the phase map for a mirror.
            For a flat mirror: phase = 0 (or constant).
            For a spherical mirror: phase = -2 * (x² + y²) / R * 1e6
            For a parabolic mirror: phase ≈ - (x² + y²) / f * 1e6

        Args:
            grid_x_mm (np.ndarray): Grille de positions en x (mm).
            grid_y_mm (np.ndarray): Grille de positions en y (mm).

        Returns:
            np.ndarray: Carte de phase en nm.
        """
        if self.mirror_type == MirrorType.FLAT:
            # Miroir plat : phase constante (0)
            return np.zeros_like(grid_x_mm)
        
        elif self.mirror_type == MirrorType.SPHERICAL:
            # Miroir sphérique : φ = -2 * (x² + y²) / R
            R_mm = self.radius_of_curvature_mm
            r_squared = grid_x_mm**2 + grid_y_mm**2
            phase_nm = -2 * r_squared / R_mm * 1e6
            return phase_nm
        
        elif self.mirror_type == MirrorType.PARABOLIC:
            # Miroir parabolique : φ ≈ - (x² + y²) / f
            f_mm = self.focal_length_mm
            r_squared = grid_x_mm**2 + grid_y_mm**2
            phase_nm = -r_squared / f_mm * 1e6
            return phase_nm
        
        else:
            return np.zeros_like(grid_x_mm)
    
    def get_reflection_map(
        self,
        grid_x_mm: np.ndarray,
        grid_y_mm: np.ndarray,
        angle_deg: float = 0.0,
        polarization: Polarization = Polarization.NONE,
        n_medium: float = 1.0,
    ) -> np.ndarray:
        """
        FR: Calcule la carte de réflexion pour un miroir.
            Prend en compte la réflectivité du revêtement.

        EN: Calculates the reflection map for a mirror.
            Takes into account the coating reflectivity.
        """
        # Réflexion du miroir = réflectivité du revêtement × masque d'aperture
        mask = self.get_aperture_mask(grid_x_mm, grid_y_mm)
        return self.coating_reflectivity * mask
    
    def _get_effective_thickness(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        """
        FR: Pour un miroir, l'épaisseur effective est constante (épaisseur du substrat).

        EN: For a mirror, the effective thickness is constant (substrate thickness).
        """
        return np.full_like(grid_x_mm, self.specifications.thickness_mm)


class Beamsplitter(OpticalElement):
    """
    FR: Séparateur de faisceau (lame ou cube).

    EN: Beamsplitter (plate or cube).

    Attributes:
        beamsplitter_type (BeamsplitterType): Type de séparateur.
        transmission_ratio (float): Ratio de transmission (0-1).
        reflection_ratio (float): Ratio de réflexion (0-1).
        polarization_axis (str): Axe de polarisation pour les séparateurs polarisants.
    """
    
    def __init__(
        self,
        name: str,
        diameter_mm: float,
        beamsplitter_type: BeamsplitterType = BeamsplitterType.PLATE,
        transmission_ratio: float = 0.5,
        reflection_ratio: float = 0.5,
        material_name: str = "BK7",
        polarization_axis: Optional[str] = None,
        specifications: Optional[OpticSpecifications] = None,
        **kwargs,
    ):
        """
        FR: Initialise un séparateur de faisceau.

        EN: Initializes a beamsplitter.

        Args:
            name (str): Nom du séparateur.
            diameter_mm (float): Diamètre en mm.
            beamsplitter_type (BeamsplitterType): Type de séparateur.
            transmission_ratio (float): Ratio de transmission (0-1).
            reflection_ratio (float): Ratio de réflexion (0-1).
            material_name (str): Nom du matériau.
            polarization_axis (str): Axe de polarisation (pour les séparateurs polarisants).
            specifications (OpticSpecifications, optional): Spécifications techniques.
            **kwargs: Arguments supplémentaires pour OpticalElement.
        """
        if specifications is None:
            specifications = OpticSpecifications(
                diameter_mm=diameter_mm,
                thickness_mm=1.0,  # Épaisseur typique pour une lame
                material_name=material_name,
            )
        
        self.beamsplitter_type = beamsplitter_type
        self.transmission_ratio = transmission_ratio
        self.reflection_ratio = reflection_ratio
        self.polarization_axis = polarization_axis
        
        # Normaliser les ratios
        total = transmission_ratio + reflection_ratio
        if total > 0:
            self.transmission_ratio /= total
            self.reflection_ratio /= total
        
        super().__init__(
            name=name,
            specifications=specifications,
            **kwargs,
        )
    
    def get_phase_map(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        """
        FR: Calcule la carte de phase pour un séparateur de faisceau.
            Pour une lame : phase dépend de l'épaisseur et de l'indice.

        EN: Calculates the phase map for a beamsplitter.
            For a plate: phase depends on thickness and refractive index.

        Args:
            grid_x_mm (np.ndarray): Grille de positions en x (mm).
            grid_y_mm (np.ndarray): Grille de positions en y (mm).

        Returns:
            np.ndarray: Carte de phase en nm.
        """
        if self.material is None:
            return np.zeros_like(grid_x_mm)
        
        # Calculer l'indice de réfraction
        n = self.material.get_refractive_index(self.wavelength_nm, self.temperature_K)
        
        # Calculer l'épaisseur effective
        effective_thickness_mm = self._get_effective_thickness(grid_x_mm, grid_y_mm)
        
        # Phase = (n - 1) * épaisseur * 1e6
        phase_nm = (n - 1) * effective_thickness_mm * 1e6
        
        return phase_nm
    
    def get_transmission_map(
        self,
        grid_x_mm: np.ndarray,
        grid_y_mm: np.ndarray,
        angle_deg: float = 0.0,
        polarization: Polarization = Polarization.NONE,
    ) -> np.ndarray:
        """
        FR: Calcule la carte de transmission pour un séparateur de faisceau.

        EN: Calculates the transmission map for a beamsplitter.
        """
        mask = self.get_aperture_mask(grid_x_mm, grid_y_mm)
        
        # Prendre en compte la polarisation si le séparateur est polarisant
        if self.beamsplitter_type == BeamsplitterType.POLARIZING and self.polarization_axis is not None:
            # Pour un séparateur polarisant, la transmission dépend de la polarisation
            if polarization == Polarization.S or polarization == Polarization.P:
                # Supposons que l'axe de polarisation est à 0° (horizontal)
                if self.polarization_axis == "horizontal":
                    if polarization == Polarization.P:
                        transmission = 0.0  # Bloque la polarisation P
                    else:
                        transmission = 1.0  # Transmet la polarisation S
                else:  # vertical
                    if polarization == Polarization.S:
                        transmission = 0.0  # Bloque la polarisation S
                    else:
                        transmission = 1.0  # Transmet la polarisation P
            else:
                # Lumière non polarisée : transmission moyenne
                transmission = 0.5
        else:
            # Séparateur non polarisant
            transmission = self.transmission_ratio
        
        return transmission * mask
    
    def get_reflection_map(
        self,
        grid_x_mm: np.ndarray,
        grid_y_mm: np.ndarray,
        angle_deg: float = 0.0,
        polarization: Polarization = Polarization.NONE,
        n_medium: float = 1.0,
    ) -> np.ndarray:
        """
        FR: Calcule la carte de réflexion pour un séparateur de faisceau.

        EN: Calculates the reflection map for a beamsplitter.
        """
        mask = self.get_aperture_mask(grid_x_mm, grid_y_mm)
        
        # Prendre en compte la polarisation si le séparateur est polarisant
        if self.beamsplitter_type == BeamsplitterType.POLARIZING and self.polarization_axis is not None:
            if polarization == Polarization.S or polarization == Polarization.P:
                if self.polarization_axis == "horizontal":
                    if polarization == Polarization.P:
                        reflection = 1.0  # Réfléchit la polarisation P
                    else:
                        reflection = 0.0  # Transmet la polarisation S
                else:  # vertical
                    if polarization == Polarization.S:
                        reflection = 1.0  # Réfléchit la polarisation S
                    else:
                        reflection = 0.0  # Transmet la polarisation P
            else:
                # Lumière non polarisée : réflexion moyenne
                reflection = 0.5
        else:
            # Séparateur non polarisant
            reflection = self.reflection_ratio
        
        return reflection * mask
    
    def _get_effective_thickness(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        """
        FR: Calcule l'épaisseur effective pour un séparateur de faisceau.

        EN: Calculates the effective thickness for a beamsplitter.
        """
        if self.beamsplitter_type == BeamsplitterType.CUBE:
            # Pour un cube, l'épaisseur effective dépend de l'angle d'incidence
            # Approximation : épaisseur constante
            return np.full_like(grid_x_mm, self.specifications.thickness_mm)
        else:
            # Pour une lame, l'épaisseur est constante
            return np.full_like(grid_x_mm, self.specifications.thickness_mm)


class Window(OpticalElement):
    """
    FR: Fenêtre optique plate (lame de verre).

    EN: Optical window (flat glass plate).

    Attributes:
        tilt_deg (Tuple[float, float]): Inclinaison de la fenêtre (θx, θy).
    """
    
    def __init__(
        self,
        name: str,
        diameter_mm: float,
        thickness_mm: float,
        material_name: str = "Fused_Silica",
        specifications: Optional[OpticSpecifications] = None,
        **kwargs,
    ):
        """
        FR: Initialise une fenêtre optique.

        EN: Initializes an optical window.

        Args:
            name (str): Nom de la fenêtre.
            diameter_mm (float): Diamètre en mm.
            thickness_mm (float): Épaisseur en mm.
            material_name (str): Nom du matériau.
            specifications (OpticSpecifications, optional): Spécifications techniques.
            **kwargs: Arguments supplémentaires pour OpticalElement.
        """
        if specifications is None:
            specifications = OpticSpecifications(
                diameter_mm=diameter_mm,
                thickness_mm=thickness_mm,
                material_name=material_name,
            )
        
        super().__init__(
            name=name,
            specifications=specifications,
            **kwargs,
        )
    
    def get_phase_map(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        """
        FR: Calcule la carte de phase pour une fenêtre optique.
            φ = (n - 1) * épaisseur * 1e6

        EN: Calculates the phase map for an optical window.

        Args:
            grid_x_mm (np.ndarray): Grille de positions en x (mm).
            grid_y_mm (np.ndarray): Grille de positions en y (mm).

        Returns:
            np.ndarray: Carte de phase en nm.
        """
        if self.material is None:
            return np.zeros_like(grid_x_mm)
        
        # Calculer l'indice de réfraction
        n = self.material.get_refractive_index(self.wavelength_nm, self.temperature_K)
        
        # Épaisseur constante
        thickness_mm = self.specifications.thickness_mm
        
        # Phase = (n - 1) * épaisseur * 1e6
        phase_nm = (n - 1) * thickness_mm * 1e6
        
        # Si la fenêtre est inclinée, ajouter une composante linéaire
        if self.tilt_deg[0] != 0 or self.tilt_deg[1] != 0:
            tilt_x_rad = np.deg2rad(self.tilt_deg[0])
            tilt_y_rad = np.deg2rad(self.tilt_deg[1])
            
            # Phase de tilt : φ_tilt = (n - 1) * (x*tilt_x + y*tilt_y) * 1e3
            phase_nm += (n - 1) * (grid_x_mm * tilt_x_rad + grid_y_mm * tilt_y_rad) * 1e3
        
        return phase_nm
    
    def _get_effective_thickness(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        """
        FR: Calcule l'épaisseur effective pour une fenêtre optique.

        EN: Calculates the effective thickness for an optical window.
        """
        return np.full_like(grid_x_mm, self.specifications.thickness_mm)


class AsphericLens(OpticalElement):
    """
    FR: Lentille asphérique.
        Utilise une équation asphérique : z = (r²/R) / (1 + sqrt(1 - (1+k)(r²/R²))) + a₄r⁴ + a₆r⁶ + ...

    EN: Aspheric lens.
        Uses an aspheric equation: z = (r²/R) / (1 + sqrt(1 - (1+k)(r²/R²))) + a₄r⁴ + a₆r⁶ + ...

    Attributes:
        radius_of_curvature_mm (float): Rayon de courbure au sommet en mm.
        conic_constant (float): Constante conique (k).
        aspheric_coefficients (Dict): Coefficients asphériques {4: a4, 6: a6, ...}.
    """
    
    def __init__(
        self,
        name: str,
        radius_of_curvature_mm: float,
        diameter_mm: float,
        thickness_mm: float,
        conic_constant: float = 0.0,
        aspheric_coefficients: Optional[Dict[int, float]] = None,
        material_name: str = "BK7",
        specifications: Optional[OpticSpecifications] = None,
        **kwargs,
    ):
        """
        FR: Initialise une lentille asphérique.

        EN: Initializes an aspheric lens.

        Args:
            name (str): Nom de la lentille.
            radius_of_curvature_mm (float): Rayon de courbure au sommet en mm.
            diameter_mm (float): Diamètre de la lentille en mm.
            thickness_mm (float): Épaisseur de la lentille en mm.
            conic_constant (float): Constante conique (k).
            aspheric_coefficients (Dict): Coefficients asphériques {4: a4, 6: a6, ...}.
            material_name (str): Nom du matériau.
            specifications (OpticSpecifications, optional): Spécifications techniques.
            **kwargs: Arguments supplémentaires pour OpticalElement.
        """
        if specifications is None:
            specifications = OpticSpecifications(
                diameter_mm=diameter_mm,
                thickness_mm=thickness_mm,
                material_name=material_name,
            )
        
        self.radius_of_curvature_mm = radius_of_curvature_mm
        self.conic_constant = conic_constant
        self.aspheric_coefficients = aspheric_coefficients or {}
        
        super().__init__(
            name=name,
            specifications=specifications,
            **kwargs,
        )
    
    def get_phase_map(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        """
        FR: Calcule la carte de phase pour une lentille asphérique.
            Utilise l'équation asphérique pour calculer la sagitta (flèche).

        EN: Calculates the phase map for an aspheric lens.
            Uses the aspheric equation to calculate the sag (height).

        Args:
            grid_x_mm (np.ndarray): Grille de positions en x (mm).
            grid_y_mm (np.ndarray): Grille de positions en y (mm).

        Returns:
            np.ndarray: Carte de phase en nm.
        """
        if self.material is None:
            return np.zeros_like(grid_x_mm)
        
        # Calculer l'indice de réfraction
        n = self.material.get_refractive_index(self.wavelength_nm, self.temperature_K)
        
        # Calculer la distance radiale
        r_mm = np.sqrt(grid_x_mm**2 + grid_y_mm**2)
        R_mm = self.radius_of_curvature_mm
        
        # Calculer la sagitta (flèche) avec l'équation asphérique
        # z = (r²/R) / (1 + sqrt(1 - (1+k)(r²/R²))) + Σ a_i r^i
        with np.errstate(divide='ignore', invalid='ignore'):
            term1 = (r_mm**2 / R_mm) / (1 + np.sqrt(1 - (1 + self.conic_constant) * (r_mm**2 / R_mm**2)))
            
            # Ajouter les termes asphériques
            sag_mm = term1.copy()
            for i, a_i in self.aspheric_coefficients.items():
                sag_mm += a_i * (r_mm ** i)
            
            # Remplacer les NaN par 0
            sag_mm = np.nan_to_num(sag_mm, nan=0.0)
        
        # Calculer l'épaisseur effective
        # Pour une lentille asphérique, l'épaisseur varie avec r
        # On suppose que la lentille est plan-convexe avec la face asphérique avant
        d_mm = self.specifications.thickness_mm
        effective_thickness_mm = d_mm - sag_mm
        
        # Limiter à des valeurs positives
        effective_thickness_mm = np.maximum(effective_thickness_mm, 0.0)
        
        # Calculer la phase
        phase_nm = (n - 1) * effective_thickness_mm * 1e6
        
        return phase_nm
    
    def _get_effective_thickness(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        """
        FR: Calcule l'épaisseur effective pour une lentille asphérique.

        EN: Calculates the effective thickness for an aspheric lens.
        """
        r_mm = np.sqrt(grid_x_mm**2 + grid_y_mm**2)
        R_mm = self.radius_of_curvature_mm
        
        with np.errstate(divide='ignore', invalid='ignore'):
            term1 = (r_mm**2 / R_mm) / (1 + np.sqrt(1 - (1 + self.conic_constant) * (r_mm**2 / R_mm**2)))
            
            sag_mm = term1.copy()
            for i, a_i in self.aspheric_coefficients.items():
                sag_mm += a_i * (r_mm ** i)
            
            sag_mm = np.nan_to_num(sag_mm, nan=0.0)
        
        d_mm = self.specifications.thickness_mm
        effective_thickness_mm = d_mm - sag_mm
        effective_thickness_mm = np.maximum(effective_thickness_mm, 0.0)
        
        return effective_thickness_mm


class Prism(OpticalElement):
    """
    FR: Prisme optique.
        Dévie un faisceau selon l'angle du prisme et l'indice de réfraction.

    EN: Optical prism.
        Deviates a beam according to the prism angle and refractive index.

    Attributes:
        prism_type (PrismType): Type de prisme.
        apex_angle_deg (float): Angle au sommet en degrés.
        base_length_mm (float): Longueur de la base en mm.
        height_mm (float): Hauteur du prisme en mm.
    """
    
    def __init__(
        self,
        name: str,
        diameter_mm: float,
        prism_type: PrismType = PrismType.RIGHT_ANGLE,
        apex_angle_deg: float = 60.0,
        base_length_mm: Optional[float] = None,
        height_mm: Optional[float] = None,
        material_name: str = "BK7",
        specifications: Optional[OpticSpecifications] = None,
        **kwargs,
    ):
        """
        FR: Initialise un prisme optique.

        EN: Initializes an optical prism.

        Args:
            name (str): Nom du prisme.
            diameter_mm (float): Diamètre en mm.
            prism_type (PrismType): Type de prisme.
            apex_angle_deg (float): Angle au sommet en degrés.
            base_length_mm (float): Longueur de la base en mm.
            height_mm (float): Hauteur du prisme en mm.
            material_name (str): Nom du matériau.
            specifications (OpticSpecifications, optional): Spécifications techniques.
            **kwargs: Arguments supplémentaires pour OpticalElement.
        """
        if specifications is None:
            specifications = OpticSpecifications(
                diameter_mm=diameter_mm,
                thickness_mm=height_mm or diameter_mm,
                material_name=material_name,
                shape_type=ShapeType.CUSTOM,  # Les prismes ont des formes personnalisées
            )
        
        self.prism_type = prism_type
        self.apex_angle_deg = apex_angle_deg
        self.base_length_mm = base_length_mm or diameter_mm
        self.height_mm = height_mm or diameter_mm
        
        super().__init__(
            name=name,
            specifications=specifications,
            **kwargs,
        )
    
    def get_phase_map(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        """
        FR: Calcule la carte de phase pour un prisme.
            Pour un prisme : phase linéaire selon l'épaisseur variable.
            φ = (n - 1) * épaisseur(x,y) * 1e6

        EN: Calculates the phase map for a prism.
            For a prism: linear phase according to variable thickness.
            φ = (n - 1) * thickness(x,y) * 1e6

        Args:
            grid_x_mm (np.ndarray): Grille de positions en x (mm).
            grid_y_mm (np.ndarray): Grille de positions en y (mm).

        Returns:
            np.ndarray: Carte de phase en nm.
        """
        if self.material is None:
            return np.zeros_like(grid_x_mm)
        
        # Calculer l'indice de réfraction
        n = self.material.get_refractive_index(self.wavelength_nm, self.temperature_K)
        
        # Calculer l'épaisseur effective du prisme
        effective_thickness_mm = self._get_effective_thickness(grid_x_mm, grid_y_mm)
        
        # Phase = (n - 1) * épaisseur * 1e6
        phase_nm = (n - 1) * effective_thickness_mm * 1e6
        
        return phase_nm
    
    def _get_effective_thickness(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        """
        FR: Calcule l'épaisseur effective pour un prisme.

        EN: Calculates the effective thickness for a prism.
        """
        # Pour un prisme, l'épaisseur varie linéairement
        # On suppose que le prisme est orienté avec la base en bas
        
        # Calculer la distance par rapport au centre
        x_center = self.decentering_mm[0]
        y_center = self.decentering_mm[1]
        
        x = grid_x_mm - x_center
        y = grid_y_mm - y_center
        
        # Épaisseur variable selon la position y
        # Le prisme a une épaisseur maximale à la base et minimale au sommet
        if self.prism_type == PrismType.RIGHT_ANGLE:
            # Prisme à angle droit : épaisseur varie linéairement en y
            # Épaisseur maximale = height_mm, minimale = 0
            effective_thickness = self.height_mm * (1 - np.abs(y) / (self.base_length_mm / 2))
        elif self.prism_type == PrismType.EQUILATERAL:
            # Prisme équilatéral : épaisseur varie selon la position
            # Approximation : épaisseur maximale au centre
            effective_thickness = self.height_mm * (1 - np.abs(x) / (self.base_length_mm / 2))
        else:
            # Par défaut : épaisseur constante
            effective_thickness = np.full_like(grid_x_mm, self.height_mm)
        
        # Limiter à des valeurs positives
        effective_thickness = np.maximum(effective_thickness, 0.0)
        
        return effective_thickness
    
    def get_aperture_mask(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        """
        FR: Génère un masque binaire pour l'ouverture du prisme.
            Forme triangulaire ou rectangulaire selon le type.

        EN: Generates a binary mask for the prism aperture.
            Triangular or rectangular shape depending on type.

        Args:
            grid_x_mm (np.ndarray): Grille de positions en x (mm).
            grid_y_mm (np.ndarray): Grille de positions en y (mm).

        Returns:
            np.ndarray: Masque binaire (0 ou 1).
        """
        # Calculer la distance par rapport au centre
        x_center = self.decentering_mm[0]
        y_center = self.decentering_mm[1]
        
        x = grid_x_mm - x_center
        y = grid_y_mm - y_center
        
        if self.prism_type == PrismType.RIGHT_ANGLE:
            # Prisme à angle droit : forme triangulaire
            # Triangle rectangle avec base = base_length_mm, hauteur = height_mm
            mask = ((np.abs(x) <= self.base_length_mm / 2) & 
                    (np.abs(y) <= self.height_mm / 2) & 
                    (np.abs(x) + np.abs(y) <= self.base_length_mm / 2)).astype(float)
        elif self.prism_type == PrismType.EQUILATERAL:
            # Prisme équilatéral : forme triangulaire
            # Triangle équilatéral
            mask = ((np.abs(x) <= self.base_length_mm / 2) & 
                    (np.abs(y) <= self.height_mm / 2) & 
                    (np.abs(x) <= self.base_length_mm / 2 - np.abs(y) * self.base_length_mm / self.height_mm)).astype(float)
        else:
            # Par défaut : rectangulaire
            mask = ((np.abs(x) <= self.base_length_mm / 2) & 
                    (np.abs(y) <= self.height_mm / 2)).astype(float)
        
        return mask


# =============================================================================
# 5. FONCTIONS UTILITAIRES / UTILITY FUNCTIONS
# =============================================================================

def create_optic(
    optic_type: OpticType,
    name: str = "Optic",
    **kwargs,
) -> OpticalElement:
    """
    FR: Fabrique une optique de type spécifié.

    EN: Factory function to create an optic of the specified type.

    Args:
        optic_type (OpticType): Type d'optique à créer.
        name (str): Nom de l'optique.
        **kwargs: Arguments spécifiques au type d'optique.

    Returns:
        OpticalElement: L'optique créée.

    Raises:
        ValueError: Si le type d'optique est inconnu.
    """
    optic_classes = {
        OpticType.IDEAL_LENS: IdealLens,
        OpticType.SIMPLE_LENS: SimpleLens,
        OpticType.DOUBLE_LENS: DoubleLens,
        OpticType.CEMENTED_DOUBLET: CementedDoublet,
        OpticType.MIRROR: Mirror,
        OpticType.BEAMSPLITTER: Beamsplitter,
        OpticType.WINDOW: Window,
        OpticType.PRISM: Prism,
        OpticType.ASPHERIC_LENS: AsphericLens,
    }
    
    if optic_type not in optic_classes:
        raise ValueError(f"Type d'optique inconnu: {optic_type}")
    
    return optic_classes[optic_type](name=name, **kwargs)


def create_lens_from_preset(
    preset_name: str,
    diameter_mm: float,
    focal_length_mm: float,
    material_name: str = "BK7",
    **kwargs,
) -> OpticalElement:
    """
    FR: Crée une lentille à partir d'une configuration prédéfinie.

    EN: Creates a lens from a predefined configuration.

    Args:
        preset_name (str): Nom de la configuration prédéfinie.
        diameter_mm (float): Diamètre de la lentille en mm.
        focal_length_mm (float): Distance focale en mm.
        material_name (str): Nom du matériau.
        **kwargs: Arguments supplémentaires.

    Returns:
        OpticalElement: La lentille créée.

    Raises:
        ValueError: Si la configuration est inconnue.
    """
    presets = {
        # Lentilles plan-convexes
        "plan_convex_standard": {
            "type": OpticType.SIMPLE_LENS,
            "lens_type": LensType.PLAN_CONVEX,
            "curved_face_position": "front",
        },
        "plan_convex_thin": {
            "type": OpticType.SIMPLE_LENS,
            "lens_type": LensType.PLAN_CONVEX,
            "curved_face_position": "front",
            "thickness_mm": 3.0,
        },
        
        # Lentilles biconvexes
        "biconvex_standard": {
            "type": OpticType.DOUBLE_LENS,
            "lens_type": LensType.BICONVEX,
        },
        "biconvex_symmetrical": {
            "type": OpticType.DOUBLE_LENS,
            "lens_type": LensType.BICONVEX,
            "radius_of_curvature_1_mm": lambda f: f,  # R1 = f
            "radius_of_curvature_2_mm": lambda f: -f,  # R2 = -f (symétrique)
        },
        
        # Lentilles biconcaves
        "biconcave_standard": {
            "type": OpticType.DOUBLE_LENS,
            "lens_type": LensType.BICONCAVE,
        },
        
        # Lentilles ménisques
        "meniscus_convex_concave": {
            "type": OpticType.DOUBLE_LENS,
            "lens_type": LensType.MENISCUS,
            "radius_of_curvature_1_mm": lambda f: 2 * f,
            "radius_of_curvature_2_mm": lambda f: -3 * f,
        },
        
        # Lentilles idéales
        "ideal": {
            "type": OpticType.IDEAL_LENS,
        },
    }
    
    if preset_name not in presets:
        raise ValueError(f"Configuration prédéfinie inconnue: {preset_name}")
    
    preset = presets[preset_name]
    optic_type = preset["type"]
    
    # Préparer les arguments
    optic_kwargs = {**kwargs}
    optic_kwargs["name"] = f"{preset_name}_{diameter_mm}mm_f{focal_length_mm}mm"
    optic_kwargs["diameter_mm"] = diameter_mm
    optic_kwargs["material_name"] = material_name
    
    # Ajouter les paramètres spécifiques
    for key, value in preset.items():
        if key != "type":
            if callable(value):
                optic_kwargs[key] = value(focal_length_mm)
            else:
                optic_kwargs[key] = value
    
    # Ajouter la distance focale si nécessaire
    if optic_type == OpticType.IDEAL_LENS:
        optic_kwargs["focal_length_mm"] = focal_length_mm
    elif optic_type in [OpticType.SIMPLE_LENS, OpticType.DOUBLE_LENS]:
        # Calculer le rayon de courbure à partir de la distance focale
        # Formule du fabricant de lentilles : 1/f = (n - 1) * (1/R1 - 1/R2 + (n-1)*d/(n*R1*R2))
        # Approximation paraxiale : 1/f ≈ (n - 1) * (1/R1 - 1/R2)
        if material_name in ["Fused_Silica", "BK7", "SF5", "Silicon"]:
            from Material_Behaviour import MaterialBehaviour
            material = MaterialBehaviour(material_name)
            n = material.get_refractive_index(633.0)
        else:
            n = 1.5  # Indice par défaut
        
        if optic_type == OpticType.SIMPLE_LENS:
            # Lentille plan-convexe : 1/f = (n - 1) * (1/R)
            R = (n - 1) * focal_length_mm
            optic_kwargs["radius_of_curvature_mm"] = R
        elif optic_type == OpticType.DOUBLE_LENS:
            # Lentille biconvexe symétrique : 1/f = (n - 1) * (2/R)
            R = 2 * (n - 1) * focal_length_mm
            optic_kwargs["radius_of_curvature_1_mm"] = R
            optic_kwargs["radius_of_curvature_2_mm"] = -R
    
    return create_optic(optic_type, **optic_kwargs)


def create_doublet_from_preset(
    preset_name: str,
    diameter_mm: float,
    focal_length_mm: float,
    material1_name: str = "BK7",
    material2_name: str = "SF5",
    **kwargs,
) -> CementedDoublet:
    """
    FR: Crée un doublet collé à partir d'une configuration prédéfinie.

    EN: Creates a cemented doublet from a predefined configuration.

    Args:
        preset_name (str): Nom de la configuration prédéfinie.
        diameter_mm (float): Diamètre du doublet en mm.
        focal_length_mm (float): Distance focale en mm.
        material1_name (str): Nom du premier matériau.
        material2_name (str): Nom du deuxième matériau.
        **kwargs: Arguments supplémentaires.

    Returns:
        CementedDoublet: Le doublet créé.

    Raises:
        ValueError: Si la configuration est inconnue.
    """
    presets = {
        "achromatic_standard": {
            "lens1_type": OpticType.SIMPLE_LENS,
            "lens1_lens_type": LensType.PLAN_CONVEX,
            "lens1_curved_face_position": "front",
            "lens2_type": OpticType.SIMPLE_LENS,
            "lens2_lens_type": LensType.PLAN_CONCAVE,
            "lens2_curved_face_position": "back",
            "interface_radius_mm": lambda f: f,  # Interface sphérique
        },
        "achromatic_symmetrical": {
            "lens1_type": OpticType.DOUBLE_LENS,
            "lens1_lens_type": LensType.BICONVEX,
            "lens2_type": OpticType.DOUBLE_LENS,
            "lens2_lens_type": LensType.BICONCAVE,
            "interface_radius_mm": lambda f: f,
        },
    }
    
    if preset_name not in presets:
        raise ValueError(f"Configuration prédéfinie inconnue: {preset_name}")
    
    preset = presets[preset_name]
    
    # Créer les deux lentilles
    lens1 = create_lens_from_preset(
        preset_name="plan_convex_standard",
        diameter_mm=diameter_mm,
        focal_length_mm=focal_length_mm,
        material_name=material1_name,
    )
    
    lens2 = create_lens_from_preset(
        preset_name="plan_concave_standard",
        diameter_mm=diameter_mm,
        focal_length_mm=focal_length_mm,
        material_name=material2_name,
    )
    
    # Créer le doublet
    doublet = CementedDoublet(
        name=f"Doublet {preset_name}_{diameter_mm}mm_f{focal_length_mm}mm",
        lens1=lens1,
        lens2=lens2,
        interface_radius_mm=preset["interface_radius_mm"](focal_length_mm),
        material_name=f"{material1_name}+{material2_name}",
        **kwargs,
    )
    
    return doublet


# =============================================================================
# 6. CLASSE POUR LES SYSTÈMES OPTIQUES / OPTICAL SYSTEM CLASS
# =============================================================================

class OpticalSystem:
    """
    FR: Classe représentant un système optique composé de plusieurs éléments.
        Permet de propager un faisceau à travers plusieurs optiques.

    EN: Class representing an optical system composed of multiple elements.
        Allows propagating a beam through multiple optics.

    Attributes:
        elements (List[OpticalElement]): Liste des éléments optiques.
    """
    
    def __init__(self):
        """FR: Initialise un système optique vide."""
        self.elements: List[OpticalElement] = []
    
    def add_element(self, element: OpticalElement) -> None:
        """
        FR: Ajoute un élément optique au système.

        EN: Adds an optical element to the system.

        Args:
            element (OpticalElement): Élément optique à ajouter.
        """
        self.elements.append(element)
    
    def add_lens(
        self,
        name: str,
        focal_length_mm: float,
        diameter_mm: float,
        material_name: str = "BK7",
        position_z_mm: float = 0.0,
        **kwargs,
    ) -> OpticalElement:
        """
        FR: Ajoute une lentille au système.

        EN: Adds a lens to the system.

        Args:
            name (str): Nom de la lentille.
            focal_length_mm (float): Distance focale en mm.
            diameter_mm (float): Diamètre en mm.
            material_name (str): Nom du matériau.
            position_z_mm (float): Position en z en mm.
            **kwargs: Arguments supplémentaires pour la lentille.

        Returns:
            OpticalElement: La lentille créée.
        """
        # Créer une lentille idéale par défaut
        lens = IdealLens(
            name=name,
            focal_length_mm=focal_length_mm,
            diameter_mm=diameter_mm,
            material_name=material_name,
            position_mm=(0.0, 0.0, position_z_mm),
            **kwargs,
        )
        self.add_element(lens)
        return lens
    
    def add_mirror(
        self,
        name: str,
        diameter_mm: float,
        mirror_type: MirrorType = MirrorType.FLAT,
        radius_of_curvature_mm: Optional[float] = None,
        focal_length_mm: Optional[float] = None,
        position_z_mm: float = 0.0,
        tilt_deg: Tuple[float, float] = (0.0, 0.0),
        **kwargs,
    ) -> OpticalElement:
        """
        FR: Ajoute un miroir au système.

        EN: Adds a mirror to the system.

        Args:
            name (str): Nom du miroir.
            diameter_mm (float): Diamètre en mm.
            mirror_type (MirrorType): Type de miroir.
            radius_of_curvature_mm (float): Rayon de courbure en mm.
            focal_length_mm (float): Distance focale en mm.
            position_z_mm (float): Position en z en mm.
            tilt_deg (Tuple[float, float]): Inclinaison (θx, θy) en degrés.
            **kwargs: Arguments supplémentaires pour le miroir.

        Returns:
            OpticalElement: Le miroir créé.
        """
        mirror = Mirror(
            name=name,
            diameter_mm=diameter_mm,
            mirror_type=mirror_type,
            radius_of_curvature_mm=radius_of_curvature_mm,
            focal_length_mm=focal_length_mm,
            position_mm=(0.0, 0.0, position_z_mm),
            tilt_deg=tilt_deg,
            **kwargs,
        )
        self.add_element(mirror)
        return mirror
    
    def add_prism(
        self,
        name: str,
        diameter_mm: float,
        prism_type: PrismType = PrismType.RIGHT_ANGLE,
        apex_angle_deg: float = 60.0,
        material_name: str = "BK7",
        position_z_mm: float = 0.0,
        **kwargs,
    ) -> OpticalElement:
        """
        FR: Ajoute un prisme au système.

        EN: Adds a prism to the system.

        Args:
            name (str): Nom du prisme.
            diameter_mm (float): Diamètre en mm.
            prism_type (PrismType): Type de prisme.
            apex_angle_deg (float): Angle au sommet en degrés.
            material_name (str): Nom du matériau.
            position_z_mm (float): Position en z en mm.
            **kwargs: Arguments supplémentaires pour le prisme.

        Returns:
            OpticalElement: Le prisme créé.
        """
        prism = Prism(
            name=name,
            diameter_mm=diameter_mm,
            prism_type=prism_type,
            apex_angle_deg=apex_angle_deg,
            material_name=material_name,
            position_mm=(0.0, 0.0, position_z_mm),
            **kwargs,
        )
        self.add_element(prism)
        return prism
    
    def sort_elements_by_position(self) -> None:
        """
        FR: Trie les éléments optiques par position z croissante.

        EN: Sorts optical elements by increasing z position.
        """
        self.elements.sort(key=lambda x: x.position_mm[2])
    
    def propagate_beam(
        self,
        beam: 'Beam',
        initial_position_mm: float = 0.0,
        use_propagation: bool = True,
    ) -> 'Beam':
        """
        FR: Propage un faisceau à travers le système optique.
            Pour chaque optique, le faisceau est propagé jusqu'à l'optique,
            puis l'effet de l'optique est appliqué.

        EN: Propagates a beam through the optical system.
            For each optic, the beam is propagated to the optic,
            then the optic's effect is applied.

        Args:
            beam (Beam): Faisceau incident.
            initial_position_mm (float): Position initiale en z en mm.
            use_propagation (bool): Utiliser Propagation.py pour propager le faisceau entre les optiques.

        Returns:
            Beam: Faisceau après passage à travers le système.
        """
        from Propagation import Propagation
        
        # Trier les éléments par position
        self.sort_elements_by_position()
        
        current_position_z = initial_position_mm
        current_beam = beam
        
        for element in self.elements:
            element_z = element.position_mm[2]
            
            # Calculer la distance de propagation jusqu'à l'optique
            distance_mm = element_z - current_position_z
            
            if distance_mm > 0 and use_propagation:
                # Propager le faisceau jusqu'à l'optique
                propagator = Propagation(
                    wavelength_nm=current_beam.wavelength_nm,
                    propagation_distance_mm=distance_mm,
                    input_diameter_mm=current_beam.diameter_mm,
                    output_diameter_mm=current_beam.diameter_mm,
                    num_points=current_beam.num_points,
                    method="angular_spectrum",
                    coherence=current_beam.coherence,
                )
                
                # Propager le champ électrique
                propagated_field = propagator.propagate(current_beam.electric_field)
                
                # Mettre à jour le faisceau
                current_beam.electric_field = propagated_field
                current_beam.intensity = current_beam.compute_intensity_from_electric_field(propagated_field)
                current_beam.phase = current_beam.extract_phase_from_electric_field(propagated_field)
            
            # Appliquer l'optique au faisceau
            current_beam = element.apply_to_beam(current_beam)
            
            # Mettre à jour la position
            current_position_z = element_z
        
        return current_beam
    
    def get_total_phase_map(
        self,
        grid_x_mm: np.ndarray,
        grid_y_mm: np.ndarray,
    ) -> np.ndarray:
        """
        FR: Calcule la carte de phase totale du système optique.
            Somme les cartes de phase de tous les éléments.

        EN: Calculates the total phase map of the optical system.
            Sums the phase maps of all elements.

        Args:
            grid_x_mm (np.ndarray): Grille de positions en x (mm).
            grid_y_mm (np.ndarray): Grille de positions en y (mm).

        Returns:
            np.ndarray: Carte de phase totale en nm.
        """
        total_phase = np.zeros_like(grid_x_mm)
        
        for element in self.elements:
            phase_map = element.get_full_phase_map(grid_x_mm, grid_y_mm)
            total_phase += phase_map
        
        return total_phase
    
    def get_element_at_position(self, z_mm: float, tolerance_mm: float = 0.1) -> Optional[OpticalElement]:
        """
        FR: Retourne l'élément optique à une position z donnée.

        EN: Returns the optical element at a given z position.

        Args:
            z_mm (float): Position z en mm.
            tolerance_mm (float): Tolérance en mm.

        Returns:
            Optional[OpticalElement]: L'élément optique trouvé, ou None.
        """
        for element in self.elements:
            if abs(element.position_mm[2] - z_mm) <= tolerance_mm:
                return element
        return None


# =============================================================================
# 7. TESTS UNITAIRES / UNIT TESTS
# =============================================================================

class TestOptiques:
    """
    FR: Classe de tests unitaires pour Optiques.py.
    EN: Unit test class for Optiques.py.
    """

    def test_ideal_lens_phase(self):
        """Test la phase d'une lentille idéale."""
        lens = IdealLens(
            name="Test Ideal Lens",
            focal_length_mm=100.0,
            diameter_mm=10.0,
            wavelength_nm=633.0,
        )
        
        # Créer une grille
        grid_x, grid_y = create_grid(10.0, 256)
        
        # Calculer la phase
        phase = lens.get_phase_map(grid_x, grid_y)
        
        # Vérifier que la phase est quadratique
        r_squared = grid_x**2 + grid_y**2
        expected_phase = - (r_squared / (2 * 100.0)) * 1e9  # Simplification
        
        # La phase doit être négative (lentille convergente)
        assert np.all(phase <= 0), "La phase d'une lentille convergente doit être négative"
        
        # La phase au centre doit être nulle
        center = phase.shape[0] // 2, phase.shape[1] // 2
        assert abs(phase[center, center]) < 1e-6, "La phase au centre doit être nulle"

    def test_simple_lens_phase(self):
        """Test la phase d'une lentille simple."""
        lens = SimpleLens(
            name="Test Simple Lens",
            radius_of_curvature_mm=100.0,
            diameter_mm=10.0,
            thickness_mm=5.0,
            lens_type=LensType.PLAN_CONVEX,
            material_name="BK7",
            wavelength_nm=633.0,
        )
        
        grid_x, grid_y = create_grid(10.0, 256)
        phase = lens.get_phase_map(grid_x, grid_y)
        
        # La phase doit être positive (lentille convergente)
        assert np.all(phase >= 0), "La phase d'une lentille plan-convexe doit être positive"

    def test_mirror_phase(self):
        """Test la phase d'un miroir sphérique."""
        mirror = Mirror(
            name="Test Mirror",
            diameter_mm=10.0,
            mirror_type=MirrorType.SPHERICAL,
            radius_of_curvature_mm=200.0,  # f = 100 mm
            material_name="Aluminum",
            wavelength_nm=633.0,
        )
        
        grid_x, grid_y = create_grid(10.0, 256)
        phase = mirror.get_phase_map(grid_x, grid_y)
        
        # La phase doit être négative (miroir concave)
        assert np.all(phase <= 0), "La phase d'un miroir concave doit être négative"

    def test_window_phase(self):
        """Test la phase d'une fenêtre optique."""
        window = Window(
            name="Test Window",
            diameter_mm=10.0,
            thickness_mm=2.0,
            material_name="Fused_Silica",
            wavelength_nm=633.0,
        )
        
        grid_x, grid_y = create_grid(10.0, 256)
        phase = window.get_phase_map(grid_x, grid_y)
        
        # La phase doit être constante (fenêtre plate)
        assert np.allclose(phase, phase[0, 0], atol=1e-6), "La phase d'une fenêtre plate doit être constante"

    def test_aperture_mask(self):
        """Test le masque d'aperture."""
        lens = IdealLens(
            name="Test Lens",
            focal_length_mm=100.0,
            diameter_mm=10.0,
        )
        
        grid_x, grid_y = create_grid(20.0, 256)  # Grille plus grande que la lentille
        mask = lens.get_aperture_mask(grid_x, grid_y)
        
        # Le masque doit être 1 à l'intérieur du diamètre et 0 à l'extérieur
        r = np.sqrt(grid_x**2 + grid_y**2)
        expected_mask = (r <= 5.0).astype(float)  # Diamètre = 10 mm → rayon = 5 mm
        
        assert np.allclose(mask, expected_mask), "Le masque d'aperture est incorrect"

    def test_wfe_surface_roughness(self):
        """Test les erreurs de front d'onde (rugosité de surface)."""
        wfe = WaveFrontError(
            surface_roughness_nm=10.0,
            seed=42,
        )
        
        grid_x, grid_y = create_grid(10.0, 256)
        phase = wfe.generate_phase_map(grid_x, grid_y, 633.0, seed=42)
        
        # La phase doit avoir une RMS proche de 10 nm
        rms = np.std(phase)
        assert 8.0 < rms < 12.0, f"RMS attendu ~10 nm, obtenu {rms:.2f} nm"

    def test_wfe_parallelism(self):
        """Test les erreurs de front d'onde (parallélisme)."""
        wfe = WaveFrontError(
            parallelism_arcsec=10.0,  # 10 secondes d'arc
        )
        
        grid_x, grid_y = create_grid(10.0, 256)
        phase = wfe.generate_phase_map(grid_x, grid_y, 633.0)
        
        # La phase doit avoir une composante linéaire
        # Calculer la pente moyenne
        dx = grid_x[0, 1] - grid_x[0, 0]
        dy = grid_y[1, 0] - grid_y[0, 0]
        
        # Différence de phase entre les bords
        phase_diff_x = phase[0, -1] - phase[0, 0]
        phase_diff_y = phase[-1, 0] - phase[0, 0]
        
        # La phase doit varier linéairement
        assert abs(phase_diff_x) > 1.0, "La phase doit varier avec le parallélisme"

    def test_wfe_zernike(self):
        """Test les erreurs de front d'onde (Zernike)."""
        wfe = WaveFrontError(
            zernike_coefficients={(2, 0): 100.0},  # Defocus
        )
        
        grid_x, grid_y = create_grid(10.0, 256)
        phase = wfe.generate_phase_map(grid_x, grid_y, 633.0)
        
        # La phase doit avoir une forme quadratique (defocus)
        r_squared = grid_x**2 + grid_y**2
        corr = np.corrcoef(phase.flatten(), r_squared.flatten())[0, 1]
        
        assert abs(corr) > 0.9, "La phase doit être corrélée avec r² pour le defocus"

    def test_optical_system_propagation(self):
        """Test la propagation à travers un système optique."""
        system = OpticalSystem()
        
        # Ajouter une lentille
        system.add_lens(
            name="Lens 1",
            focal_length_mm=100.0,
            diameter_mm=10.0,
            position_z_mm=50.0,
        )
        
        # Créer un faisceau
        from Beam import Beam
        beam = Beam(
            wavelength_nm=633.0,
            diameter_mm=10.0,
            num_points=256,
        )
        electric_field = beam.generate_electric_field(method="gaussian")
        beam.electric_field = electric_field
        
        # Propager le faisceau
        propagated_beam = system.propagate_beam(beam, initial_position_mm=0.0, use_propagation=False)
        
        # Le faisceau doit avoir été modifié
        assert propagated_beam is not beam, "Le faisceau doit être modifié"

    def test_create_lens_from_preset(self):
        """Test la création de lentilles à partir de configurations prédéfinies."""
        lens = create_lens_from_preset(
            preset_name="ideal",
            diameter_mm=10.0,
            focal_length_mm=100.0,
        )
        
        assert isinstance(lens, IdealLens), "La lentille doit être de type IdealLens"
        assert lens.focal_length_mm == 100.0, "La distance focale doit être correcte"

    def test_create_optic_factory(self):
        """Test la fabrique d'optiques."""
        # Créer une lentille idéale
        lens = create_optic(
            optic_type=OpticType.IDEAL_LENS,
            name="Factory Lens",
            focal_length_mm=50.0,
            diameter_mm=10.0,
        )
        
        assert isinstance(lens, IdealLens), "L'optique doit être de type IdealLens"
        
        # Créer un miroir
        mirror = create_optic(
            optic_type=OpticType.MIRROR,
            name="Factory Mirror",
            diameter_mm=10.0,
            mirror_type=MirrorType.SPHERICAL,
            radius_of_curvature_mm=100.0,
        )
        
        assert isinstance(mirror, Mirror), "L'optique doit être de type Mirror"

    def test_cemented_doublet(self):
        """Test la création d'un doublet collé."""
        doublet = create_doublet_from_preset(
            preset_name="achromatic_standard",
            diameter_mm=10.0,
            focal_length_mm=100.0,
            material1_name="BK7",
            material2_name="SF5",
        )
        
        assert isinstance(doublet, CementedDoublet), "Le doublet doit être de type CementedDoublet"
        assert doublet.lens1 is not None, "La première lentille doit exister"
        assert doublet.lens2 is not None, "La deuxième lentille doit exister"
        
        # Calculer la phase
        grid_x, grid_y = create_grid(10.0, 256)
        phase = doublet.get_phase_map(grid_x, grid_y)
        
        # La phase doit être non nulle
        assert np.any(phase != 0), "La phase du doublet doit être non nulle"

    def test_prism_phase(self):
        """Test la phase d'un prisme."""
        prism = Prism(
            name="Test Prism",
            diameter_mm=10.0,
            prism_type=PrismType.RIGHT_ANGLE,
            apex_angle_deg=60.0,
            base_length_mm=10.0,
            height_mm=10.0,
            material_name="BK7",
            wavelength_nm=633.0,
        )
        
        grid_x, grid_y = create_grid(10.0, 256)
        phase = prism.get_phase_map(grid_x, grid_y)
        
        # La phase doit varier linéairement (caractéristique d'un prisme)
        # Vérifier que la phase n'est pas constante
        assert np.std(phase) > 0, "La phase d'un prisme doit varier"

    def test_shape_aperture_masks(self):
        """Test les masques d'aperture pour différentes formes."""
        shapes = [
            (ShapeType.CIRCULAR, {}),
            (ShapeType.RECTANGULAR, {"width_mm": 10.0, "height_mm": 5.0}),
            (ShapeType.SQUARE, {"side_mm": 8.0}),
            (ShapeType.ELLIPTICAL, {"semi_major_axis_mm": 5.0, "semi_minor_axis_mm": 3.0}),
            (ShapeType.HEXAGONAL, {"radius_mm": 5.0}),
            (ShapeType.OCTAGONAL, {"radius_mm": 5.0}),
        ]
        
        grid_x, grid_y = create_grid(20.0, 256)
        
        for shape_type, dimensions in shapes:
            lens = IdealLens(
                name=f"Lens {shape_type.value}",
                focal_length_mm=100.0,
                diameter_mm=10.0,
                shape_type=shape_type,
                shape_dimensions=dimensions,
            )
            
            mask = lens.get_aperture_mask(grid_x, grid_y)
            
            # Le masque doit être binaire
            assert np.all((mask == 0) | (mask == 1)), f"Le masque {shape_type.value} doit être binaire"
            
            # Le masque doit avoir des valeurs 1 (à l'intérieur)
            assert np.any(mask == 1), f"Le masque {shape_type.value} doit avoir des valeurs 1"


if __name__ == "__main__":
    import unittest
    unittest.main()

"""
Microstructure.py
FR: Module pour la gestion de matrices de micro-optiques (microlentilles, microtrous, microprismes, etc.).
    
    Fonctionnalités principales :
    - Création de matrices de micro-optiques de différents types
    - Gestion des espacements (bord à bord par défaut, ou centre à centre)
    - Application d'une WFE globale à la matrice complète
    - Prise en compte de la dilatation thermique et de son effet sur les positions
    - Intégration complète avec Optiques.py et Material_Behaviour.py
    
    Types de matrices supportés :
    - MicrolensArray : Matrice de microlentilles (idéales, simples, doubles, asphériques)
    - MicroholeArray : Matrice de microtrous (diaphragmes)
    - MicroprismArray : Matrice de microprismes
    - MicrogratingArray : Matrice de microréseaux de diffraction
    - Microstructure : Matrice générique (tout type d'optique)

EN: Module for managing micro-optics arrays (microlenses, microholes, microprisms, etc.).
    
    Main features:
    - Creation of micro-optics arrays of different types
    - Edge-to-edge spacing management (default) or center-to-center
    - Global WFE application to the entire array
    - Thermal expansion consideration and its effect on positions
    - Full integration with Optiques.py and Material_Behaviour.py
    
    Supported array types:
    - MicrolensArray: Array of microlenses (ideal, simple, double, aspheric)
    - MicroholeArray: Array of microholes (aperture stops)
    - MicroprismArray: Array of microprisms
    - MicrogratingArray: Array of micro diffraction gratings
    - Microstructure: Generic array (any optic type)

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch

Dependencies:
    - numpy
    - Optiques.py (for optical elements)
    - Material_Behaviour.py (for thermal expansion)
    - Beam.py (for beam manipulation)
    - MathAndPhysicsTools.py (for grid creation)
    - Visualization.py (for display, optional)
"""

import numpy as np
import os
import logging
from typing import Optional, Tuple, Dict, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# IMPORT DES DÉPENDANCES LOCALES / IMPORT LOCAL DEPENDENCIES
# =============================================================================

# Gestion des imports optionnels
try:
    from Optiques import (
        OpticalElement, OpticType, OpticSpecifications, ApertureShape,
        IdealLens, SimpleLens, DoubleLens, DoubletLens, Mirror, Beamsplitter,
        Window, Prism, AsphericLens, ApertureStop, DiffractionGrating,
        WaveFrontError, create_optic, LensType, MirrorType, BeamsplitterType, GratingType
    )
    OPTIQUES_AVAILABLE = True
except ImportError as e:
    OPTIQUES_AVAILABLE = False
    logging.warning(f"Optiques module not available: {e}. Microstructure features will be limited.")

try:
    from Material_Behaviour import MaterialBehaviour, STANDARD_TEMPERATURE_K
    MATERIAL_BEHAVIOUR_AVAILABLE = True
except ImportError as e:
    MATERIAL_BEHAVIOUR_AVAILABLE = False
    logging.warning(f"Material_Behaviour module not available: {e}. Thermal expansion features will be limited.")

try:
    from Beam import Beam
    BEAM_AVAILABLE = True
except ImportError as e:
    BEAM_AVAILABLE = False
    logging.warning(f"Beam module not available: {e}. Beam manipulation features will be limited.")

try:
    from MathAndPhysicsTools import create_grid
    MATH_TOOLS_AVAILABLE = True
except ImportError as e:
    MATH_TOOLS_AVAILABLE = False
    logging.warning(f"MathAndPhysicsTools module not available: {e}. Grid creation will be limited.")

try:
    from Visualization import plot_intensity, plot_phase
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    VISUALIZATION_AVAILABLE = False
    logging.warning(f"Visualization module not available: {e}. Display features will be disabled.")


# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Microstructure")


# =============================================================================
# 1. ENUMS ET CONSTANTES / ENUMS AND CONSTANTS
# =============================================================================

class MicroOpticType(Enum):
    """FR: Type de micro-optique."""
    LENS = "lens"                     # Microlentille
    HOLE = "hole"                     # Microtrou (diaphragme)
    PRISM = "prism"                   # Microprisme
    GRATING = "grating"               # Microréseau de diffraction
    CUSTOM = "custom"                 # Micro-optique personnalisée


class SpacingType(Enum):
    """FR: Type d'espacement entre les micro-optiques."""
    EDGE_TO_EDGE = "edge_to_edge"     # Bord à bord (défaut)
    CENTER_TO_CENTER = "center_to_center"  # Centre à centre


# =============================================================================
# 2. CLASSE POUR LES ÉLÉMENTS INDIVIDUELS / INDIVIDUAL ELEMENT CLASS
# =============================================================================

@dataclass
class MicroOpticElement:
    """
    FR: Élément individuel d'une microstructure.
        Contient une optique et sa position dans la matrice.

    EN: Individual element of a microstructure.
        Contains an optic and its position in the array.

    Attributes:
        optic (OpticalElement): L'optique elle-même.
        position_mm (Tuple[float, float]): Position (x, y) dans la matrice en mm.
        index (Tuple[int, int]): Indices (i, j) dans la matrice.
        element_diameter_mm (float): Diamètre de l'élément en mm.
    """
    optic: OpticalElement
    position_mm: Tuple[float, float]
    index: Tuple[int, int]
    element_diameter_mm: float


# =============================================================================
# 3. CLASSE DE BASE POUR LES MICROSTRUCTURES / BASE CLASS FOR MICROSTRUCTURES
# =============================================================================

class Microstructure:
    """
    FR: Matrice de micro-optiques.
        Permet de créer et manipuler des matrices de micro-optiques de différents types.
        
        Caractéristiques principales :
        - Espacement bord à bord par défaut (configurable)
        - Application d'une WFE globale à la matrice complète
        - Prise en compte de la dilatation thermique
        - Calcul de la phase totale de la matrice
        - Application à un faisceau

    EN: Micro-optics array.
        Allows creating and manipulating arrays of micro-optics of different types.
        
        Main features:
        - Edge-to-edge spacing by default (configurable)
        - Global WFE application to the entire array
        - Thermal expansion consideration
        - Total phase calculation for the array
        - Application to a beam

    Attributes:
        name (str): Nom de la microstructure.
        pitch_mm (float): Distance de bord à bord entre les éléments en mm.
        num_elements_x (int): Nombre d'éléments en x.
        num_elements_y (int): Nombre d'éléments en y.
        element_type (str): Type d'élément par défaut.
        element_kwargs (Dict): Arguments pour créer les éléments.
        spacing_type (SpacingType): Type d'espacement (EDGE_TO_EDGE ou CENTER_TO_CENTER).
        material_name (str): Nom du matériau de la matrice.
        temperature_K (float): Température en Kelvin.
        wavelength_nm (float): Longueur d'onde en nm.
        center_to_center_mm (float): Distance centre à centre calculée.
        elements (List[MicroOpticElement]): Liste des éléments de la matrice.
        global_wfe (WaveFrontError): Erreurs de front d'onde globales.
        material (MaterialBehaviour): Matériau de la matrice.
    """
    
    def __init__(
        self,
        name: str,
        pitch_mm: float,
        num_elements_x: int,
        num_elements_y: int,
        element_type: str = "lens",
        element_kwargs: Optional[Dict] = None,
        spacing_type: SpacingType = SpacingType.EDGE_TO_EDGE,
        material_name: str = "Fused_Silica",
        temperature_K: float = STANDARD_TEMPERATURE_K if MATERIAL_BEHAVIOUR_AVAILABLE else 293.15,
        wavelength_nm: float = 633.0,
        display: bool = False,
        display_dir: str = "output",
    ):
        """
        FR: Initialise une microstructure.

        EN: Initializes a microstructure.

        Args:
            name (str): Nom de la microstructure.
            pitch_mm (float): Distance de bord à bord entre les éléments en mm.
            num_elements_x (int): Nombre d'éléments en x.
            num_elements_y (int): Nombre d'éléments en y.
            element_type (str): Type d'élément par défaut ('lens', 'hole', 'prism', 'grating', 'custom').
            element_kwargs (Dict): Arguments pour créer les éléments.
            spacing_type (SpacingType): Type d'espacement (EDGE_TO_EDGE par défaut).
            material_name (str): Nom du matériau de la matrice.
            temperature_K (float): Température en Kelvin.
            wavelength_nm (float): Longueur d'onde en nm.
            display (bool): Afficher automatiquement les cartes.
            display_dir (str): Répertoire pour sauvegarder les images.
        """
        self.name = name
        self.pitch_mm = pitch_mm
        self.num_elements_x = num_elements_x
        self.num_elements_y = num_elements_y
        self.element_type = element_type
        self.element_kwargs = element_kwargs or {}
        self.spacing_type = spacing_type
        self.material_name = material_name
        self.temperature_K = temperature_K
        self.wavelength_nm = wavelength_nm
        self.display = display
        self.display_dir = display_dir
        
        # Calculer la distance centre à centre
        self._calculate_center_to_center()
        
        # Liste des éléments
        self.elements: List[MicroOpticElement] = []
        
        # WFE globale pour la matrice
        self.global_wfe: Optional[WaveFrontError] = None
        
        # Matériau de la matrice
        if MATERIAL_BEHAVIOUR_AVAILABLE:
            try:
                self.material = MaterialBehaviour(material_name)
            except Exception as e:
                logger.warning(f"Matériau inconnu: {material_name}. {e}")
                self.material = None
        else:
            self.material = None
        
        # Créer les éléments
        self._create_elements()
        
        # Créer le répertoire d'affichage si nécessaire
        if self.display:
            os.makedirs(self.display_dir, exist_ok=True)
    
    def _calculate_center_to_center(self) -> None:
        """
        FR: Calcule la distance centre à centre entre les éléments.
            Si spacing_type = EDGE_TO_EDGE : centre_to_centre = pitch + diamètre
            Si spacing_type = CENTER_TO_CENTER : centre_to_centre = pitch
        """
        # Déterminer le diamètre des éléments
        if 'diameter_mm' in self.element_kwargs:
            element_diameter = self.element_kwargs['diameter_mm']
        else:
            # Par défaut, diamètre = pitch * clear_aperture_ratio
            clear_aperture_ratio = self.element_kwargs.get('clear_aperture_ratio', 0.9)
            element_diameter = self.pitch_mm * clear_aperture_ratio
        
        self.element_diameter_mm = element_diameter
        
        if self.spacing_type == SpacingType.EDGE_TO_EDGE:
            # Distance centre à centre = distance bord à bord + diamètre
            self.center_to_center_mm = self.pitch_mm + self.element_diameter_mm
        else:  # CENTER_TO_CENTER
            self.center_to_center_mm = self.pitch_mm
    
    def _create_elements(self) -> None:
        """
        FR: Crée les éléments de la microstructure.
            Chaque élément est positionné selon la grille définie par num_elements_x et num_elements_y.
        """
        self.elements = []
        
        for i in range(self.num_elements_x):
            for j in range(self.num_elements_y):
                # Calculer la position (x, y) du centre de l'élément
                x = (i - (self.num_elements_x - 1) / 2) * self.center_to_center_mm
                y = (j - (self.num_elements_y - 1) / 2) * self.center_to_center_mm
                
                # Créer l'optique
                optic = self._create_optic(i, j, (x, y))
                
                # Ajouter à la liste
                self.elements.append(MicroOpticElement(
                    optic=optic,
                    position_mm=(x, y),
                    index=(i, j),
                    element_diameter_mm=self.element_diameter_mm,
                ))
    
    def _create_optic(self, i: int, j: int, position_mm: Tuple[float, float]) -> OpticalElement:
        """
        FR: Crée une optique en fonction du type d'élément.
            Doit être surchargée dans les classes dérivées pour des types spécifiques.

        EN: Creates an optic based on the element type.
            Should be overridden in derived classes for specific types.

        Args:
            i (int): Indice x de l'élément.
            j (int): Indice y de l'élément.
            position_mm (Tuple[float, float]): Position (x, y) de l'élément.

        Returns:
            OpticalElement: L'optique créée.
        """
        if not OPTIQUES_AVAILABLE:
            raise ImportError("Optiques module is required for _create_optic().")
        
        # Préparer les arguments
        kwargs = self.element_kwargs.copy()
        kwargs['wavelength_nm'] = self.wavelength_nm
        kwargs['temperature_K'] = self.temperature_K
        kwargs['position_mm'] = (*position_mm, 0.0)  # z = 0
        
        # Définir le diamètre si non spécifié
        if 'diameter_mm' not in kwargs:
            kwargs['diameter_mm'] = self.element_diameter_mm
        
        # Créer l'optique en fonction du type
        if self.element_type == "lens" or self.element_type == "microlens":
            # Par défaut : lentille idéale
            if 'focal_length_mm' not in kwargs:
                kwargs['focal_length_mm'] = 10.0  # Distance focale par défaut
            return IdealLens(name=f"Microlentille ({i},{j})", **kwargs)
        
        elif self.element_type == "hole" or self.element_type == "microhole":
            return ApertureStop(name=f"Microtrou ({i},{j})", **kwargs)
        
        elif self.element_type == "prism" or self.element_type == "microprism":
            if 'apex_angle_deg' not in kwargs:
                kwargs['apex_angle_deg'] = 10.0
            if 'base_length_mm' not in kwargs:
                kwargs['base_length_mm'] = self.element_diameter_mm
            if 'height_mm' not in kwargs:
                kwargs['height_mm'] = self.element_diameter_mm
            return Prism(name=f"Microprisme ({i},{j})", **kwargs)
        
        elif self.element_type == "grating" or self.element_type == "micrograting":
            if 'lines_per_mm' not in kwargs:
                kwargs['lines_per_mm'] = 100.0
            return DiffractionGrating(name=f"Microréseau ({i},{j})", **kwargs)
        
        elif self.element_type == "custom":
            # Utiliser la classe spécifiée dans element_kwargs
            optic_class = kwargs.pop('optic_class', IdealLens)
            return optic_class(name=f"Micro-optique ({i},{j})", **kwargs)
        
        else:
            raise ValueError(f"Type d'élément inconnu: {self.element_type}")
    
    def get_phase_map(
        self,
        grid_x_mm: np.ndarray,
        grid_y_mm: np.ndarray,
        include_global_wfe: bool = True,
    ) -> np.ndarray:
        """
        FR: Calcule la carte de phase totale de la matrice.
            Somme les cartes de phase de tous les éléments, en tenant compte de leur position.
            Ajoute la WFE globale si include_global_wfe = True.

        EN: Calculates the total phase map of the array.
            Sums the phase maps of all elements, taking into account their position.
            Adds the global WFE if include_global_wfe = True.

        Args:
            grid_x_mm (np.ndarray): Grille de positions en x (mm).
            grid_y_mm (np.ndarray): Grille de positions en y (mm).
            include_global_wfe (bool): Inclure la WFE globale.

        Returns:
            np.ndarray: Carte de phase totale en nm.
        """
        if not OPTIQUES_AVAILABLE:
            raise ImportError("Optiques module is required for get_phase_map().")
        
        # Initialiser la carte de phase totale
        total_phase = np.zeros_like(grid_x_mm)
        
        # Ajouter la contribution de chaque élément
        for element in self.elements:
            # Déplacer la grille pour chaque élément
            x_shifted = grid_x_mm - element.position_mm[0]
            y_shifted = grid_y_mm - element.position_mm[1]
            
            # Calculer la phase de l'élément
            phase = element.optic.get_phase_map(x_shifted, y_shifted)
            
            # Calculer le masque de l'élément
            mask = element.optic.get_aperture_mask(x_shifted, y_shifted)
            
            # Ajouter la phase masquée
            total_phase += phase * mask
        
        # Ajouter la WFE globale si activé
        if include_global_wfe and self.global_wfe is not None:
            wfe_phase = self.global_wfe.generate_phase_map(
                grid_x_mm, grid_y_mm, self.wavelength_nm
            )
            total_phase += wfe_phase
        
        return total_phase
    
    def get_transmission_map(
        self,
        grid_x_mm: np.ndarray,
        grid_y_mm: np.ndarray,
    ) -> np.ndarray:
        """
        FR: Calcule la carte de transmission totale de la matrice.
            Pour chaque point, la transmission est la transmission maximale parmi tous les éléments
            qui couvrent ce point.

        EN: Calculates the total transmission map of the array.
            For each point, the transmission is the maximum transmission among all elements
            that cover that point.

        Args:
            grid_x_mm (np.ndarray): Grille de positions en x (mm).
            grid_y_mm (np.ndarray): Grille de positions en y (mm).

        Returns:
            np.ndarray: Carte de transmission (0-1).
        """
        if not OPTIQUES_AVAILABLE:
            raise ImportError("Optiques module is required for get_transmission_map().")
        
        total_transmission = np.zeros_like(grid_x_mm)
        
        for element in self.elements:
            x_shifted = grid_x_mm - element.position_mm[0]
            y_shifted = grid_y_mm - element.position_mm[1]
            
            transmission = element.optic.get_transmission_map(x_shifted, y_shifted)
            total_transmission = np.maximum(total_transmission, transmission)
        
        return total_transmission
    
    def get_intensity_map(
        self,
        grid_x_mm: np.ndarray,
        grid_y_mm: np.ndarray,
    ) -> np.ndarray:
        """
        FR: Calcule la carte d'intensité totale de la matrice.
            Pour un diaphragme, l'intensité est 1 à l'intérieur et 0 à l'extérieur.
            Pour une matrice de microlentilles, l'intensité est la somme des intensités
            de chaque élément.

        EN: Calculates the total intensity map of the array.

        Args:
            grid_x_mm (np.ndarray): Grille de positions en x (mm).
            grid_y_mm (np.ndarray): Grille de positions en y (mm).

        Returns:
            np.ndarray: Carte d'intensité (0-1).
        """
        if not OPTIQUES_AVAILABLE:
            raise ImportError("Optiques module is required for get_intensity_map().")
        
        # Pour une matrice de microtrous, l'intensité est la transmission
        if self.element_type == "hole" or self.element_type == "microhole":
            return self.get_transmission_map(grid_x_mm, grid_y_mm)
        
        # Pour une matrice de microlentilles, l'intensité est la somme des masques
        total_intensity = np.zeros_like(grid_x_mm)
        
        for element in self.elements:
            x_shifted = grid_x_mm - element.position_mm[0]
            y_shifted = grid_y_mm - element.position_mm[1]
            
            mask = element.optic.get_aperture_mask(x_shifted, y_shifted)
            total_intensity += mask
        
        # Normaliser pour que l'intensité maximale soit 1
        if np.max(total_intensity) > 0:
            total_intensity /= np.max(total_intensity)
        
        return total_intensity
    
    def apply_to_beam(self, beam: any) -> any:
        """
        FR: Applique la matrice à un faisceau.
            Calcule la phase totale de la matrice et l'applique au faisceau.
            Affiche automatiquement les cartes si display=True.

        EN: Applies the array to a beam.
            Calculates the total phase of the array and applies it to the beam.
            Automatically displays maps if display=True.

        Args:
            beam (Beam): Faisceau incident.

        Returns:
            Beam: Faisceau après passage à travers la matrice.
        """
        if not BEAM_AVAILABLE:
            raise ImportError("Beam module is required for apply_to_beam().")
        if not OPTIQUES_AVAILABLE:
            raise ImportError("Optiques module is required for apply_to_beam().")
        
        # Créer une grille correspondant au faisceau
        if MATH_TOOLS_AVAILABLE:
            grid_x_mm, grid_y_mm = create_grid(beam.diameter_mm, beam.num_points)
        else:
            x = np.linspace(-beam.diameter_mm/2, beam.diameter_mm/2, beam.num_points)
            y = np.linspace(-beam.diameter_mm/2, beam.diameter_mm/2, beam.num_points)
            grid_x_mm, grid_y_mm = np.meshgrid(x, y)
        
        # Affichage avant application si activé
        if self.display and VISUALIZATION_AVAILABLE:
            os.makedirs(self.display_dir, exist_ok=True)
            if hasattr(beam, 'intensity') and beam.intensity is not None:
                plot_intensity(
                    beam.intensity,
                    beam.diameter_mm,
                    title=f"Before {self.name}",
                    intensity_unit="a.u.",
                    save_path=os.path.join(self.display_dir, f"before_{self.name}_intensity.png")
                )
            if hasattr(beam, 'phase') and beam.phase is not None:
                plot_phase(
                    beam.phase,
                    beam.diameter_mm,
                    title=f"Before {self.name}",
                    save_path=os.path.join(self.display_dir, f"before_{self.name}_phase.png")
                )
        
        # Calculer la carte de phase totale
        total_phase = self.get_phase_map(grid_x_mm, grid_y_mm)
        
        # Créer un nouveau faisceau
        new_beam = Beam(
            wavelength_nm=beam.wavelength_nm,
            diameter_mm=beam.diameter_mm,
            energy=beam.energy,
            num_points=beam.num_points,
            coherence=beam.coherence,
        )
        
        # Appliquer la phase
        if beam.electric_field is not None:
            amplitude = np.abs(beam.electric_field)
            initial_phase = np.angle(beam.electric_field)
            phase_rad = total_phase * 2 * np.pi / self.wavelength_nm
            new_phase = initial_phase + phase_rad
            new_electric_field = amplitude * np.exp(1j * new_phase)
            new_beam.electric_field = new_electric_field
            new_beam.intensity = new_beam.compute_intensity_from_electric_field(new_electric_field)
            new_beam.phase = new_beam.extract_phase_from_electric_field(new_electric_field)
        
        # Affichage après application si activé
        if self.display and VISUALIZATION_AVAILABLE:
            if hasattr(new_beam, 'intensity') and new_beam.intensity is not None:
                plot_intensity(
                    new_beam.intensity,
                    new_beam.diameter_mm,
                    title=f"After {self.name}",
                    intensity_unit="a.u.",
                    save_path=os.path.join(self.display_dir, f"after_{self.name}_intensity.png")
                )
            if hasattr(new_beam, 'phase') and new_beam.phase is not None:
                plot_phase(
                    new_beam.phase,
                    new_beam.diameter_mm,
                    title=f"After {self.name}",
                    save_path=os.path.join(self.display_dir, f"after_{self.name}_phase.png")
                )
        
        return new_beam
    
    def apply_global_wfe(
        self,
        wfe: WaveFrontError,
    ) -> None:
        """
        FR: Applique une WFE globale à la matrice.
            Cette WFE sera ajoutée à la phase totale de la matrice.

        EN: Applies a global WFE to the array.
            This WFE will be added to the total phase of the array.

        Args:
            wfe (WaveFrontError): Erreurs de front d'onde globales.
        """
        self.global_wfe = wfe
    
    def update_temperature(
        self,
        new_temperature_K: float,
    ) -> None:
        """
        FR: Met à jour la température de la matrice et recalcule les positions des éléments.
            Prend en compte la dilatation thermique du matériau.

        EN: Updates the temperature of the array and recalculates the positions of the elements.
            Takes into account the thermal expansion of the material.

        Args:
            new_temperature_K (float): Nouvelle température en Kelvin.
        """
        if self.material is None:
            logger.warning("Matériau non défini, la dilatation thermique ne peut pas être calculée.")
            return
        
        # Calculer la variation de température
        delta_T = new_temperature_K - self.temperature_K
        
        if delta_T == 0:
            return
        
        # Calculer le coefficient de dilatation thermique
        try:
            thermal_expansion = self.material.get_thermal_expansion(new_temperature_K, self.temperature_K)
        except Exception as e:
            logger.warning(f"Impossible de calculer la dilatation thermique: {e}")
            return
        
        # Mettre à jour le pitch (distance bord à bord)
        if self.spacing_type == SpacingType.EDGE_TO_EDGE:
            # Le pitch est multiplié par (1 + αΔT)
            self.pitch_mm *= (1 + thermal_expansion)
            # Recalculer le centre à centre
            self._calculate_center_to_center()
        else:  # CENTER_TO_CENTER
            # Le centre à centre est multiplié par (1 + αΔT)
            self.center_to_center_mm *= (1 + thermal_expansion)
            # Recalculer le pitch
            self.pitch_mm = self.center_to_center_mm - self.element_diameter_mm
        
        # Mettre à jour les positions des éléments
        for element in self.elements:
            i, j = element.index
            x = (i - (self.num_elements_x - 1) / 2) * self.center_to_center_mm
            y = (j - (self.num_elements_y - 1) / 2) * self.center_to_center_mm
            element.position_mm = (x, y)
            element.optic.position_mm = (x, y, 0.0)
        
        # Mettre à jour la température
        self.temperature_K = new_temperature_K
        for element in self.elements:
            element.optic.temperature_K = new_temperature_K
    
    def get_element_spacing(self) -> Dict[str, float]:
        """
        FR: Retourne les informations d'espacement entre les éléments.

        EN: Returns spacing information between elements.

        Returns:
            Dict[str, float]: Dictionnaire avec les informations d'espacement.
        """
        if self.spacing_type == SpacingType.EDGE_TO_EDGE:
            return {
                "spacing_type": "edge_to_edge",
                "edge_to_edge_mm": self.pitch_mm,
                "center_to_center_mm": self.center_to_center_mm,
                "element_diameter_mm": self.element_diameter_mm,
                "gap_mm": self.pitch_mm,  # Espace entre les bords
            }
        else:
            return {
                "spacing_type": "center_to_center",
                "center_to_center_mm": self.center_to_center_mm,
                "element_diameter_mm": self.element_diameter_mm,
                "gap_mm": self.center_to_center_mm - self.element_diameter_mm,  # Espace entre les bords
            }
    
    def get_element(self, i: int, j: int) -> Optional[MicroOpticElement]:
        """
        FR: Retourne l'élément aux indices (i, j).

        EN: Returns the element at indices (i, j).

        Args:
            i (int): Indice x.
            j (int): Indice y.

        Returns:
            Optional[MicroOpticElement]: L'élément trouvé, ou None.
        """
        for element in self.elements:
            if element.index == (i, j):
                return element
        return None
    
    def get_element_at_position(
        self,
        x_mm: float,
        y_mm: float,
        tolerance_mm: float = 0.1,
    ) -> Optional[MicroOpticElement]:
        """
        FR: Retourne l'élément à une position (x, y) donnée.

        EN: Returns the element at a given position (x, y).

        Args:
            x_mm (float): Position x en mm.
            y_mm (float): Position y en mm.
            tolerance_mm (float): Tolérance en mm.

        Returns:
            Optional[MicroOpticElement]: L'élément trouvé, ou None.
        """
        for element in self.elements:
            dx = abs(element.position_mm[0] - x_mm)
            dy = abs(element.position_mm[1] - y_mm)
            if dx <= tolerance_mm and dy <= tolerance_mm:
                return element
        return None
    
    def get_total_phase_at_point(self, x_mm: float, y_mm: float) -> float:
        """
        FR: Calcule la phase totale à un point (x, y) donné.

        EN: Calculates the total phase at a given point (x, y).

        Args:
            x_mm (float): Position x en mm.
            y_mm (float): Position y en mm.

        Returns:
            float: Phase totale en nm.
        """
        total_phase = 0.0
        
        for element in self.elements:
            x_shifted = x_mm - element.position_mm[0]
            y_shifted = y_mm - element.position_mm[1]
            
            # Vérifier si le point est dans l'aperture de l'élément
            mask = element.optic.get_aperture_mask(
                np.array([[x_shifted]]),
                np.array([[y_shifted]])
            )
            
            if mask[0, 0] > 0:
                # Calculer la phase de l'élément à ce point
                phase = element.optic.get_phase_map(
                    np.array([[x_shifted]]),
                    np.array([[y_shifted]])
                )
                total_phase += phase[0, 0]
        
        # Ajouter la WFE globale
        if self.global_wfe is not None:
            wfe_phase = self.global_wfe.generate_phase_map(
                np.array([[x_mm]]),
                np.array([[y_mm]]),
                self.wavelength_nm
            )
            total_phase += wfe_phase[0, 0]
        
        return total_phase
    
    def visualize(
        self,
        grid_size_mm: float = 10.0,
        num_points: int = 256,
        save_dir: Optional[str] = None,
    ) -> None:
        """
        FR: Visualise la matrice (phase et transmission).

        EN: Visualizes the array (phase and transmission).

        Args:
            grid_size_mm (float): Taille de la grille en mm.
            num_points (int): Nombre de points dans la grille.
            save_dir (str): Répertoire pour sauvegarder les images.
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Visualization module not available. Cannot visualize.")
            return
        
        if save_dir is None:
            save_dir = self.display_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        if MATH_TOOLS_AVAILABLE:
            grid_x, grid_y = create_grid(grid_size_mm, num_points)
        else:
            x = np.linspace(-grid_size_mm/2, grid_size_mm/2, num_points)
            y = np.linspace(-grid_size_mm/2, grid_size_mm/2, num_points)
            grid_x, grid_y = np.meshgrid(x, y)
        
        # Visualiser la phase
        phase_map = self.get_phase_map(grid_x, grid_y)
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Phase
        im1 = axes[0].imshow(
            phase_map,
            extent=[-grid_size_mm/2, grid_size_mm/2, -grid_size_mm/2, grid_size_mm/2],
            cmap='coolwarm'
        )
        axes[0].set_title(f"{self.name} - Phase totale")
        axes[0].set_xlabel("x (mm)")
        axes[0].set_ylabel("y (mm)")
        plt.colorbar(im1, ax=axes[0], label="Phase (nm)")
        
        # Transmission
        transmission_map = self.get_transmission_map(grid_x, grid_y)
        im2 = axes[1].imshow(
            transmission_map,
            extent=[-grid_size_mm/2, grid_size_mm/2, -grid_size_mm/2, grid_size_mm/2],
            cmap='gray'
        )
        axes[1].set_title(f"{self.name} - Transmission")
        axes[1].set_xlabel("x (mm)")
        axes[1].set_ylabel("y (mm)")
        plt.colorbar(im2, ax=axes[1], label="Transmission")
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{self.name}_visualization.png"), dpi=150, bbox_inches='tight')
        plt.close('all')


# =============================================================================
# 4. CLASSES SPÉCIFIQUES / SPECIFIC CLASSES
# =============================================================================

class MicrolensArray(Microstructure):
    """
    FR: Matrice de microlentilles.
        Permet de créer des matrices de microlentilles de différents types.

    EN: Microlens array.
        Allows creating arrays of microlenses of different types.
    """
    
    def __init__(
        self,
        name: str,
        pitch_mm: float,
        num_elements_x: int,
        num_elements_y: int,
        focal_length_mm: float = 10.0,
        lens_type: str = "ideal",  # "ideal", "simple", "double", "aspheric", "doublet"
        material_name: str = "Fused_Silica",
        spacing_type: SpacingType = SpacingType.EDGE_TO_EDGE,
        temperature_K: float = STANDARD_TEMPERATURE_K if MATERIAL_BEHAVIOUR_AVAILABLE else 293.15,
        wavelength_nm: float = 633.0,
        display: bool = False,
        display_dir: str = "output",
    ):
        """
        FR: Initialise une matrice de microlentilles.

        EN: Initializes a microlens array.

        Args:
            name (str): Nom de la matrice.
            pitch_mm (float): Distance de bord à bord entre les lentilles en mm.
            num_elements_x (int): Nombre de lentilles en x.
            num_elements_y (int): Nombre de lentilles en y.
            focal_length_mm (float): Distance focale des lentilles en mm.
            lens_type (str): Type de lentille ("ideal", "simple", "double", "aspheric", "doublet").
            material_name (str): Nom du matériau.
            spacing_type (SpacingType): Type d'espacement.
            temperature_K (float): Température en Kelvin.
            wavelength_nm (float): Longueur d'onde en nm.
            display (bool): Afficher automatiquement les cartes.
            display_dir (str): Répertoire pour sauvegarder les images.
        """
        # Préparer les arguments pour les lentilles
        element_kwargs = {
            'focal_length_mm': focal_length_mm,
            'material_name': material_name,
            'wavelength_nm': wavelength_nm,
            'temperature_K': temperature_K,
        }
        
        # Définir le type d'élément
        if lens_type == "simple":
            element_type = "simple_lens"
            element_kwargs['radius_of_curvature_mm'] = 2 * focal_length_mm * 1.5  # Approximation
            element_kwargs['thickness_mm'] = 2.0
            element_kwargs['lens_type'] = LensType.PLAN_CONVEX
        elif lens_type == "double":
            element_type = "double_lens"
            element_kwargs['radius_of_curvature_1_mm'] = 2 * focal_length_mm * 1.5
            element_kwargs['radius_of_curvature_2_mm'] = -2 * focal_length_mm * 1.5
            element_kwargs['thickness_mm'] = 3.0
            element_kwargs['lens_type'] = LensType.BICONVEX
        elif lens_type == "aspheric":
            element_type = "aspheric_lens"
            element_kwargs['radius_of_curvature_mm'] = focal_length_mm
            element_kwargs['thickness_mm'] = 2.0
            element_kwargs['conic_constant'] = 0.0
        elif lens_type == "doublet":
            element_type = "doublet_lens"
            # Pour un doublet, on utilise des valeurs par défaut
            element_kwargs['radius_of_curvature_1_mm'] = 100.0
            element_kwargs['radius_of_curvature_2_mm'] = -50.0
            element_kwargs['thickness_1_mm'] = 3.0
            element_kwargs['material_1_name'] = "BK7"
            element_kwargs['radius_of_curvature_3_mm'] = 50.0
            element_kwargs['radius_of_curvature_4_mm'] = -200.0
            element_kwargs['thickness_2_mm'] = 2.0
            element_kwargs['material_2_name'] = "SF5"
        else:  # ideal
            element_type = "lens"
        
        super().__init__(
            name=name,
            pitch_mm=pitch_mm,
            num_elements_x=num_elements_x,
            num_elements_y=num_elements_y,
            element_type=element_type,
            element_kwargs=element_kwargs,
            spacing_type=spacing_type,
            material_name=material_name,
            temperature_K=temperature_K,
            wavelength_nm=wavelength_nm,
            display=display,
            display_dir=display_dir,
        )
    
    def _create_optic(self, i: int, j: int, position_mm: Tuple[float, float]) -> OpticalElement:
        """FR: Crée une microlentille."""
        if not OPTIQUES_AVAILABLE:
            raise ImportError("Optiques module is required for _create_optic().")
        
        kwargs = self.element_kwargs.copy()
        kwargs['diameter_mm'] = self.element_diameter_mm
        kwargs['position_mm'] = (*position_mm, 0.0)
        kwargs['wavelength_nm'] = self.wavelength_nm
        kwargs['temperature_K'] = self.temperature_K
        
        if self.element_type == "simple_lens":
            return SimpleLens(name=f"Microlentille ({i},{j})", **kwargs)
        elif self.element_type == "double_lens":
            return DoubleLens(name=f"Microlentille ({i},{j})", **kwargs)
        elif self.element_type == "aspheric_lens":
            return AsphericLens(name=f"Microlentille ({i},{j})", **kwargs)
        elif self.element_type == "doublet_lens":
            return DoubletLens(name=f"Microlentille ({i},{j})", **kwargs)
        else:  # ideal
            return IdealLens(name=f"Microlentille ({i},{j})", **kwargs)


class MicroholeArray(Microstructure):
    """
    FR: Matrice de microtrous (diaphragmes).
        Permet de créer des masques de phase avec des trous.

    EN: Microhole array (aperture stops).
        Allows creating phase masks with holes.
    """
    
    def __init__(
        self,
        name: str,
        pitch_mm: float,
        num_elements_x: int,
        num_elements_y: int,
        hole_diameter_mm: Optional[float] = None,
        aperture_shape: ApertureShape = ApertureShape.CIRCULAR,
        material_name: str = "opaque",
        spacing_type: SpacingType = SpacingType.EDGE_TO_EDGE,
        temperature_K: float = STANDARD_TEMPERATURE_K if MATERIAL_BEHAVIOUR_AVAILABLE else 293.15,
        wavelength_nm: float = 633.0,
        display: bool = False,
        display_dir: str = "output",
    ):
        """
        FR: Initialise une matrice de microtrous.

        EN: Initializes a microhole array.

        Args:
            name (str): Nom de la matrice.
            pitch_mm (float): Distance de bord à bord entre les trous en mm.
            num_elements_x (int): Nombre de trous en x.
            num_elements_y (int): Nombre de trous en y.
            hole_diameter_mm (float): Diamètre des trous en mm.
            aperture_shape (ApertureShape): Forme des trous.
            material_name (str): Nom du matériau (opaque par défaut).
            spacing_type (SpacingType): Type d'espacement.
            temperature_K (float): Température en Kelvin.
            wavelength_nm (float): Longueur d'onde en nm.
            display (bool): Afficher automatiquement les cartes.
            display_dir (str): Répertoire pour sauvegarder les images.
        """
        # Préparer les arguments pour les trous
        element_kwargs = {
            'material_name': material_name,
            'aperture_shape': aperture_shape,
            'wavelength_nm': wavelength_nm,
            'temperature_K': temperature_K,
        }
        
        if hole_diameter_mm is not None:
            element_kwargs['diameter_mm'] = hole_diameter_mm
        
        super().__init__(
            name=name,
            pitch_mm=pitch_mm,
            num_elements_x=num_elements_x,
            num_elements_y=num_elements_y,
            element_type="hole",
            element_kwargs=element_kwargs,
            spacing_type=spacing_type,
            material_name=material_name,
            temperature_K=temperature_K,
            wavelength_nm=wavelength_nm,
            display=display,
            display_dir=display_dir,
        )
    
    def _create_optic(self, i: int, j: int, position_mm: Tuple[float, float]) -> OpticalElement:
        """FR: Crée un microtrou."""
        if not OPTIQUES_AVAILABLE:
            raise ImportError("Optiques module is required for _create_optic().")
        
        kwargs = self.element_kwargs.copy()
        kwargs['diameter_mm'] = self.element_diameter_mm
        kwargs['position_mm'] = (*position_mm, 0.0)
        kwargs['wavelength_nm'] = self.wavelength_nm
        kwargs['temperature_K'] = self.temperature_K
        
        return ApertureStop(name=f"Microtrou ({i},{j})", **kwargs)


class MicroprismArray(Microstructure):
    """
    FR: Matrice de microprismes.
        Permet de créer des matrices de microprismes pour dévier la lumière.

    EN: Microprism array.
        Allows creating arrays of microprisms to deviate light.
    """
    
    def __init__(
        self,
        name: str,
        pitch_mm: float,
        num_elements_x: int,
        num_elements_y: int,
        apex_angle_deg: float = 10.0,
        base_length_mm: Optional[float] = None,
        height_mm: Optional[float] = None,
        orientation_deg: float = 0.0,
        material_name: str = "Fused_Silica",
        spacing_type: SpacingType = SpacingType.EDGE_TO_EDGE,
        temperature_K: float = STANDARD_TEMPERATURE_K if MATERIAL_BEHAVIOUR_AVAILABLE else 293.15,
        wavelength_nm: float = 633.0,
        display: bool = False,
        display_dir: str = "output",
    ):
        """
        FR: Initialise une matrice de microprismes.

        EN: Initializes a microprism array.

        Args:
            name (str): Nom de la matrice.
            pitch_mm (float): Distance de bord à bord entre les prismes en mm.
            num_elements_x (int): Nombre de prismes en x.
            num_elements_y (int): Nombre de prismes en y.
            apex_angle_deg (float): Angle au sommet des prismes en degrés.
            base_length_mm (float): Longueur de la base des prismes en mm.
            height_mm (float): Hauteur des prismes en mm.
            orientation_deg (float): Orientation des prismes en degrés.
            material_name (str): Nom du matériau.
            spacing_type (SpacingType): Type d'espacement.
            temperature_K (float): Température en Kelvin.
            wavelength_nm (float): Longueur d'onde en nm.
            display (bool): Afficher automatiquement les cartes.
            display_dir (str): Répertoire pour sauvegarder les images.
        """
        # Préparer les arguments pour les prismes
        element_kwargs = {
            'apex_angle_deg': apex_angle_deg,
            'orientation_deg': orientation_deg,
            'material_name': material_name,
            'wavelength_nm': wavelength_nm,
            'temperature_K': temperature_K,
        }
        
        if base_length_mm is not None:
            element_kwargs['base_length_mm'] = base_length_mm
        if height_mm is not None:
            element_kwargs['height_mm'] = height_mm
        
        super().__init__(
            name=name,
            pitch_mm=pitch_mm,
            num_elements_x=num_elements_x,
            num_elements_y=num_elements_y,
            element_type="prism",
            element_kwargs=element_kwargs,
            spacing_type=spacing_type,
            material_name=material_name,
            temperature_K=temperature_K,
            wavelength_nm=wavelength_nm,
            display=display,
            display_dir=display_dir,
        )
    
    def _create_optic(self, i: int, j: int, position_mm: Tuple[float, float]) -> OpticalElement:
        """FR: Crée un microprisme."""
        if not OPTIQUES_AVAILABLE:
            raise ImportError("Optiques module is required for _create_optic().")
        
        kwargs = self.element_kwargs.copy()
        kwargs['position_mm'] = (*position_mm, 0.0)
        kwargs['wavelength_nm'] = self.wavelength_nm
        kwargs['temperature_K'] = self.temperature_K
        
        # Si base_length et height ne sont pas spécifiés, utiliser element_diameter
        if 'base_length_mm' not in kwargs:
            kwargs['base_length_mm'] = self.element_diameter_mm
        if 'height_mm' not in kwargs:
            kwargs['height_mm'] = self.element_diameter_mm
        
        return Prism(name=f"Microprisme ({i},{j})", **kwargs)
    
    def get_deviation_map(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        """
        FR: Calcule la carte de déviation pour la matrice de microprismes.
            Retourne l'angle de déviation à chaque point.

        EN: Calculates the deviation map for the microprism array.
            Returns the deviation angle at each point.

        Args:
            grid_x_mm (np.ndarray): Grille de positions en x (mm).
            grid_y_mm (np.ndarray): Grille de positions en y (mm).

        Returns:
            np.ndarray: Carte de déviation en degrés.
        """
        deviation_map = np.zeros_like(grid_x_mm)
        
        for element in self.elements:
            x_shifted = grid_x_mm - element.position_mm[0]
            y_shifted = grid_y_mm - element.position_mm[1]
            
            # Vérifier si le point est dans l'aperture du prisme
            mask = element.optic.get_aperture_mask(x_shifted, y_shifted)
            
            # Calculer la déviation pour ce prisme
            deviation = element.optic.get_deviation_angle()
            
            # Ajouter la déviation (pondérée par le masque)
            deviation_map += deviation * mask
        
        return deviation_map


class MicrogratingArray(Microstructure):
    """
    FR: Matrice de microréseaux de diffraction.
        Permet de créer des matrices de microréseaux pour la diffraction.

    EN: Micro diffraction grating array.
        Allows creating arrays of micro diffraction gratings.
    """
    
    def __init__(
        self,
        name: str,
        pitch_mm: float,
        num_elements_x: int,
        num_elements_y: int,
        lines_per_mm: float = 100.0,
        orientation_deg: float = 0.0,
        grating_type: GratingType = GratingType.TRANSMISSION,
        material_name: str = "Fused_Silica",
        spacing_type: SpacingType = SpacingType.EDGE_TO_EDGE,
        temperature_K: float = STANDARD_TEMPERATURE_K if MATERIAL_BEHAVIOUR_AVAILABLE else 293.15,
        wavelength_nm: float = 633.0,
        display: bool = False,
        display_dir: str = "output",
    ):
        """
        FR: Initialise une matrice de microréseaux de diffraction.

        EN: Initializes a micro diffraction grating array.

        Args:
            name (str): Nom de la matrice.
            pitch_mm (float): Distance de bord à bord entre les réseaux en mm.
            num_elements_x (int): Nombre de réseaux en x.
            num_elements_y (int): Nombre de réseaux en y.
            lines_per_mm (float): Nombre de lignes par mm pour les réseaux.
            orientation_deg (float): Orientation des réseaux en degrés.
            grating_type (GratingType): Type de réseau (TRANSMISSION ou REFLECTION).
            material_name (str): Nom du matériau.
            spacing_type (SpacingType): Type d'espacement.
            temperature_K (float): Température en Kelvin.
            wavelength_nm (float): Longueur d'onde en nm.
            display (bool): Afficher automatiquement les cartes.
            display_dir (str): Répertoire pour sauvegarder les images.
        """
        # Préparer les arguments pour les réseaux
        element_kwargs = {
            'lines_per_mm': lines_per_mm,
            'orientation_deg': orientation_deg,
            'grating_type': grating_type,
            'material_name': material_name,
            'wavelength_nm': wavelength_nm,
            'temperature_K': temperature_K,
        }
        
        super().__init__(
            name=name,
            pitch_mm=pitch_mm,
            num_elements_x=num_elements_x,
            num_elements_y=num_elements_y,
            element_type="grating",
            element_kwargs=element_kwargs,
            spacing_type=spacing_type,
            material_name=material_name,
            temperature_K=temperature_K,
            wavelength_nm=wavelength_nm,
            display=display,
            display_dir=display_dir,
        )
    
    def _create_optic(self, i: int, j: int, position_mm: Tuple[float, float]) -> OpticalElement:
        """FR: Crée un microréseau de diffraction."""
        if not OPTIQUES_AVAILABLE:
            raise ImportError("Optiques module is required for _create_optic().")
        
        kwargs = self.element_kwargs.copy()
        kwargs['diameter_mm'] = self.element_diameter_mm
        kwargs['position_mm'] = (*position_mm, 0.0)
        kwargs['wavelength_nm'] = self.wavelength_nm
        kwargs['temperature_K'] = self.temperature_K
        
        return DiffractionGrating(name=f"Microréseau ({i},{j})", **kwargs)
    
    def get_diffraction_orders_map(
        self,
        order: int,
        grid_x_mm: np.ndarray,
        grid_y_mm: np.ndarray,
    ) -> np.ndarray:
        """
        FR: Calcule la carte d'efficacité de diffraction pour un ordre donné.

        EN: Calculates the diffraction efficiency map for a given order.

        Args:
            order (int): Ordre de diffraction.
            grid_x_mm (np.ndarray): Grille de positions en x (mm).
            grid_y_mm (np.ndarray): Grille de positions en y (mm).

        Returns:
            np.ndarray: Carte d'efficacité pour l'ordre donné.
        """
        efficiency_map = np.zeros_like(grid_x_mm)
        
        for element in self.elements:
            x_shifted = grid_x_mm - element.position_mm[0]
            y_shifted = grid_y_mm - element.position_mm[1]
            
            # Vérifier si le point est dans l'aperture du réseau
            mask = element.optic.get_aperture_mask(x_shifted, y_shifted)
            
            # Calculer l'efficacité pour cet ordre
            efficiency = element.optic.get_diffraction_efficiency(order)
            
            # Ajouter l'efficacité (pondérée par le masque)
            efficiency_map += efficiency * mask
        
        return efficiency_map


# =============================================================================
# 5. FONCTIONS UTILITAIRES / UTILITY FUNCTIONS
# =============================================================================

def create_microstructure(
    microstructure_type: str,
    name: str = "Microstructure",
    pitch_mm: float = 0.5,
    num_elements_x: int = 11,
    num_elements_y: int = 11,
    **kwargs
) -> Microstructure:
    """
    FR: Fabrique une microstructure de type spécifié.

    EN: Factory function to create a microstructure of the specified type.

    Args:
        microstructure_type (str): Type de microstructure ('microlens_array', 'microhole_array', 'microprism_array', 'micrograting_array', 'custom').
        name (str): Nom de la microstructure.
        pitch_mm (float): Distance de bord à bord entre les éléments en mm.
        num_elements_x (int): Nombre d'éléments en x.
        num_elements_y (int): Nombre d'éléments en y.
        **kwargs: Arguments spécifiques au type de microstructure.

    Returns:
        Microstructure: La microstructure créée.

    Raises:
        ValueError: Si le type de microstructure est inconnu.
    """
    microstructure_classes = {
        "microlens_array": MicrolensArray,
        "microhole_array": MicroholeArray,
        "microprism_array": MicroprismArray,
        "micrograting_array": MicrogratingArray,
        "custom": Microstructure,
    }
    
    if microstructure_type not in microstructure_classes:
        raise ValueError(f"Type de microstructure inconnu: {microstructure_type}")
    
    return microstructure_classes[microstructure_type](
        name=name,
        pitch_mm=pitch_mm,
        num_elements_x=num_elements_x,
        num_elements_y=num_elements_y,
        **kwargs
    )


# =============================================================================
# 6. TESTS UNITAIRES / UNIT TESTS
# =============================================================================

class TestMicrostructure:
    """FR: Tests unitaires pour Microstructure.py."""
    
    def test_microlens_array_creation(self):
        """Test la création d'une matrice de microlentilles."""
        if not OPTIQUES_AVAILABLE:
            return
        
        array = MicrolensArray(
            name="Test Microlens Array",
            pitch_mm=0.5,
            num_elements_x=3,
            num_elements_y=3,
            focal_length_mm=10.0,
        )
        
        assert len(array.elements) == 9
        assert array.get_element(0, 0) is not None
        assert array.get_element(2, 2) is not None
        
        # Vérifier l'espacement
        spacing = array.get_element_spacing()
        assert spacing["spacing_type"] == "edge_to_edge"
        assert abs(spacing["edge_to_edge_mm"] - 0.5) < 1e-6
    
    def test_microhole_array_creation(self):
        """Test la création d'une matrice de microtrous."""
        if not OPTIQUES_AVAILABLE:
            return
        
        array = MicroholeArray(
            name="Test Microhole Array",
            pitch_mm=0.5,
            num_elements_x=3,
            num_elements_y=3,
            hole_diameter_mm=0.4,
        )
        
        assert len(array.elements) == 9
        for element in array.elements:
            assert isinstance(element.optic, ApertureStop)
    
    def test_microprism_array_creation(self):
        """Test la création d'une matrice de microprismes."""
        if not OPTIQUES_AVAILABLE:
            return
        
        array = MicroprismArray(
            name="Test Microprism Array",
            pitch_mm=0.5,
            num_elements_x=3,
            num_elements_y=3,
            apex_angle_deg=10.0,
        )
        
        assert len(array.elements) == 9
        for element in array.elements:
            assert isinstance(element.optic, Prism)
    
    def test_micrograting_array_creation(self):
        """Test la création d'une matrice de microréseaux."""
        if not OPTIQUES_AVAILABLE:
            return
        
        array = MicrogratingArray(
            name="Test Micrograting Array",
            pitch_mm=0.5,
            num_elements_x=3,
            num_elements_y=3,
            lines_per_mm=100.0,
        )
        
        assert len(array.elements) == 9
        for element in array.elements:
            assert isinstance(element.optic, DiffractionGrating)
    
    def test_global_wfe(self):
        """Test l'application d'une WFE globale."""
        if not OPTIQUES_AVAILABLE:
            return
        
        array = MicrolensArray(
            name="Test Microlens Array",
            pitch_mm=0.5,
            num_elements_x=3,
            num_elements_y=3,
            focal_length_mm=10.0,
        )
        
        # Appliquer une WFE globale
        wfe = WaveFrontError(
            surface_roughness_nm=5.0,
            parallelism_arcsec=2.0,
        )
        array.apply_global_wfe(wfe)
        
        assert array.global_wfe is not None
        assert array.global_wfe.surface_roughness_nm == 5.0
    
    def test_thermal_expansion(self):
        """Test la dilatation thermique."""
        if not OPTIQUES_AVAILABLE or not MATERIAL_BEHAVIOUR_AVAILABLE:
            return
        
        array = MicrolensArray(
            name="Test Microlens Array",
            pitch_mm=0.5,
            num_elements_x=3,
            num_elements_y=3,
            focal_length_mm=10.0,
            temperature_K=293.15,  # 20°C
        )
        
        initial_pitch = array.pitch_mm
        initial_positions = [e.position_mm for e in array.elements]
        
        # Changer la température à 100°C
        array.update_temperature(373.15)
        
        # Vérifier que le pitch a changé
        assert array.pitch_mm > initial_pitch
        
        # Vérifier que les positions ont changé
        new_positions = [e.position_mm for e in array.elements]
        for old_pos, new_pos in zip(initial_positions, new_positions):
            if old_pos != (0.0, 0.0):  # Éviter l'élément central
                assert old_pos != new_pos
    
    def test_phase_map(self):
        """Test le calcul de la carte de phase."""
        if not OPTIQUES_AVAILABLE or not MATH_TOOLS_AVAILABLE:
            return
        
        array = MicrolensArray(
            name="Test Microlens Array",
            pitch_mm=1.0,
            num_elements_x=3,
            num_elements_y=3,
            focal_length_mm=10.0,
        )
        
        grid_x, grid_y = create_grid(5.0, 256)
        phase_map = array.get_phase_map(grid_x, grid_y)
        
        assert phase_map.shape == grid_x.shape
        assert not np.all(phase_map == 0)
    
    def test_transmission_map(self):
        """Test le calcul de la carte de transmission."""
        if not OPTIQUES_AVAILABLE or not MATH_TOOLS_AVAILABLE:
            return
        
        array = MicroholeArray(
            name="Test Microhole Array",
            pitch_mm=1.0,
            num_elements_x=3,
            num_elements_y=3,
            hole_diameter_mm=0.8,
        )
        
        grid_x, grid_y = create_grid(5.0, 256)
        transmission_map = array.get_transmission_map(grid_x, grid_y)
        
        assert transmission_map.shape == grid_x.shape
        assert np.any(transmission_map > 0)
    
    def test_apply_to_beam(self):
        """Test l'application à un faisceau."""
        if not OPTIQUES_AVAILABLE or not BEAM_AVAILABLE or not MATH_TOOLS_AVAILABLE:
            return
        
        array = MicrolensArray(
            name="Test Microlens Array",
            pitch_mm=1.0,
            num_elements_x=3,
            num_elements_y=3,
            focal_length_mm=10.0,
        )
        
        beam = Beam(wavelength_nm=633.0, diameter_mm=5.0, num_points=256)
        beam.electric_field = beam.generate_electric_field(method="gaussian")
        
        beam_after = array.apply_to_beam(beam)
        
        assert beam_after is not beam
        assert beam_after.phase is not None
    
    def test_element_spacing(self):
        """Test les informations d'espacement."""
        if not OPTIQUES_AVAILABLE:
            return
        
        # Test edge_to_edge
        array_edge = MicrolensArray(
            name="Test Edge",
            pitch_mm=0.5,
            num_elements_x=3,
            num_elements_y=3,
            spacing_type=SpacingType.EDGE_TO_EDGE,
        )
        spacing_edge = array_edge.get_element_spacing()
        assert spacing_edge["spacing_type"] == "edge_to_edge"
        assert abs(spacing_edge["edge_to_edge_mm"] - 0.5) < 1e-6
        
        # Test center_to_center
        array_center = MicrolensArray(
            name="Test Center",
            pitch_mm=1.0,
            num_elements_x=3,
            num_elements_y=3,
            spacing_type=SpacingType.CENTER_TO_CENTER,
        )
        spacing_center = array_center.get_element_spacing()
        assert spacing_center["spacing_type"] == "center_to_center"
        assert abs(spacing_center["center_to_center_mm"] - 1.0) < 1e-6
    
    def test_create_microstructure_factory(self):
        """Test la fabrique de microstructures."""
        if not OPTIQUES_AVAILABLE:
            return
        
        # Créer une matrice de microlentilles
        array = create_microstructure(
            microstructure_type="microlens_array",
            name="Test Factory",
            pitch_mm=0.5,
            num_elements_x=3,
            num_elements_y=3,
            focal_length_mm=10.0,
        )
        assert isinstance(array, MicrolensArray)
        assert len(array.elements) == 9
        
        # Créer une matrice de microtrous
        array = create_microstructure(
            microstructure_type="microhole_array",
            name="Test Factory",
            pitch_mm=0.5,
            num_elements_x=3,
            num_elements_y=3,
        )
        assert isinstance(array, MicroholeArray)
        assert len(array.elements) == 9


if __name__ == "__main__":
    import unittest
    unittest.main()

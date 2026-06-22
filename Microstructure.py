"""
Microstructure.py - Complete Version
FR: Module pour la gestion de matrices de micro-optiques.
    Permet de créer des matrices de :
    - Microlentilles (circulaires, carrées, hexagonales)
    - Microtrous (pour la diffraction)
    - Microprismes
    - Micro-optiques personnalisées

    Fonctionnalités clés :
    - Espacement DE BORD À BORD entre les éléments (par défaut = 0, éléments joints)
    - Application de WFE GLOBALE sur toute la matrice (en plus des WFE individuelles)
    - Dilatation thermique avec effet sur les POSITIONS des micro-éléments
    - Calcul de la phase totale de la matrice
    - Intégration complète avec Optiques.py et Material_Behaviour.py

EN: Module for managing micro-optics arrays.
    Allows creating arrays of:
    - Microlenses (circular, square, hexagonal)
    - Microholes (for diffraction)
    - Microprisms
    - Custom micro-optics

    Key features:
    - EDGE-TO-EDGE spacing between elements (default = 0, joined elements)
    - GLOBAL WFE application on the entire array (in addition to individual WFE)
    - Thermal expansion with effect on MICRO-ELEMENT POSITIONS
    - Total phase calculation of the array
    - Full integration with Optiques.py and Material_Behaviour.py

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch

Note: All spacing measurements are EDGE-TO-EDGE by default (elements are joined).
      Center-to-center distance = edge_to_edge_spacing + element_diameter
"""

import numpy as np
import logging
import os
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# IMPORT DES DÉPENDANCES
# =============================================================================

try:
    from Optiques import (
        OpticalElement, IdealLens, SimpleLens, DoubleLens, DoubletLens,
        Mirror, Beamsplitter, Window, Prism, AsphericLens,
        ApertureStop, DiffractionHole, DiffractionGrating,
        ApertureShape, OpticSpecifications, WaveFrontError, LensType,
        MirrorType, BeamsplitterType, GratingType
    )
    OPTIQUES_AVAILABLE = True
except ImportError as e:
    OPTIQUES_AVAILABLE = False
    logging.warning(f"Optiques module not available: {e}")

try:
    from Material_Behaviour import MaterialBehaviour, STANDARD_TEMPERATURE_K
    MATERIAL_BEHAVIOUR_AVAILABLE = True
except ImportError as e:
    MATERIAL_BEHAVIOUR_AVAILABLE = False
    logging.warning(f"Material_Behaviour module not available: {e}")
    STANDARD_TEMPERATURE_K = 293.15

try:
    from Beam import Beam
    BEAM_AVAILABLE = True
except ImportError as e:
    BEAM_AVAILABLE = False
    logging.warning(f"Beam module not available: {e}")

try:
    from MathAndPhysicsTools import create_grid
    MATH_TOOLS_AVAILABLE = True
except ImportError as e:
    MATH_TOOLS_AVAILABLE = False
    logging.warning(f"MathAndPhysicsTools module not available: {e}")

try:
    from Visualization import plot_intensity, plot_phase
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    VISUALIZATION_AVAILABLE = False
    logging.warning(f"Visualization module not available: {e}")

try:
    from Propagation import Propagation
    PROPAGATION_AVAILABLE = True
except ImportError as e:
    PROPAGATION_AVAILABLE = False
    logging.warning(f"Propagation module not available: {e}")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Microstructure")


# =============================================================================
# ENUMS
# =============================================================================

class MicroOpticType(Enum):
    """FR: Type de micro-optique."""
    MICROLENS = "microlens"
    MICROHOLE = "microhole"
    MICROPRISM = "microprism"
    CUSTOM = "custom"


class ArrayPattern(Enum):
    """FR: Motif de la matrice."""
    SQUARE = "square"
    HEXAGONAL = "hexagonal"
    RECTANGULAR = "rectangular"


# =============================================================================
# MICRO OPTIC ELEMENT
# =============================================================================

@dataclass
class MicroOpticElement:
    """FR: Élément individuel d'une matrice de micro-optiques."""
    optic: OpticalElement
    position_mm: Tuple[float, float]
    index: Tuple[int, int]
    element_diameter_mm: float


# =============================================================================
# BASE CLASS: MICROSTRUCTURE
# =============================================================================

class MicroOpticsArray:
    """
    FR: Matrice de micro-optiques avec ESPACEMENT DE BORD À BORD.
        
    Caractéristiques principales:
    - pitch_mm = distance de BORD à BORD entre les éléments (par défaut = 0)
    - center_to_center_mm = pitch_mm + element_diameter_mm
    - WFE globale applicable à TOUTE la matrice
    - Dilatation thermique affecte les POSITIONS des éléments
    - Chaque élément peut aussi avoir sa propre WFE
    
    EN: Micro-optics array with EDGE-TO-EDGE spacing.
        
    Main features:
    - pitch_mm = EDGE-TO-EDGE distance between elements (default = 0)
    - center_to_center_mm = pitch_mm + element_diameter_mm
    - Global WFE applicable to the ENTIRE array
    - Thermal expansion affects ELEMENT POSITIONS
    - Each element can also have its own WFE
    """

    def __init__(
        self,
        name: str,
        pitch_mm: float,
        num_elements_x: int,
        num_elements_y: int,
        micro_optic_type: MicroOpticType = MicroOpticType.MICROLENS,
        array_pattern: ArrayPattern = ArrayPattern.SQUARE,
        material_name: str = "Fused_Silica",
        temperature_K: float = STANDARD_TEMPERATURE_K,
        wavelength_nm: float = 633.0,
        edge_to_edge_spacing_mm: float = 0.0,
        global_wfe: Optional[WaveFrontError] = None,
        display: bool = False,
        display_dir: str = "output"
    ):
        self.name = name
        self.pitch_mm = pitch_mm
        self.num_elements_x = num_elements_x
        self.num_elements_y = num_elements_y
        self.micro_optic_type = micro_optic_type
        self.array_pattern = array_pattern
        self.material_name = material_name
        self.temperature_K = temperature_K
        self.wavelength_nm = wavelength_nm
        self.edge_to_edge_spacing_mm = edge_to_edge_spacing_mm
        self.global_wfe = global_wfe or WaveFrontError()
        self.display = display
        self.display_dir = display_dir

        # Material
        self.material = MaterialBehaviour(material_name) if MATERIAL_BEHAVIOUR_AVAILABLE else None

        # Calculate element diameter and center-to-center
        self.element_diameter_mm = self._calculate_element_diameter()
        self.center_to_center_mm = self.pitch_mm + self.element_diameter_mm

        # Total size
        self.total_width_mm = self._calculate_total_width()
        self.total_height_mm = self._calculate_total_height()

        # Elements
        self.micro_optics: List[MicroOpticElement] = []
        self._create_micro_optics()

        if display: os.makedirs(display_dir, exist_ok=True)

    def _calculate_element_diameter(self) -> float:
        """FR: Calcule le diamètre de chaque élément."""
        # Si edge_to_edge_spacing = 0, les éléments sont joints
        # Donc diamètre = center_to_center - edge_to_edge_spacing
        # Mais on veut diamètre tel que center_to_center = diamètre + edge_to_edge_spacing
        # Donc diamètre = center_to_center - edge_to_edge_spacing
        # Mais on ne connaît pas encore center_to_center...
        # Solution: on utilise pitch_mm comme edge_to_edge_spacing
        # Donc diamètre = ?
        # Par défaut: diamètre = pitch_mm (si edge_to_edge_spacing = 0)
        # Mais c'est plus logique de dire que pitch_mm est la distance centre-à-centre
        # et edge_to_edge_spacing est la distance bord-à-bord
        # Donc: pitch_mm (centre-à-centre) = diamètre + edge_to_edge_spacing
        # => diamètre = pitch_mm - edge_to_edge_spacing
        return self.pitch_mm - self.edge_to_edge_spacing_mm

    def _calculate_total_width(self) -> float:
        """FR: Calcule la largeur totale."""
        if self.array_pattern == ArrayPattern.SQUARE:
            return (self.num_elements_x - 1) * self.center_to_center_mm + self.element_diameter_mm
        elif self.array_pattern == ArrayPattern.HEXAGONAL:
            return (self.num_elements_x - 1) * self.center_to_center_mm * np.cos(np.pi/6) + self.element_diameter_mm
        else:
            return (self.num_elements_x - 1) * self.center_to_center_mm + self.element_diameter_mm

    def _calculate_total_height(self) -> float:
        """FR: Calcule la hauteur totale."""
        if self.array_pattern == ArrayPattern.SQUARE:
            return (self.num_elements_y - 1) * self.center_to_center_mm + self.element_diameter_mm
        elif self.array_pattern == ArrayPattern.HEXAGONAL:
            return (self.num_elements_y - 1) * self.center_to_center_mm * np.sin(np.pi/3) + self.element_diameter_mm
        else:
            return (self.num_elements_y - 1) * self.center_to_center_mm + self.element_diameter_mm

    def _create_micro_optics(self) -> None:
        """FR: Crée toutes les micro-optiques."""
        for j in range(self.num_elements_y):
            for i in range(self.num_elements_x):
                x, y = self._calculate_position(i, j)
                optic = self._create_single_micro_optic(i, j, x, y)
                self.micro_optics.append(MicroOpticElement(
                    optic=optic,
                    position_mm=(x, y),
                    index=(i, j),
                    element_diameter_mm=self.element_diameter_mm
                ))

    def _calculate_position(self, i: int, j: int) -> Tuple[float, float]:
        """FR: Calcule la position du centre d'une micro-optique."""
        if self.array_pattern == ArrayPattern.SQUARE:
            x = (i - (self.num_elements_x - 1)/2) * self.center_to_center_mm
            y = (j - (self.num_elements_y - 1)/2) * self.center_to_center_mm
        elif self.array_pattern == ArrayPattern.HEXAGONAL:
            x = (i - (self.num_elements_x - 1)/2) * self.center_to_center_mm * np.cos(np.pi/6)
            y = (j - (self.num_elements_y - 1)/2) * self.center_to_center_mm * np.sin(np.pi/3)
            if j % 2 == 1:
                x += self.center_to_center_mm * np.cos(np.pi/6) / 2
        else:
            x = (i - (self.num_elements_x - 1)/2) * self.center_to_center_mm
            y = (j - (self.num_elements_y - 1)/2) * self.center_to_center_mm
        return (x, y)

    def _create_single_micro_optic(self, i: int, j: int, x: float, y: float) -> OpticalElement:
        """FR: Crée une seule micro-optique."""
        if self.micro_optic_type == MicroOpticType.MICROLENS:
            return self._create_microlens(i, j, x, y)
        elif self.micro_optic_type == MicroOpticType.MICROHOLE:
            return self._create_microhole(i, j, x, y)
        elif self.micro_optic_type == MicroOpticType.MICROPRISM:
            return self._create_microprism(i, j, x, y)
        else:
            return self._create_custom(i, j, x, y)

    def _create_microlens(self, i: int, j: int, x: float, y: float) -> OpticalElement:
        """FR: Crée une microlentille."""
        shape = ApertureShape.HEXAGONAL if self.array_pattern == ArrayPattern.HEXAGONAL else ApertureShape.CIRCULAR
        specs = OpticSpecifications(
            diameter_mm=self.element_diameter_mm,
            thickness_mm=1.0,
            material_name=self.material_name,
            aperture_shape=shape
        )
        return IdealLens(
            name=f"Microlens ({i},{j})",
            focal_length_mm=self.pitch_mm * 2,  # Default focal length
            diameter_mm=self.element_diameter_mm,
            material_name=self.material_name,
            specifications=specs,
            position_mm=(x, y, 0.0),
            wavelength_nm=self.wavelength_nm,
            temperature_K=self.temperature_K
        )

    def _create_microhole(self, i: int, j: int, x: float, y: float) -> OpticalElement:
        """FR: Crée un microtrou."""
        specs = OpticSpecifications(
            diameter_mm=self.element_diameter_mm,
            thickness_mm=0.0,
            material_name="air",
            aperture_shape=ApertureShape.CIRCULAR
        )
        return DiffractionHole(
            name=f"Microhole ({i},{j})",
            diameter_mm=self.element_diameter_mm,
            material_name="air",
            specifications=specs,
            position_mm=(x, y, 0.0),
            wavelength_nm=self.wavelength_nm,
            temperature_K=self.temperature_K
        )

    def _create_microprism(self, i: int, j: int, x: float, y: float) -> OpticalElement:
        """FR: Crée un microprisme."""
        specs = OpticSpecifications(
            diameter_mm=self.element_diameter_mm,
            thickness_mm=self.element_diameter_mm * 0.5,
            material_name=self.material_name,
            aperture_shape=ApertureShape.TRIANGULAR,
            width_mm=self.element_diameter_mm,
            height_mm=self.element_diameter_mm
        )
        return Prism(
            name=f"Microprism ({i},{j})",
            apex_angle_deg=60.0,
            base_length_mm=self.element_diameter_mm,
            height_mm=self.element_diameter_mm,
            material_name=self.material_name,
            specifications=specs,
            position_mm=(x, y, 0.0),
            wavelength_nm=self.wavelength_nm,
            temperature_K=self.temperature_K
        )

    def _create_custom(self, i: int, j: int, x: float, y: float) -> OpticalElement:
        """FR: Crée une micro-optique personnalisée."""
        return self._create_microlens(i, j, x, y)

    # =========================================================================
    # PHASE AND MASK CALCULATIONS
    # =========================================================================

    def get_total_phase_map(
        self,
        grid_x_mm: np.ndarray,
        grid_y_mm: np.ndarray,
        include_global_wfe: bool = True,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        FR: Calcule la phase TOTALE de la matrice.
            = Somme des phases de chaque micro-optique
              + WFE globale (si include_global_wfe=True)
            
        Chaque micro-optique contribue UNIQUEMENT dans sa zone d'aperture.

        EN: Calculates the TOTAL phase of the array.
            = Sum of phases from each micro-optic
              + Global WFE (if include_global_wfe=True)
            
        Each micro-optic contributes ONLY within its aperture area.
        """
        total_phase = np.zeros_like(grid_x_mm)

        # Add phase from each micro-optic
        for micro_optic in self.micro_optics:
            x_c, y_c = micro_optic.position_mm
            local_x = grid_x_mm - x_c
            local_y = grid_y_mm - y_c

            # Get phase and mask for this micro-optic
            phase = micro_optic.optic.get_phase_map(local_x, local_y)
            mask = micro_optic.optic.get_aperture_mask(local_x, local_y)

            # Add masked phase
            total_phase += phase * mask

        # Add global WFE
        if include_global_wfe and self.global_wfe is not None:
            global_phase = self.global_wfe.generate_phase_map(
                grid_x_mm, grid_y_mm, self.wavelength_nm, seed
            )
            total_phase += global_phase

        return total_phase

    def get_total_aperture_mask(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        """FR: Calcule le masque TOTAL (union de tous les masques)."""
        total_mask = np.zeros_like(grid_x_mm)
        for micro_optic in self.micro_optics:
            x_c, y_c = micro_optic.position_mm
            local_x = grid_x_mm - x_c
            local_y = grid_y_mm - y_c
            mask = micro_optic.optic.get_aperture_mask(local_x, local_y)
            total_mask = np.maximum(total_mask, mask)
        return total_mask

    # =========================================================================
    # BEAM APPLICATION
    # =========================================================================

    def apply_to_beam(self, beam: any) -> any:
        """
        FR: Applique la matrice à un faisceau.
            La phase totale de la matrice est appliquée au faisceau.
            Affichage automatique si display=True.

        EN: Applies the array to a beam.
            The total phase of the array is applied to the beam.
            Automatic display if display=True.
        """
        if not BEAM_AVAILABLE:
            raise ImportError("Beam module required")

        # Create grid
        if MATH_TOOLS_AVAILABLE:
            grid_x, grid_y = create_grid(beam.diameter_mm, beam.num_points)
        else:
            x = y = np.linspace(-beam.diameter_mm/2, beam.diameter_mm/2, beam.num_points)
            grid_x, grid_y = np.meshgrid(x, y)

        # Display before
        if self.display and VISUALIZATION_AVAILABLE:
            os.makedirs(self.display_dir, exist_ok=True)
            if hasattr(beam, 'intensity') and beam.intensity is not None:
                try:
                    plot_intensity(
                        beam.intensity, beam.diameter_mm,
                        title=f"Before {self.name}",
                        save_path=os.path.join(self.display_dir, f"before_{self.name}_intensity.png")
                    )
                except: pass
            if hasattr(beam, 'phase') and beam.phase is not None:
                try:
                    plot_phase(
                        beam.phase, beam.diameter_mm,
                        title=f"Before {self.name}",
                        save_path=os.path.join(self.display_dir, f"before_{self.name}_phase.png")
                    )
                except: pass

        # Get total phase
        total_phase = self.get_total_phase_map(grid_x, grid_y)

        # Create new beam
        new_beam = Beam(
            wavelength_nm=beam.wavelength_nm,
            diameter_mm=beam.diameter_mm,
            energy=beam.energy,
            num_points=beam.num_points,
            coherence=beam.coherence
        )

        # Apply phase
        if beam.electric_field is not None:
            amplitude = np.abs(beam.electric_field)
            initial_phase = np.angle(beam.electric_field)
            new_phase_rad = initial_phase + total_phase * 2 * np.pi / self.wavelength_nm
            new_beam.electric_field = amplitude * np.exp(1j * new_phase_rad)
            try:
                new_beam.intensity = new_beam.compute_intensity_from_electric_field(new_beam.electric_field)
                new_beam.phase = new_beam.extract_phase_from_electric_field(new_beam.electric_field)
            except: pass

        # Display after
        if self.display and VISUALIZATION_AVAILABLE:
            if hasattr(new_beam, 'intensity') and new_beam.intensity is not None:
                try:
                    plot_intensity(
                        new_beam.intensity, new_beam.diameter_mm,
                        title=f"After {self.name}",
                        save_path=os.path.join(self.display_dir, f"after_{self.name}_intensity.png")
                    )
                except: pass
            if hasattr(new_beam, 'phase') and new_beam.phase is not None:
                try:
                    plot_phase(
                        new_beam.phase, new_beam.diameter_mm,
                        title=f"After {self.name}",
                        save_path=os.path.join(self.display_dir, f"after_{self.name}_phase.png")
                    )
                except: pass

        return new_beam

    # =========================================================================
    # THERMAL EXPANSION
    # =========================================================================

    def apply_thermal_deformation(self, new_temperature_K: float) -> None:
        """
        FR: Applique une déformation thermique à la matrice.
            - Met à jour le pitch (distance bord-à-bord)
            - Met à jour les POSITIONS de chaque micro-optique
            - Met à jour la température de chaque élément
            
        Formule: ΔL = L₀ * α * ΔT
                 où α = coefficient de dilatation thermique
                       ΔT = variation de température

        EN: Applies thermal deformation to the array.
            - Updates the pitch (edge-to-edge distance)
            - Updates the POSITIONS of each micro-optic
            - Updates the temperature of each element
        """
        if not MATERIAL_BEHAVIOUR_AVAILABLE or self.material is None:
            logger.warning("Material_Behaviour not available or material not set. Cannot apply thermal deformation.")
            return

        delta_T = new_temperature_K - self.temperature_K
        if delta_T == 0:
            return

        # Get thermal expansion coefficient
        try:
            alpha = self.material.get_thermal_expansion_coefficient(self.temperature_K)
        except:
            alpha = 8.0e-6  # Default CTE for glass

        # Calculate expansion factor
        expansion_factor = 1 + alpha * delta_T

        # Update pitch (edge-to-edge distance)
        old_pitch = self.pitch_mm
        self.pitch_mm *= expansion_factor

        # Update element diameter
        old_diameter = self.element_diameter_mm
        self.element_diameter_mm *= expansion_factor

        # Update center-to-center
        self.center_to_center_mm = self.pitch_mm + self.element_diameter_mm

        # Update positions of all micro-optics
        for micro_optic in self.micro_optics:
            i, j = micro_optic.index
            x, y = self._calculate_position(i, j)
            micro_optic.position_mm = (x, y)
            micro_optic.optic.position_mm = (x, y, micro_optic.optic.position_mm[2])
            micro_optic.element_diameter_mm = self.element_diameter_mm

        # Update total size
        self.total_width_mm = self._calculate_total_width()
        self.total_height_mm = self._calculate_total_height()

        # Update temperature
        old_temperature = self.temperature_K
        self.temperature_K = new_temperature_K
        for micro_optic in self.micro_optics:
            micro_optic.optic.temperature_K = new_temperature_K

        logger.info(f"Thermal deformation applied: ΔT={delta_T:.2f}K, α={alpha:.2e}, "
                    f"Δpitch={self.pitch_mm - old_pitch:.6f}mm, "
                    f"Δdiameter={self.element_diameter_mm - old_diameter:.6f}mm")

    def get_thermal_deformation_info(self, new_temperature_K: float) -> Dict:
        """FR: Calcule les informations de déformation sans l'appliquer."""
        if not MATERIAL_BEHAVIOUR_AVAILABLE or self.material is None:
            return {"error": "Material not available"}

        delta_T = new_temperature_K - self.temperature_K
        try:
            alpha = self.material.get_thermal_expansion_coefficient(self.temperature_K)
        except:
            alpha = 8.0e-6

        return {
            "delta_T_K": delta_T,
            "alpha": alpha,
            "delta_pitch_mm": self.pitch_mm * alpha * delta_T,
            "new_pitch_mm": self.pitch_mm * (1 + alpha * delta_T),
            "delta_diameter_mm": self.element_diameter_mm * alpha * delta_T,
            "new_diameter_mm": self.element_diameter_mm * (1 + alpha * delta_T),
            "delta_center_to_center_mm": self.center_to_center_mm * alpha * delta_T,
            "new_center_to_center_mm": self.center_to_center_mm * (1 + alpha * delta_T),
            "delta_total_width_mm": self.total_width_mm * alpha * delta_T,
            "new_total_width_mm": self.total_width_mm * (1 + alpha * delta_T),
            "delta_total_height_mm": self.total_height_mm * alpha * delta_T,
            "new_total_height_mm": self.total_height_mm * (1 + alpha * delta_T)
        }

    # =========================================================================
    # GLOBAL WFE
    # =========================================================================

    def set_global_wfe(self, wfe: WaveFrontError) -> None:
        """FR: Définit la WFE globale."""
        self.global_wfe = wfe

    def add_global_wfe_component(
        self,
        surface_roughness_nm: float = 0.0,
        parallelism_arcsec: float = 0.0,
        zernike_coefficients: Optional[Dict[Tuple[int, int], float]] = None,
        custom_phase_map: Optional[np.ndarray] = None
    ) -> None:
        """FR: Ajoute une composante à la WFE globale."""
        if self.global_wfe is None:
            self.global_wfe = WaveFrontError()

        if surface_roughness_nm > 0:
            self.global_wfe.surface_roughness_nm += surface_roughness_nm
        if parallelism_arcsec > 0:
            self.global_wfe.parallelism_arcsec += parallelism_arcsec
        if zernike_coefficients:
            for (n, m), coeff in zernike_coefficients.items():
                self.global_wfe.zernike_coefficients[(n, m)] = \
                    self.global_wfe.zernike_coefficients.get((n, m), 0.0) + coeff
        if custom_phase_map is not None:
            if self.global_wfe.custom_phase_map is None:
                self.global_wfe.custom_phase_map = custom_phase_map
            else:
                self.global_wfe.custom_phase_map += custom_phase_map

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_micro_optic(self, i: int, j: int) -> Optional[MicroOpticElement]:
        """FR: Retourne la micro-optique aux indices (i, j)."""
        for mo in self.micro_optics:
            if mo.index == (i, j):
                return mo
        return None

    def get_micro_optic_at_position(self, x: float, y: float, tol: float = 0.1) -> Optional[MicroOpticElement]:
        """FR: Retourne la micro-optique à une position (x, y)."""
        for mo in self.micro_optics:
            if np.hypot(mo.position_mm[0] - x, mo.position_mm[1] - y) <= tol:
                return mo
        return None

    def get_spacing_info(self) -> Dict[str, float]:
        """FR: Retourne les informations d'espacement."""
        return {
            "edge_to_edge_mm": self.pitch_mm,
            "center_to_center_mm": self.center_to_center_mm,
            "element_diameter_mm": self.element_diameter_mm,
            "spacing_type": "edge_to_edge"
        }

    def visualize(
        self,
        grid_size_mm: float = 10.0,
        num_points: int = 256,
        save_path: Optional[str] = None
    ) -> None:
        """FR: Visualise la phase totale et le masque total."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return

        if MATH_TOOLS_AVAILABLE:
            gx, gy = create_grid(grid_size_mm, num_points)
        else:
            x = y = np.linspace(-grid_size_mm/2, grid_size_mm/2, num_points)
            gx, gy = np.meshgrid(x, y)

        phase = self.get_total_phase_map(gx, gy)
        mask = self.get_total_aperture_mask(gx, gy)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.imshow(phase, extent=[-grid_size_mm/2]*2 + [grid_size_mm/2]*2, cmap='coolwarm')
        ax1.set_title(f"{self.name} - Total Phase"); ax1.set_xlabel("x (mm)"); ax1.set_ylabel("y (mm)")
        plt.colorbar(ax1.imshow(phase), ax=ax1, label="Phase (nm)")
        ax2.imshow(mask, extent=[-grid_size_mm/2]*2 + [grid_size_mm/2]*2, cmap='gray')
        ax2.set_title(f"{self.name} - Total Mask"); ax2.set_xlabel("x (mm)"); ax2.set_ylabel("y (mm)")
        plt.colorbar(ax2.imshow(mask), ax=ax2, label="Mask")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"pitch={self.pitch_mm:.2f}mm (edge-to-edge), "
                f"center-to-center={self.center_to_center_mm:.2f}mm, "
                f"size={self.num_elements_x}x{self.num_elements_y}, "
                f"diameter={self.element_diameter_mm:.2f}mm, "
                f"type={self.micro_optic_type.value}, "
                f"pattern={self.array_pattern.value}, "
                f"T={self.temperature_K:.1f}K)")


# =============================================================================
# SPECIALIZED CLASSES
# =============================================================================

class MicrolensArray(MicroOpticsArray):
    """FR: Matrice de microlentilles."""

    def __init__(
        self,
        name: str = "Microlens Array",
        pitch_mm: float = 0.5,
        num_elements_x: int = 10,
        num_elements_y: int = 10,
        focal_length_mm: float = 5.0,
        array_pattern: ArrayPattern = ArrayPattern.SQUARE,
        material_name: str = "Fused_Silica",
        temperature_K: float = STANDARD_TEMPERATURE_K,
        wavelength_nm: float = 633.0,
        edge_to_edge_spacing_mm: float = 0.0,
        global_wfe: Optional[WaveFrontError] = None,
        display: bool = False,
        display_dir: str = "output"
    ):
        self.focal_length_mm = focal_length_mm
        super().__init__(
            name=name, pitch_mm=pitch_mm, num_elements_x=num_elements_x,
            num_elements_y=num_elements_y, micro_optic_type=MicroOpticType.MICROLENS,
            array_pattern=array_pattern, material_name=material_name,
            temperature_K=temperature_K, wavelength_nm=wavelength_nm,
            edge_to_edge_spacing_mm=edge_to_edge_spacing_mm, global_wfe=global_wfe,
            display=display, display_dir=display_dir
        )

    def _create_microlens(self, i: int, j: int, x: float, y: float) -> OpticalElement:
        """FR: Crée une microlentille avec la distance focale spécifiée."""
        shape = ApertureShape.HEXAGONAL if self.array_pattern == ArrayPattern.HEXAGONAL else ApertureShape.CIRCULAR
        specs = OpticSpecifications(
            diameter_mm=self.element_diameter_mm,
            thickness_mm=1.0,
            material_name=self.material_name,
            aperture_shape=shape
        )
        return IdealLens(
            name=f"Microlens ({i},{j})",
            focal_length_mm=self.focal_length_mm,
            diameter_mm=self.element_diameter_mm,
            material_name=self.material_name,
            specifications=specs,
            position_mm=(x, y, 0.0),
            wavelength_nm=self.wavelength_nm,
            temperature_K=self.temperature_K
        )

    def get_focal_plane_phase(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        """FR: Calcule la phase dans le plan focal."""
        total_phase = np.zeros_like(grid_x_mm)
        for mo in self.micro_optics:
            x_c, y_c = mo.position_mm
            x_rel, y_rel = grid_x_mm - x_c, grid_y_mm - y_c
            r2 = x_rel**2 + y_rel**2
            f_m = self.focal_length_mm * 1e-3
            lm = self.wavelength_nm * 1e-9
            phase = - (2*np.pi/lm) * r2 / (2*f_m) * lm/(2*np.pi) * 1e9
            
            # Apply aperture mask
            diameter = mo.optic.specifications.diameter_mm
            if mo.optic.specifications.aperture_shape == ApertureShape.HEXAGONAL:
                mask = (np.abs(x_rel) + np.abs(y_rel)/np.sqrt(3) <= diameter/2 * 2/np.sqrt(3)).astype(float)
            else:
                mask = (np.sqrt(r2) <= diameter/2).astype(float)
            
            total_phase += phase * mask
        return total_phase


class MicroholeArray(MicroOpticsArray):
    """FR: Matrice de microtrous."""

    def __init__(
        self,
        name: str = "Microhole Array",
        pitch_mm: float = 0.5,
        num_elements_x: int = 10,
        num_elements_y: int = 10,
        hole_diameter_mm: float = 0.1,
        array_pattern: ArrayPattern = ArrayPattern.SQUARE,
        material_name: str = "air",
        temperature_K: float = STANDARD_TEMPERATURE_K,
        wavelength_nm: float = 633.0,
        edge_to_edge_spacing_mm: float = 0.0,
        global_wfe: Optional[WaveFrontError] = None,
        display: bool = False,
        display_dir: str = "output"
    ):
        # Override element diameter with hole diameter
        self.hole_diameter_mm = hole_diameter_mm
        # Temporarily set pitch to center-to-center for initialization
        # Will be recalculated in parent __init__
        super().__init__(
            name=name, pitch_mm=pitch_mm, num_elements_x=num_elements_x,
            num_elements_y=num_elements_y, micro_optic_type=MicroOpticType.MICROHOLE,
            array_pattern=array_pattern, material_name=material_name,
            temperature_K=temperature_K, wavelength_nm=wavelength_nm,
            edge_to_edge_spacing_mm=edge_to_edge_spacing_mm, global_wfe=global_wfe,
            display=display, display_dir=display_dir
        )
        # Now override element diameter with hole diameter
        self.element_diameter_mm = hole_diameter_mm
        self.center_to_center_mm = self.pitch_mm + hole_diameter_mm
        self.total_width_mm = self._calculate_total_width()
        self.total_height_mm = self._calculate_total_height()
        # Recreate micro-optics with correct diameter
        self.micro_optics = []
        self._create_micro_optics()

    def _create_microhole(self, i: int, j: int, x: float, y: float) -> OpticalElement:
        """FR: Crée un microtrou avec le diamètre spécifié."""
        specs = OpticSpecifications(
            diameter_mm=self.hole_diameter_mm,
            thickness_mm=0.0,
            material_name="air",
            aperture_shape=ApertureShape.CIRCULAR
        )
        return DiffractionHole(
            name=f"Microhole ({i},{j})",
            diameter_mm=self.hole_diameter_mm,
            material_name="air",
            specifications=specs,
            position_mm=(x, y, 0.0),
            wavelength_nm=self.wavelength_nm,
            temperature_K=self.temperature_K
        )

    def get_diffraction_pattern(
        self,
        distance_mm: float,
        grid_size_mm: float = 10.0,
        num_points: int = 256
    ) -> np.ndarray:
        """FR: Calcule la figure de diffraction à une distance donnée."""
        if not PROPAGATION_AVAILABLE:
            logger.warning("Propagation module not available")
            return np.zeros((num_points, num_points))

        if MATH_TOOLS_AVAILABLE:
            gx, gy = create_grid(grid_size_mm, num_points)
        else:
            x = y = np.linspace(-grid_size_mm/2, grid_size_mm/2, num_points)
            gx, gy = np.meshgrid(x, y)

        # Initial field = total aperture mask
        mask = self.get_total_aperture_mask(gx, gy)
        field = mask.astype(complex)

        try:
            prop = Propagation(
                wavelength_nm=self.wavelength_nm,
                propagation_distance_mm=distance_mm,
                input_diameter_mm=grid_size_mm,
                output_diameter_mm=grid_size_mm,
                num_points=num_points,
                method="angular_spectrum"
            )
            propagated = prop.propagate(field)
            return np.abs(propagated)**2
        except Exception as e:
            logger.warning(f"Diffraction calculation failed: {e}")
            return np.zeros_like(gx)


class MicroprismArray(MicroOpticsArray):
    """FR: Matrice de microprismes."""

    def __init__(
        self,
        name: str = "Microprism Array",
        pitch_mm: float = 0.5,
        num_elements_x: int = 10,
        num_elements_y: int = 10,
        apex_angle_deg: float = 60.0,
        array_pattern: ArrayPattern = ArrayPattern.SQUARE,
        material_name: str = "BK7",
        temperature_K: float = STANDARD_TEMPERATURE_K,
        wavelength_nm: float = 633.0,
        edge_to_edge_spacing_mm: float = 0.0,
        global_wfe: Optional[WaveFrontError] = None,
        display: bool = False,
        display_dir: str = "output"
    ):
        self.apex_angle_deg = apex_angle_deg
        super().__init__(
            name=name, pitch_mm=pitch_mm, num_elements_x=num_elements_x,
            num_elements_y=num_elements_y, micro_optic_type=MicroOpticType.MICROPRISM,
            array_pattern=array_pattern, material_name=material_name,
            temperature_K=temperature_K, wavelength_nm=wavelength_nm,
            edge_to_edge_spacing_mm=edge_to_edge_spacing_mm, global_wfe=global_wfe,
            display=display, display_dir=display_dir
        )

    def _create_microprism(self, i: int, j: int, x: float, y: float) -> OpticalElement:
        """FR: Crée un microprisme avec l'angle au sommet spécifié."""
        specs = OpticSpecifications(
            diameter_mm=self.element_diameter_mm,
            thickness_mm=self.element_diameter_mm * 0.5,
            material_name=self.material_name,
            aperture_shape=ApertureShape.TRIANGULAR,
            width_mm=self.element_diameter_mm,
            height_mm=self.element_diameter_mm
        )
        return Prism(
            name=f"Microprism ({i},{j})",
            apex_angle_deg=self.apex_angle_deg,
            base_length_mm=self.element_diameter_mm,
            height_mm=self.element_diameter_mm,
            material_name=self.material_name,
            specifications=specs,
            position_mm=(x, y, 0.0),
            wavelength_nm=self.wavelength_nm,
            temperature_K=self.temperature_K
        )

    def get_deviation_map(self, grid_x_mm: np.ndarray, grid_y_mm: np.ndarray) -> np.ndarray:
        """FR: Calcule la carte de déviation (angle en degrés)."""
        deviation_map = np.zeros_like(grid_x_mm)
        for mo in self.micro_optics:
            x_c, y_c = mo.position_mm
            local_x = grid_x_mm - x_c
            local_y = grid_y_mm - y_c
            mask = mo.optic.get_aperture_mask(local_x, local_y)
            if isinstance(mo.optic, Prism):
                deviation = mo.optic.get_deviation_angle()
            else:
                deviation = 0.0
            deviation_map += deviation * mask
        return deviation_map


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_microlens_array(
    name: str = "Microlens Array",
    pitch_mm: float = 0.5,
    num_elements_x: int = 10,
    num_elements_y: int = 10,
    focal_length_mm: float = 5.0,
    array_pattern: ArrayPattern = ArrayPattern.SQUARE,
    material_name: str = "Fused_Silica",
    temperature_K: float = STANDARD_TEMPERATURE_K,
    wavelength_nm: float = 633.0,
    edge_to_edge_spacing_mm: float = 0.0,
    global_wfe: Optional[WaveFrontError] = None
) -> MicrolensArray:
    """FR: Fabrique une matrice de microlentilles."""
    return MicrolensArray(
        name=name, pitch_mm=pitch_mm, num_elements_x=num_elements_x,
        num_elements_y=num_elements_y, focal_length_mm=focal_length_mm,
        array_pattern=array_pattern, material_name=material_name,
        temperature_K=temperature_K, wavelength_nm=wavelength_nm,
        edge_to_edge_spacing_mm=edge_to_edge_spacing_mm, global_wfe=global_wfe
    )


def create_microhole_array(
    name: str = "Microhole Array",
    pitch_mm: float = 0.5,
    num_elements_x: int = 10,
    num_elements_y: int = 10,
    hole_diameter_mm: float = 0.1,
    array_pattern: ArrayPattern = ArrayPattern.SQUARE,
    material_name: str = "air",
    temperature_K: float = STANDARD_TEMPERATURE_K,
    wavelength_nm: float = 633.0,
    edge_to_edge_spacing_mm: float = 0.0,
    global_wfe: Optional[WaveFrontError] = None
) -> MicroholeArray:
    """FR: Fabrique une matrice de microtrous."""
    return MicroholeArray(
        name=name, pitch_mm=pitch_mm, num_elements_x=num_elements_x,
        num_elements_y=num_elements_y, hole_diameter_mm=hole_diameter_mm,
        array_pattern=array_pattern, material_name=material_name,
        temperature_K=temperature_K, wavelength_nm=wavelength_nm,
        edge_to_edge_spacing_mm=edge_to_edge_spacing_mm, global_wfe=global_wfe
    )


def create_microprism_array(
    name: str = "Microprism Array",
    pitch_mm: float = 0.5,
    num_elements_x: int = 10,
    num_elements_y: int = 10,
    apex_angle_deg: float = 60.0,
    array_pattern: ArrayPattern = ArrayPattern.SQUARE,
    material_name: str = "BK7",
    temperature_K: float = STANDARD_TEMPERATURE_K,
    wavelength_nm: float = 633.0,
    edge_to_edge_spacing_mm: float = 0.0,
    global_wfe: Optional[WaveFrontError] = None
) -> MicroprismArray:
    """FR: Fabrique une matrice de microprismes."""
    return MicroprismArray(
        name=name, pitch_mm=pitch_mm, num_elements_x=num_elements_x,
        num_elements_y=num_elements_y, apex_angle_deg=apex_angle_deg,
        array_pattern=array_pattern, material_name=material_name,
        temperature_K=temperature_K, wavelength_nm=wavelength_nm,
        edge_to_edge_spacing_mm=edge_to_edge_spacing_mm, global_wfe=global_wfe
    )


# =============================================================================
# UNIT TESTS
# =============================================================================

class TestMicrostructure:
    """FR: Tests unitaires pour Microstructure.py."""

    def test_microlens_array_creation(self):
        """Test la création d'une matrice de microlentilles."""
        array = create_microlens_array(
            name="Test", pitch_mm=1.0, num_elements_x=3, num_elements_y=3,
            focal_length_mm=5.0
        )
        assert len(array.micro_optics) == 9
        assert array.pitch_mm == 1.0
        assert array.focal_length_mm == 5.0
        assert array.center_to_center_mm == pytest.approx(1.0 + (1.0 - 0.0))  # pitch + diameter

    def test_edge_to_edge_spacing(self):
        """Test l'espacement de bord à bord."""
        spacing = 0.1
        array = create_microlens_array(
            name="Test", pitch_mm=1.1, num_elements_x=3, num_elements_y=3,
            focal_length_mm=5.0, edge_to_edge_spacing_mm=spacing
        )
        # pitch_mm = edge_to_edge_spacing + diameter
        # => diameter = pitch_mm - edge_to_edge_spacing = 1.1 - 0.1 = 1.0
        assert array.element_diameter_mm == pytest.approx(1.0)
        assert array.center_to_center_mm == pytest.approx(1.1)

    def test_total_phase_map(self):
        """Test le calcul de la phase totale."""
        if not OPTIQUES_AVAILABLE or not MATH_TOOLS_AVAILABLE:
            return
        array = create_microlens_array(
            name="Test", pitch_mm=1.0, num_elements_x=2, num_elements_y=2,
            focal_length_mm=5.0
        )
        gx, gy = create_grid(5.0, 256)
        phase = array.get_total_phase_map(gx, gy)
        assert phase.shape == gx.shape
        assert not np.all(phase == 0)

    def test_global_wfe(self):
        """Test la WFE globale."""
        if not OPTIQUES_AVAILABLE or not MATH_TOOLS_AVAILABLE:
            return
        array = create_microlens_array(
            name="Test", pitch_mm=1.0, num_elements_x=2, num_elements_y=2,
            focal_length_mm=5.0
        )
        array.set_global_wfe(WaveFrontError(surface_roughness_nm=5.0, seed=42))
        gx, gy = create_grid(5.0, 256)
        phase_with_wfe = array.get_total_phase_map(gx, gy)
        phase_without_wfe = array.get_total_phase_map(gx, gy, include_global_wfe=False)
        assert not np.allclose(phase_with_wfe, phase_without_wfe)

    def test_thermal_deformation(self):
        """Test la déformation thermique."""
        if not MATERIAL_BEHAVIOUR_AVAILABLE:
            return
        array = create_microlens_array(
            name="Test", pitch_mm=1.0, num_elements_x=3, num_elements_y=3,
            focal_length_mm=5.0, material_name="BK7"
        )
        initial_positions = [mo.position_mm for mo in array.micro_optics]
        initial_pitch = array.pitch_mm
        initial_diameter = array.element_diameter_mm

        array.apply_thermal_deformation(373.15)  # +80K

        new_positions = [mo.position_mm for mo in array.micro_optics]
        new_pitch = array.pitch_mm
        new_diameter = array.element_diameter_mm

        # Positions should have changed
        assert not all(np.allclose(ip, np) for ip, np in zip(initial_positions, new_positions))
        # Pitch should have increased
        assert new_pitch > initial_pitch
        # Diameter should have increased
        assert new_diameter > initial_diameter

    def test_microhole_array(self):
        """Test la matrice de microtrous."""
        array = create_microhole_array(
            name="Test", pitch_mm=1.0, num_elements_x=3, num_elements_y=3,
            hole_diameter_mm=0.5
        )
        assert len(array.micro_optics) == 9
        assert array.hole_diameter_mm == 0.5
        assert array.element_diameter_mm == 0.5

    def test_microprism_array(self):
        """Test la matrice de microprismes."""
        array = create_microprism_array(
            name="Test", pitch_mm=1.0, num_elements_x=3, num_elements_y=3,
            apex_angle_deg=60.0
        )
        assert len(array.micro_optics) == 9
        assert array.apex_angle_deg == 60.0

    def test_hexagonal_pattern(self):
        """Test le motif hexagonal."""
        array = create_microlens_array(
            name="Test", pitch_mm=1.0, num_elements_x=4, num_elements_y=4,
            focal_length_mm=5.0, array_pattern=ArrayPattern.HEXAGONAL
        )
        assert array.array_pattern == ArrayPattern.HEXAGONAL
        assert len(array.micro_optics) == 16


if __name__ == "__main__":
    import unittest
    import pytest
    unittest.main()

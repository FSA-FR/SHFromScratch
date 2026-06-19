"""
Example 11: Microstructure Arrays - Complete Demonstration
FR: Exemple complet d'utilisation du module Microstructure.py.
    Démonstration de :
    - Création de matrices de microlentilles (MicrolensArray)
    - Création de matrices de microtrous (MicroholeArray)
    - Création de matrices de microprismes (MicroprismArray)
    - Création de matrices de microréseaux (MicrogratingArray)
    - Gestion des espacements (bord à bord par défaut, centre à centre)
    - Application d'une WFE globale à la matrice
    - Prise en compte de la dilatation thermique
    - Intégration avec Beam.py et Propagation.py
    - Visualisation des matrices

EN: Complete example of using Microstructure.py module.
    Demonstrates:
    - Creation of microlens arrays (MicrolensArray)
    - Creation of microhole arrays (MicroholeArray)
    - Creation of microprism arrays (MicroprismArray)
    - Creation of micro grating arrays (MicrogratingArray)
    - Spacing management (edge-to-edge default, center-to-center)
    - Global WFE application to the array
    - Thermal expansion consideration
    - Integration with Beam.py and Propagation.py
    - Array visualization

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import shutil

from Microstructure import (
    Microstructure, MicrolensArray, MicroholeArray, MicroprismArray, MicrogratingArray,
    MicroOpticElement, MicroOpticType, SpacingType, create_microstructure
)
from Optiques import (
    IdealLens, ApertureStop, Prism, DiffractionGrating, WaveFrontError, ApertureShape,
    OpticSpecifications
)
from Beam import Beam
from Propagation import Propagation
from Visualization import plot_intensity, plot_phase, plot_beam_map


def run_microstructure_example():
    """FR: Exécute l'exemple de microstructures."""
    print("\n" + "="*80)
    print("Example 11: Microstructure Arrays - Complete Demonstration")
    print("="*80)
    
    # Paramètres globaux
    wavelength_nm = 633.0
    output_dir = 'examples/output'
    os.makedirs(output_dir, exist_ok=True)
    
    # =========================================================================
    # 1. Création de matrices de microlentilles / Microlens Arrays
    # =========================================================================
    print("\n--- 1. Matrices de microlentilles ---")
    
    # 1.1 Matrice de microlentilles idéales (espacement bord à bord)
    print("\n1.1 Matrice de microlentilles idéales (3x3, pitch bord-à-bord = 1 mm)")
    microlens_array_edge = MicrolensArray(
        name="Microlentilles bord-à-bord",
        pitch_mm=1.0,  # Distance bord à bord
        num_elements_x=3,
        num_elements_y=3,
        focal_length_mm=10.0,
        lens_type="ideal",
        material_name="Fused_Silica",
        temperature_K=293.15,
        wavelength_nm=wavelength_nm,
    )
    
    print(f"  Nombre d'éléments: {len(microlens_array_edge.elements)}")
    print(f"  Espacement bord à bord: {microlens_array_edge.pitch_mm:.3f} mm")
    print(f"  Distance centre à centre: {microlens_array_edge.center_to_center_mm:.3f} mm")
    print(f"  Diamètre des éléments: {microlens_array_edge.element_diameter_mm:.3f} mm")
    
    spacing_info = microlens_array_edge.get_element_spacing()
    print(f"  Type d'espacement: {spacing_info['spacing_type']}")
    print(f"  Espace entre bords: {spacing_info['gap_mm']:.3f} mm")
    
    # 1.2 Matrice de microlentilles idéales (espacement centre à centre)
    print("\n1.2 Matrice de microlentilles idéales (3x3, pitch centre-à-centre = 1.5 mm)")
    microlens_array_center = MicrolensArray(
        name="Microlentilles centre-à-centre",
        pitch_mm=1.5,  # Distance centre à centre
        num_elements_x=3,
        num_elements_y=3,
        focal_length_mm=10.0,
        lens_type="ideal",
        material_name="Fused_Silica",
        spacing_type=SpacingType.CENTER_TO_CENTER,
        temperature_K=293.15,
        wavelength_nm=wavelength_nm,
    )
    
    print(f"  Distance centre à centre: {microlens_array_center.center_to_center_mm:.3f} mm")
    print(f"  Espacement bord à bord: {microlens_array_center.pitch_mm:.3f} mm")
    spacing_info = microlens_array_center.get_element_spacing()
    print(f"  Espace entre bords: {spacing_info['gap_mm']:.3f} mm")
    
    # 1.3 Matrice de microlentilles asphériques
    print("\n1.3 Matrice de microlentilles asphériques (5x5)")
    microlens_array_aspheric = MicrolensArray(
        name="Microlentilles asphériques",
        pitch_mm=0.8,
        num_elements_x=5,
        num_elements_y=5,
        focal_length_mm=8.0,
        lens_type="aspheric",
        material_name="BK7",
        temperature_K=293.15,
        wavelength_nm=wavelength_nm,
    )
    
    print(f"  Nombre d'éléments: {len(microlens_array_aspheric.elements)}")
    
    # 1.4 Matrice de doublets (simplifiée)
    print("\n1.4 Matrice de doublets (2x2)")
    microlens_array_doublet = MicrolensArray(
        name="Microlentilles doublets",
        pitch_mm=2.0,
        num_elements_x=2,
        num_elements_y=2,
        focal_length_mm=20.0,
        lens_type="doublet",
        material_name="Fused_Silica",
        temperature_K=293.15,
        wavelength_nm=wavelength_nm,
    )
    
    print(f"  Nombre d'éléments: {len(microlens_array_doublet.elements)}")
    
    # =========================================================================
    # 2. Matrice de microtrous / Microhole Array
    # =========================================================================
    print("\n--- 2. Matrice de microtrous ---")
    
    # 2.1 Matrice de microtrous circulaires
    print("\n2.1 Matrice de microtrous circulaires (7x7)")
    microhole_array_circular = MicroholeArray(
        name="Microtrous circulaires",
        pitch_mm=0.5,
        num_elements_x=7,
        num_elements_y=7,
        hole_diameter_mm=0.4,
        aperture_shape=ApertureShape.CIRCULAR,
        material_name="opaque",
        temperature_K=293.15,
        wavelength_nm=wavelength_nm,
    )
    
    print(f"  Nombre de trous: {len(microhole_array_circular.elements)}")
    print(f"  Diamètre des trous: {microhole_array_circular.element_diameter_mm:.3f} mm")
    
    # 2.2 Matrice de microtrous hexagonaux
    print("\n2.2 Matrice de microtrous hexagonaux (5x5)")
    microhole_array_hexagonal = MicroholeArray(
        name="Microtrous hexagonaux",
        pitch_mm=0.6,
        num_elements_x=5,
        num_elements_y=5,
        hole_diameter_mm=0.5,
        aperture_shape=ApertureShape.HEXAGONAL,
        material_name="opaque",
        temperature_K=293.15,
        wavelength_nm=wavelength_nm,
    )
    
    print(f"  Nombre de trous: {len(microhole_array_hexagonal.elements)}")
    
    # =========================================================================
    # 3. Matrice de microprismes / Microprism Array
    # =========================================================================
    print("\n--- 3. Matrice de microprismes ---")
    
    # 3.1 Matrice de microprismes à 45°
    print("\n3.1 Matrice de microprismes (45°, 4x4)")
    microprism_array = MicroprismArray(
        name="Microprismes 45°",
        pitch_mm=1.0,
        num_elements_x=4,
        num_elements_y=4,
        apex_angle_deg=45.0,
        base_length_mm=0.9,
        height_mm=0.9,
        material_name="BK7",
        temperature_K=293.15,
        wavelength_nm=wavelength_nm,
    )
    
    print(f"  Nombre de prismes: {len(microprism_array.elements)}")
    print(f"  Angle au sommet: {microprism_array.element_kwargs.get('apex_angle_deg', 0.0)}°")
    
    # Calculer la carte de déviation
    if MATH_TOOLS_AVAILABLE:
        from MathAndPhysicsTools import create_grid
        grid_x, grid_y = create_grid(8.0, 256)
    else:
        x = np.linspace(-4, 4, 256)
        y = np.linspace(-4, 4, 256)
        grid_x, grid_y = np.meshgrid(x, y)
    
    deviation_map = microprism_array.get_deviation_map(grid_x, grid_y)
    
    # Visualisation de la déviation
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(
        deviation_map,
        extent=[-4, 4, -4, 4],
        cmap='coolwarm'
    )
    ax.set_title("Carte de déviation - Microprismes 45°")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=ax, label="Déviation (°)")
    plt.tight_layout()
    plt.savefig(f'{output_dir}/example11_microprism_deviation_map.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # =========================================================================
    # 4. Matrice de microréseaux / Micrograting Array
    # =========================================================================
    print("\n--- 4. Matrice de microréseaux ---")
    
    # 4.1 Matrice de microréseaux (100 lignes/mm)
    print("\n4.1 Matrice de microréseaux (100 lignes/mm, 3x3)")
    micrograting_array = MicrogratingArray(
        name="Microréseaux 100 l/mm",
        pitch_mm=1.5,
        num_elements_x=3,
        num_elements_y=3,
        lines_per_mm=100.0,
        orientation_deg=0.0,
        grating_type="transmission",
        material_name="Fused_Silica",
        temperature_K=293.15,
        wavelength_nm=wavelength_nm,
    )
    
    print(f"  Nombre de réseaux: {len(micrograting_array.elements)}")
    print(f"  Lignes par mm: {micrograting_array.element_kwargs.get('lines_per_mm', 0.0)}")
    
    # Calculer la carte d'efficacité pour l'ordre 1
    efficiency_map_order1 = micrograting_array.get_diffraction_orders_map(1, grid_x, grid_y)
    
    # Visualisation de l'efficacité
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(
        efficiency_map_order1,
        extent=[-4, 4, -4, 4],
        cmap='viridis'
    )
    ax.set_title("Efficacité de diffraction - Ordre 1")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=ax, label="Efficacité")
    plt.tight_layout()
    plt.savefig(f'{output_dir}/example11_micrograting_efficiency_map.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # =========================================================================
    # 5. Visualisation des matrices / Array Visualization
    # =========================================================================
    print("\n--- 5. Visualisation des matrices ---")
    
    # 5.1 Visualisation de la matrice de microlentilles
    print("\n5.1 Visualisation de la matrice de microlentilles (3x3)")
    microlens_array_edge.visualize(
        grid_size_mm=8.0,
        num_points=256,
        save_dir=output_dir,
    )
    print(f"  Image sauvegardée: {output_dir}/Microlentilles bord-à-bord_visualization.png")
    
    # 5.2 Visualisation de la matrice de microtrous
    print("\n5.2 Visualisation de la matrice de microtrous (7x7)")
    microhole_array_circular.visualize(
        grid_size_mm=8.0,
        num_points=256,
        save_dir=output_dir,
    )
    print(f"  Image sauvegardée: {output_dir}/Microtrous circulaires_visualization.png")
    
    # 5.3 Visualisation de la matrice de microprismes
    print("\n5.3 Visualisation de la matrice de microprismes (4x4)")
    microprism_array.visualize(
        grid_size_mm=8.0,
        num_points=256,
        save_dir=output_dir,
    )
    print(f"  Image sauvegardée: {output_dir}/Microprismes 45°_visualization.png")
    
    # 5.4 Visualisation de la matrice de microréseaux
    print("\n5.4 Visualisation de la matrice de microréseaux (3x3)")
    micrograting_array.visualize(
        grid_size_mm=8.0,
        num_points=256,
        save_dir=output_dir,
    )
    print(f"  Image sauvegardée: {output_dir}/Microréseaux 100 l/mm_visualization.png")
    
    # =========================================================================
    # 6. Application d'une WFE globale / Global WFE Application
    # =========================================================================
    print("\n--- 6. Application d'une WFE globale ---")
    
    # 6.1 Créer une matrice avec WFE globale
    print("\n6.1 Matrice de microlentilles avec WFE globale")
    microlens_array_wfe = MicrolensArray(
        name="Microlentilles avec WFE",
        pitch_mm=1.0,
        num_elements_x=3,
        num_elements_y=3,
        focal_length_mm=10.0,
        material_name="Fused_Silica",
        temperature_K=293.15,
        wavelength_nm=wavelength_nm,
    )
    
    # Appliquer une WFE globale
    global_wfe = WaveFrontError(
        surface_roughness_nm=5.0,
        parallelism_arcsec=2.0,
        zernike_coefficients={(2, 0): 20.0},  # Defocus
        seed=42,
    )
    microlens_array_wfe.apply_global_wfe(global_wfe)
    
    print(f"  WFE globale appliquée: rugosité={global_wfe.surface_roughness_nm} nm, parallélisme={global_wfe.parallelism_arcsec} arcsec")
    
    # Calculer la phase avec et sans WFE
    if MATH_TOOLS_AVAILABLE:
        grid_x, grid_y = create_grid(8.0, 256)
    else:
        x = np.linspace(-4, 4, 256)
        y = np.linspace(-4, 4, 256)
        grid_x, grid_y = np.meshgrid(x, y)
    
    phase_with_wfe = microlens_array_wfe.get_phase_map(grid_x, grid_y, include_global_wfe=True)
    phase_without_wfe = microlens_array_wfe.get_phase_map(grid_x, grid_y, include_global_wfe=False)
    
    # Visualisation de la comparaison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = axes[0].imshow(
        phase_without_wfe,
        extent=[-4, 4, -4, 4],
        cmap='coolwarm'
    )
    axes[0].set_title("Phase sans WFE globale")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    plt.colorbar(im1, ax=axes[0], label="Phase (nm)")
    
    im2 = axes[1].imshow(
        phase_with_wfe,
        extent=[-4, 4, -4, 4],
        cmap='coolwarm'
    )
    axes[1].set_title("Phase avec WFE globale")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("y (mm)")
    plt.colorbar(im2, ax=axes[1], label="Phase (nm)")
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/example11_wfe_comparison.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # Calculer PV et RMS
    pv_without = np.max(phase_without_wfe) - np.min(phase_without_wfe)
    rms_without = np.std(phase_without_wfe)
    pv_with = np.max(phase_with_wfe) - np.min(phase_with_wfe)
    rms_with = np.std(phase_with_wfe)
    
    print(f"  Sans WFE: PV={pv_without:.1f} nm, RMS={rms_without:.1f} nm")
    print(f"  Avec WFE: PV={pv_with:.1f} nm, RMS={rms_with:.1f} nm")
    
    # =========================================================================
    # 7. Dilatation thermique / Thermal Expansion
    # =========================================================================
    print("\n--- 7. Dilatation thermique ---")
    
    # 7.1 Matrice à température initiale
    print("\n7.1 Matrice de microlentilles à 20°C")
    microlens_array_thermal = MicrolensArray(
        name="Microlentilles thermiques",
        pitch_mm=1.0,
        num_elements_x=5,
        num_elements_y=5,
        focal_length_mm=10.0,
        material_name="Fused_Silica",
        temperature_K=293.15,  # 20°C
        wavelength_nm=wavelength_nm,
    )
    
    initial_pitch = microlens_array_thermal.pitch_mm
    initial_center_to_center = microlens_array_thermal.center_to_center_mm
    initial_positions = [e.position_mm for e in microlens_array_thermal.elements]
    
    print(f"  Pitch initial (bord à bord): {initial_pitch:.6f} mm")
    print(f"  Centre à centre initial: {initial_center_to_center:.6f} mm")
    print(f"  Position de l'élément (0,0): {initial_positions[0]}")
    print(f"  Position de l'élément (2,2): {initial_positions[12]}")
    
    # 7.2 Mettre à jour la température à 100°C
    print("\n7.2 Mise à jour de la température à 100°C")
    microlens_array_thermal.update_temperature(373.15)  # 100°C
    
    new_pitch = microlens_array_thermal.pitch_mm
    new_center_to_center = microlens_array_thermal.center_to_center_mm
    new_positions = [e.position_mm for e in microlens_array_thermal.elements]
    
    print(f"  Nouveau pitch (bord à bord): {new_pitch:.6f} mm")
    print(f"  Nouveau centre à centre: {new_center_to_center:.6f} mm")
    print(f"  Nouvelle position de l'élément (0,0): {new_positions[0]}")
    print(f"  Nouvelle position de l'élément (2,2): {new_positions[12]}")
    
    # Calculer la dilatation
    delta_pitch = new_pitch - initial_pitch
    delta_center = new_center_to_center - initial_center_to_center
    
    print(f"  Dilatation du pitch: {delta_pitch:.6f} mm ({delta_pitch/initial_pitch*100:.2f}%)")
    print(f"  Dilatation centre à centre: {delta_center:.6f} mm ({delta_center/initial_center_to_center*100:.2f}%)")
    
    # 7.3 Visualisation avant/après dilatation
    print("\n7.3 Visualisation avant/après dilatation")
    
    # Recalculer les positions initiales pour la visualisation
    microlens_array_thermal_initial = MicrolensArray(
        name="Microlentilles thermiques (initial)",
        pitch_mm=1.0,
        num_elements_x=5,
        num_elements_y=5,
        focal_length_mm=10.0,
        material_name="Fused_Silica",
        temperature_K=293.15,
        wavelength_nm=wavelength_nm,
    )
    
    # Visualiser la phase avant et après dilatation
    phase_initial = microlens_array_thermal_initial.get_phase_map(grid_x, grid_y)
    phase_expanded = microlens_array_thermal.get_phase_map(grid_x, grid_y)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = axes[0].imshow(
        phase_initial,
        extent=[-4, 4, -4, 4],
        cmap='coolwarm'
    )
    axes[0].set_title("Phase à 20°C")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    plt.colorbar(im1, ax=axes[0], label="Phase (nm)")
    
    im2 = axes[1].imshow(
        phase_expanded,
        extent=[-4, 4, -4, 4],
        cmap='coolwarm'
    )
    axes[1].set_title("Phase à 100°C")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("y (mm)")
    plt.colorbar(im2, ax=axes[1], label="Phase (nm)")
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/example11_thermal_expansion_comparison.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # =========================================================================
    # 8. Application à un faisceau / Beam Application
    # =========================================================================
    print("\n--- 8. Application à un faisceau ---")
    
    # 8.1 Application d'une matrice de microlentilles à un faisceau
    print("\n8.1 Application d'une matrice de microlentilles à un faisceau gaussien")
    
    # Créer un faisceau gaussien
    beam = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=8.0,
        energy=1.0,
        num_points=256,
    )
    electric_field = beam.generate_electric_field(method="gaussian", sigma_mm=2.0)
    beam.electric_field = electric_field
    beam.intensity = beam.compute_intensity_from_electric_field(electric_field)
    beam.phase = beam.extract_phase_from_electric_field(electric_field)
    
    pv_initial, rms_initial = beam.compute_pv_rms(beam.phase)
    print(f"  Faisceau initial: PV={pv_initial:.2f} nm, RMS={rms_initial:.2f} nm")
    
    # Appliquer la matrice de microlentilles
    beam_after_microlenses = microlens_array_edge.apply_to_beam(beam)
    
    pv_after, rms_after = beam_after_microlenses.compute_pv_rms(beam_after_microlenses.phase)
    print(f"  Faisceau après matrice: PV={pv_after:.2f} nm, RMS={rms_after:.2f} nm")
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = axes[0].imshow(
        beam.intensity,
        extent=[-4, 4, -4, 4],
        cmap='viridis'
    )
    axes[0].set_title("Intensité initiale")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    plt.colorbar(im1, ax=axes[0], label="Intensité (a.u.)")
    
    im2 = axes[1].imshow(
        beam_after_microlenses.intensity,
        extent=[-4, 4, -4, 4],
        cmap='viridis'
    )
    axes[1].set_title("Intensité après matrice de microlentilles")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("y (mm)")
    plt.colorbar(im2, ax=axes[1], label="Intensité (a.u.)")
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/example11_beam_after_microlens_array.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # 8.2 Application d'une matrice de microtrous à un faisceau
    print("\n8.2 Application d'une matrice de microtrous à un faisceau")
    
    # Créer un nouveau faisceau
    beam_holes = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=8.0,
        energy=1.0,
        num_points=256,
    )
    electric_field_holes = beam_holes.generate_electric_field(method="uniform")
    beam_holes.electric_field = electric_field_holes
    beam_holes.intensity = beam_holes.compute_intensity_from_electric_field(electric_field_holes)
    beam_holes.phase = beam_holes.extract_phase_from_electric_field(electric_field_holes)
    
    # Appliquer la matrice de microtrous
    beam_after_holes = microhole_array_circular.apply_to_beam(beam_holes)
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = axes[0].imshow(
        beam_holes.intensity,
        extent=[-4, 4, -4, 4],
        cmap='viridis'
    )
    axes[0].set_title("Intensité initiale (uniforme)")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    plt.colorbar(im1, ax=axes[0], label="Intensité (a.u.)")
    
    im2 = axes[1].imshow(
        beam_after_holes.intensity,
        extent=[-4, 4, -4, 4],
        cmap='viridis'
    )
    axes[1].set_title("Intensité après matrice de microtrous")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("y (mm)")
    plt.colorbar(im2, ax=axes[1], label="Intensité (a.u.)")
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/example11_beam_after_microhole_array.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # =========================================================================
    # 9. Intégration avec Propagation.py / Integration with Propagation.py
    # =========================================================================
    print("\n--- 9. Intégration avec Propagation.py ---")
    
    # 9.1 Propagation à travers une matrice de microlentilles
    print("\n9.1 Propagation à travers une matrice de microlentilles")
    
    # Créer un système simple avec une matrice de microlentilles
    # et une propagation après la matrice
    
    # Matrice de microlentilles
    microlens_array_prop = MicrolensArray(
        name="Microlentilles pour propagation",
        pitch_mm=1.0,
        num_elements_x=3,
        num_elements_y=3,
        focal_length_mm=10.0,
        material_name="Fused_Silica",
        temperature_K=293.15,
        wavelength_nm=wavelength_nm,
    )
    
    # Créer un faisceau
    beam_prop = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=8.0,
        energy=1.0,
        num_points=128,  # Réduire pour accélérer
    )
    electric_field_prop = beam_prop.generate_electric_field(method="gaussian", sigma_mm=2.0)
    beam_prop.electric_field = electric_field_prop
    beam_prop.intensity = beam_prop.compute_intensity_from_electric_field(electric_field_prop)
    beam_prop.phase = beam_prop.extract_phase_from_electric_field(electric_field_prop)
    
    # Appliquer la matrice
    beam_after_array = microlens_array_prop.apply_to_beam(beam_prop)
    
    # Propager le faisceau sur 50 mm
    try:
        propagator = Propagation(
            wavelength_nm=wavelength_nm,
            propagation_distance_mm=50.0,
            input_diameter_mm=8.0,
            output_diameter_mm=8.0,
            num_points=128,
            method="angular_spectrum",
        )
        
        propagated_field = propagator.propagate(beam_after_array.electric_field)
        
        # Créer un nouveau faisceau avec le champ propagé
        beam_propagated = Beam(
            wavelength_nm=wavelength_nm,
            diameter_mm=8.0,
            energy=beam_after_array.energy,
            num_points=128,
        )
        beam_propagated.electric_field = propagated_field
        beam_propagated.intensity = beam_propagated.compute_intensity_from_electric_field(propagated_field)
        beam_propagated.phase = beam_propagated.extract_phase_from_electric_field(propagated_field)
        
        print(f"  Propagation réussie sur 50 mm")
        
        # Visualisation
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        im1 = axes[0].imshow(
            beam_prop.intensity,
            extent=[-4, 4, -4, 4],
            cmap='viridis'
        )
        axes[0].set_title("Intensité initiale")
        axes[0].set_xlabel("x (mm)")
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(
            beam_after_array.intensity,
            extent=[-4, 4, -4, 4],
            cmap='viridis'
        )
        axes[1].set_title("Intensité après matrice")
        axes[1].set_xlabel("x (mm)")
        plt.colorbar(im2, ax=axes[1])
        
        im3 = axes[2].imshow(
            beam_propagated.intensity,
            extent=[-4, 4, -4, 4],
            cmap='viridis'
        )
        axes[2].set_title("Intensité après propagation")
        axes[2].set_xlabel("x (mm)")
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/example11_propagation_through_array.png', dpi=150, bbox_inches='tight')
        plt.close('all')
        
    except Exception as e:
        print(f"  ⚠️  Propagation échouée: {e}")
    
    # =========================================================================
    # 10. Comparaison des types d'espacement / Spacing Type Comparison
    # =========================================================================
    print("\n--- 10. Comparaison des types d'espacement ---")
    
    # 10.1 Matrice bord à bord vs centre à centre
    print("\n10.1 Comparaison bord à bord vs centre à centre")
    
    # Matrice bord à bord
    array_edge = MicrolensArray(
        name="Bord à bord",
        pitch_mm=1.0,
        num_elements_x=3,
        num_elements_y=3,
        focal_length_mm=10.0,
        spacing_type=SpacingType.EDGE_TO_EDGE,
    )
    
    # Matrice centre à centre
    array_center = MicrolensArray(
        name="Centre à centre",
        pitch_mm=1.0,
        num_elements_x=3,
        num_elements_y=3,
        focal_length_mm=10.0,
        spacing_type=SpacingType.CENTER_TO_CENTER,
    )
    
    print(f"  Bord à bord:")
    print(f"    - Pitch (bord à bord): {array_edge.pitch_mm:.3f} mm")
    print(f"    - Centre à centre: {array_edge.center_to_center_mm:.3f} mm")
    print(f"    - Diamètre élément: {array_edge.element_diameter_mm:.3f} mm")
    print(f"    - Espace entre bords: {array_edge.get_element_spacing()['gap_mm']:.3f} mm")
    
    print(f"  Centre à centre:")
    print(f"    - Pitch (centre à centre): {array_center.pitch_mm:.3f} mm")
    print(f"    - Centre à centre: {array_center.center_to_center_mm:.3f} mm")
    print(f"    - Diamètre élément: {array_center.element_diameter_mm:.3f} mm")
    print(f"    - Espace entre bords: {array_center.get_element_spacing()['gap_mm']:.3f} mm")
    
    # Visualisation des masques
    if MATH_TOOLS_AVAILABLE:
        grid_x, grid_y = create_grid(8.0, 256)
    else:
        x = np.linspace(-4, 4, 256)
        y = np.linspace(-4, 4, 256)
        grid_x, grid_y = np.meshgrid(x, y)
    
    transmission_edge = array_edge.get_transmission_map(grid_x, grid_y)
    transmission_center = array_center.get_transmission_map(grid_x, grid_y)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = axes[0].imshow(
        transmission_edge,
        extent=[-4, 4, -4, 4],
        cmap='gray'
    )
    axes[0].set_title("Masque - Bord à bord")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    plt.colorbar(im1, ax=axes[0], label="Transmission")
    
    im2 = axes[1].imshow(
        transmission_center,
        extent=[-4, 4, -4, 4],
        cmap='gray'
    )
    axes[1].set_title("Masque - Centre à centre")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("y (mm)")
    plt.colorbar(im2, ax=axes[1], label="Transmission")
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/example11_spacing_comparison.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # =========================================================================
    # 11. Matrice personnalisée / Custom Array
    # =========================================================================
    print("\n--- 11. Matrice personnalisée ---")
    
    # 11.1 Matrice avec des optiques personnalisées
    print("\n11.1 Matrice avec des optiques personnalisées")
    
    # Créer une matrice personnalisée avec des lentilles idéales
    custom_array = Microstructure(
        name="Matrice personnalisée",
        pitch_mm=1.0,
        num_elements_x=3,
        num_elements_y=3,
        element_type="custom",
        element_kwargs={
            'optic_class': IdealLens,
            'focal_length_mm': 15.0,
            'material_name': "BK7",
        },
        spacing_type=SpacingType.EDGE_TO_EDGE,
        material_name="Fused_Silica",
        temperature_K=293.15,
        wavelength_nm=wavelength_nm,
    )
    
    print(f"  Nombre d'éléments: {len(custom_array.elements)}")
    print(f"  Type d'élément: {type(custom_array.elements[0].optic).__name__}")
    
    # 11.2 Matrice avec un mélange d'optiques (exemple avancé)
    print("\n11.2 Matrice avec un mélange d'optiques")
    
    # Créer une matrice vide
    mixed_array = Microstructure(
        name="Matrice mixte",
        pitch_mm=1.5,
        num_elements_x=3,
        num_elements_y=3,
        element_type="lens",
        element_kwargs={
            'focal_length_mm': 10.0,
            'material_name': "Fused_Silica",
        },
        spacing_type=SpacingType.EDGE_TO_EDGE,
        material_name="Fused_Silica",
        temperature_K=293.15,
        wavelength_nm=wavelength_nm,
    )
    
    # Remplacer certains éléments par des microtrous
    # Élément central (1,1)
    specs_hole = OpticSpecifications(
        diameter_mm=mixed_array.element_diameter_mm,
        thickness_mm=0.0,
        material_name="opaque",
        aperture_shape=ApertureShape.CIRCULAR,
    )
    hole = ApertureStop(
        name="Trou central",
        diameter_mm=mixed_array.element_diameter_mm,
        specifications=specs_hole,
        position_mm=(0.0, 0.0, 0.0),
        wavelength_nm=wavelength_nm,
    )
    mixed_array.elements[4] = MicroOpticElement(
        optic=hole,
        position_mm=(0.0, 0.0),
        index=(1, 1),
        element_diameter_mm=mixed_array.element_diameter_mm,
    )
    
    print(f"  Matrice avec 8 lentilles et 1 trou central")
    
    # Visualisation
    mixed_phase = mixed_array.get_phase_map(grid_x, grid_y)
    mixed_transmission = mixed_array.get_transmission_map(grid_x, grid_y)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = axes[0].imshow(
        mixed_phase,
        extent=[-4, 4, -4, 4],
        cmap='coolwarm'
    )
    axes[0].set_title("Phase - Matrice mixte")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    plt.colorbar(im1, ax=axes[0], label="Phase (nm)")
    
    im2 = axes[1].imshow(
        mixed_transmission,
        extent=[-4, 4, -4, 4],
        cmap='gray'
    )
    axes[1].set_title("Transmission - Matrice mixte")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("y (mm)")
    plt.colorbar(im2, ax=axes[1], label="Transmission")
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/example11_mixed_array.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # =========================================================================
    # 12. Fabrique de microstructures / Microstructure Factory
    # =========================================================================
    print("\n--- 12. Fabrique de microstructures ---")
    
    # 12.1 Utilisation de la fabrique
    print("\n12.1 Utilisation de create_microstructure()")
    
    # Créer une matrice de microlentilles avec la fabrique
    array_from_factory = create_microstructure(
        microstructure_type="microlens_array",
        name="Fabrique - Microlentilles",
        pitch_mm=1.0,
        num_elements_x=3,
        num_elements_y=3,
        focal_length_mm=10.0,
    )
    
    print(f"  Type: {type(array_from_factory).__name__}")
    print(f"  Nombre d'éléments: {len(array_from_factory.elements)}")
    
    # Créer une matrice de microtrous avec la fabrique
    array_from_factory = create_microstructure(
        microstructure_type="microhole_array",
        name="Fabrique - Microtrous",
        pitch_mm=0.8,
        num_elements_x=5,
        num_elements_y=5,
        hole_diameter_mm=0.6,
    )
    
    print(f"  Type: {type(array_from_factory).__name__}")
    print(f"  Nombre d'éléments: {len(array_from_factory.elements)}")
    
    # =========================================================================
    # 13. Exemple complet : Système Shack-Hartmann simplifié
    # =========================================================================
    print("\n--- 13. Exemple complet : Système Shack-Hartmann simplifié ---")
    
    # 13.1 Création d'une matrice de microlentilles pour Shack-Hartmann
    print("\n13.1 Matrice de microlentilles pour Shack-Hartmann")
    
    shack_hartmann_array = MicrolensArray(
        name="Shack-Hartmann",
        pitch_mm=0.2,  # Pas typique pour Shack-Hartmann
        num_elements_x=11,
        num_elements_y=11,
        focal_length_mm=5.0,  # Distance focale typique
        lens_type="ideal",
        material_name="Fused_Silica",
        temperature_K=293.15,
        wavelength_nm=wavelength_nm,
    )
    
    print(f"  Nombre de microlentilles: {len(shack_hartmann_array.elements)}")
    print(f"  Pas (bord à bord): {shack_hartmann_array.pitch_mm:.3f} mm")
    print(f"  Distance focale: {shack_hartmann_array.element_kwargs.get('focal_length_mm', 0.0)} mm")
    
    # 13.2 Application à un faisceau avec front d'onde déformé
    print("\n13.2 Application à un faisceau avec front d'onde déformé")
    
    # Créer un faisceau avec une aberration (defocus)
    beam_sh = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=5.0,
        energy=1.0,
        num_points=256,
    )
    
    # Ajouter une aberration de defocus
    wfe_sh = WaveFrontError(
        zernike_coefficients={(2, 0): 50.0},  # Defocus
        seed=42,
    )
    
    # Créer une grille pour le faisceau
    if MATH_TOOLS_AVAILABLE:
        grid_x_sh, grid_y_sh = create_grid(beam_sh.diameter_mm, beam_sh.num_points)
    else:
        x = np.linspace(-beam_sh.diameter_mm/2, beam_sh.diameter_mm/2, beam_sh.num_points)
        y = np.linspace(-beam_sh.diameter_mm/2, beam_sh.diameter_mm/2, beam_sh.num_points)
        grid_x_sh, grid_y_sh = np.meshgrid(x, y)
    
    # Générer le champ électrique avec l'aberration
    phase_aberration = wfe_sh.generate_phase_map(grid_x_sh, grid_y_sh, wavelength_nm, seed=42)
    electric_field_sh = np.exp(1j * phase_aberration * 2 * np.pi / wavelength_nm)
    beam_sh.electric_field = electric_field_sh
    beam_sh.intensity = beam_sh.compute_intensity_from_electric_field(electric_field_sh)
    beam_sh.phase = beam_sh.extract_phase_from_electric_field(electric_field_sh)
    
    pv_sh_initial, rms_sh_initial = beam_sh.compute_pv_rms(beam_sh.phase)
    print(f"  Faisceau initial avec defocus: PV={pv_sh_initial:.2f} nm, RMS={rms_sh_initial:.2f} nm")
    
    # Appliquer la matrice de Shack-Hartmann
    beam_sh_after = shack_hartmann_array.apply_to_beam(beam_sh)
    
    pv_sh_after, rms_sh_after = beam_sh_after.compute_pv_rms(beam_sh_after.phase)
    print(f"  Faisceau après matrice Shack-Hartmann: PV={pv_sh_after:.2f} nm, RMS={rms_sh_after:.2f} nm")
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = axes[0].imshow(
        beam_sh.phase,
        extent=[-2.5, 2.5, -2.5, 2.5],
        cmap='coolwarm'
    )
    axes[0].set_title("Phase initiale (avec defocus)")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    plt.colorbar(im1, ax=axes[0], label="Phase (nm)")
    
    im2 = axes[1].imshow(
        beam_sh_after.phase,
        extent=[-2.5, 2.5, -2.5, 2.5],
        cmap='coolwarm'
    )
    axes[1].set_title("Phase après matrice Shack-Hartmann")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("y (mm)")
    plt.colorbar(im2, ax=axes[1], label="Phase (nm)")
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/example11_shack_hartmann.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    print("\n" + "="*80)
    print("Example 11 terminé avec succès !")
    print(f"Les images ont été sauvegardées dans {output_dir}/")
    print("="*80)


if __name__ == "__main__":
    os.makedirs('examples/output', exist_ok=True)
    run_microstructure_example()

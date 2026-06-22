"""
Example 11: Microstructure Arrays - Complete Demonstration
FR: Exemple complet d'utilisation du module Microstructure.py.
    Démonstration de :
    - Création de matrices de microlentilles (MicrolensArray)
    - Création de matrices de microtrous (MicroholeArray)
    - Création de matrices de microprismes (MicroprismArray)
    - Espacement bord-à-bord entre les éléments (par défaut = 0, éléments joints)
    - Application d'une WFE globale à la matrice complète
    - Prise en compte de la dilatation thermique et de son effet sur les positions
    - Intégration avec Optiques.py, Material_Behaviour.py, Beam.py et Propagation.py
    - Visualisation des matrices avec colormap "Jet" pour la phase et "hot" pour l'intensité
    
    Chaque image générée aura :
    - Une échelle visuelle
    - Le PV (Peak-to-Valley) et le RMS des valeurs
    
    Unités :
    - Longueurs : mm (taille de la matrice), µm (taille des éléments)
    - Longueur d'onde : nm
    - Phase : nm (principale), λ (longueur d'onde), rad, mrad
    
EN: Complete example of using Microstructure.py module.
    Demonstrates:
    - Creation of microlens arrays (MicrolensArray)
    - Creation of microhole arrays (MicroholeArray)
    - Creation of microprism arrays (MicroprismArray)
    - Edge-to-edge spacing between elements (default = 0, joined elements)
    - Global WFE application to the entire array
    - Thermal expansion consideration and its effect on positions
    - Integration with Optiques.py, Material_Behaviour.py, Beam.py and Propagation.py
    - Array visualization with "Jet" colormap for phase and "hot" for intensity
    
    Each generated image will have:
    - A visual scale
    - PV (Peak-to-Valley) and RMS values
    
    Units:
    - Lengths: mm (array size), µm (element size)
    - Wavelength: nm
    - Phase: nm (main), λ (wavelength), rad, mrad

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch

Dependencies:
    - numpy
    - matplotlib
    - Microstructure.py
    - Optiques.py
    - Beam.py
    - Material_Behaviour.py
    - Propagation.py (optional)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Import des modules locaux
from Microstructure import (
    MicroOpticsArray, MicrolensArray, MicroholeArray, MicroprismArray,
    ArrayPattern, MicroOpticType, create_microlens_array, create_microhole_array,
    create_microprism_array
)
from Optiques import WaveFrontError, ApertureShape
from Beam import Beam
from Material_Behaviour import MaterialBehaviour, STANDARD_TEMPERATURE_K


def run_microstructure_example():
    """
    FR: Exécute l'exemple complet de microstructures.
        Démonstration de toutes les fonctionnalités de Microstructure.py.
    """
    print("\n" + "="*80)
    print("Example 11: Microstructure Arrays - Complete Demonstration")
    print("="*80)
    
    # Paramètres globaux
    wavelength_nm = 633.0
    output_dir = 'examples/output'
    os.makedirs(output_dir, exist_ok=True)
    
    # =========================================================================
    # 1. ESPACEMENT BORD-À-BORD / EDGE-TO-EDGE SPACING
    # =========================================================================
    print("\n--- 1. Espacement bord-à-bord ---")
    
    # 1.1 Matrice avec éléments joints (espacement bord-à-bord = 0)
    print("\n1.1 Matrice de microlentilles JOINTE (espacement bord-à-bord = 0)")
    microlens_array_joined = create_microlens_array(
        name="Microlentilles jointes",
        pitch_mm=1.0,          # Distance bord-à-bord = 0
        num_elements_x=5,
        num_elements_y=5,
        focal_length_mm=10.0,
        array_pattern=ArrayPattern.SQUARE,
        material_name="Fused_Silica",
        wavelength_nm=wavelength_nm,
        edge_to_edge_spacing_mm=0.0,  # Éléments joints
    )
    
    print(f"  {microlens_array_joined}")
    print(f"  Diamètre des microlentilles: {microlens_array_joined.element_diameter_mm:.2f} mm")
    print(f"  Distance centre-à-centre: {microlens_array_joined.center_to_center_mm:.2f} mm")
    print(f"  Largeur totale: {microlens_array_joined.total_width_mm:.2f} mm")
    print(f"  Hauteur totale: {microlens_array_joined.total_height_mm:.2f} mm")
    
    # Vérification : centre-à-centre = bord-à-bord + diamètre
    expected_ctc = microlens_array_joined.pitch_mm + microlens_array_joined.element_diameter_mm
    assert abs(microlens_array_joined.center_to_center_mm - expected_ctc) < 1e-6, \
        f"Distance centre-à-centre incorrecte: attendu {expected_ctc:.2f}, obtenu {microlens_array_joined.center_to_center_mm:.2f}"
    print(f"  ✓ Vérification: centre-à-centre = bord-à-bord + diamètre")
    
    # Visualisation
    microlens_array_joined.visualize(
        grid_size_mm=6.0,
        num_points=256,
        save_path=f'{output_dir}/example11_microlens_array_joined.png'
    )
    print(f"  ✓ Image sauvegardée: example11_microlens_array_joined.png")
    
    # 1.2 Matrice avec espacement de 0.1 mm entre les bords
    print("\n1.2 Matrice de microlentilles avec ESPACEMENT (0.1 mm entre les bords)")
    microlens_array_spaced = create_microlens_array(
        name="Microlentilles espacées",
        pitch_mm=1.1,          # Distance bord-à-bord = 0.1 mm
        num_elements_x=5,
        num_elements_y=5,
        focal_length_mm=10.0,
        array_pattern=ArrayPattern.SQUARE,
        material_name="Fused_Silica",
        wavelength_nm=wavelength_nm,
        edge_to_edge_spacing_mm=0.1,  # Espacement de 0.1 mm entre les bords
    )
    
    print(f"  {microlens_array_spaced}")
    print(f"  Diamètre des microlentilles: {microlens_array_spaced.element_diameter_mm:.2f} mm")
    print(f"  Distance bord-à-bord: {microlens_array_spaced.pitch_mm:.2f} mm")
    print(f"  Distance centre-à-centre: {microlens_array_spaced.center_to_center_mm:.2f} mm")
    
    # Vérification : diamètre = centre-à-centre - bord-à-bord
    expected_diameter = microlens_array_spaced.center_to_center_mm - microlens_array_spaced.pitch_mm
    assert abs(microlens_array_spaced.element_diameter_mm - expected_diameter) < 1e-6, \
        f"Diamètre incorrect: attendu {expected_diameter:.2f}, obtenu {microlens_array_spaced.element_diameter_mm:.2f}"
    print(f"  ✓ Vérification: diamètre = centre-à-centre - bord-à-bord")
    
    microlens_array_spaced.visualize(
        grid_size_mm=6.0,
        num_points=256,
        save_path=f'{output_dir}/example11_microlens_array_spaced.png'
    )
    print(f"  ✓ Image sauvegardée: example11_microlens_array_spaced.png")
    
    # 1.3 Matrice hexagonale
    print("\n1.3 Matrice hexagonale de microlentilles")
    microlens_array_hex = create_microlens_array(
        name="Microlentilles hexagonales",
        pitch_mm=1.0,
        num_elements_x=4,
        num_elements_y=4,
        focal_length_mm=10.0,
        array_pattern=ArrayPattern.HEXAGONAL,
        material_name="Fused_Silica",
        wavelength_nm=wavelength_nm,
    )
    
    print(f"  {microlens_array_hex}")
    print(f"  Motif: {microlens_array_hex.array_pattern.value}")
    print(f"  Largeur totale: {microlens_array_hex.total_width_mm:.2f} mm")
    print(f"  Hauteur totale: {microlens_array_hex.total_height_mm:.2f} mm")
    
    microlens_array_hex.visualize(
        grid_size_mm=6.0,
        num_points=256,
        save_path=f'{output_dir}/example11_microlens_array_hexagonal.png'
    )
    print(f"  ✓ Image sauvegardée: example11_microlens_array_hexagonal.png")
    
    # =========================================================================
    # 2. WFE GLOBALE / GLOBAL WAVEFRONT ERROR
    # =========================================================================
    print("\n--- 2. Application de WFE globale ---")
    
    # 2.1 WFE globale sur une matrice de microlentilles
    print("\n2.1 WFE globale sur une matrice de microlentilles (3x3)")
    microlens_with_wfe = create_microlens_array(
        name="Microlentilles avec WFE",
        pitch_mm=1.0,
        num_elements_x=3,
        num_elements_y=3,
        focal_length_mm=10.0,
        wavelength_nm=wavelength_nm,
    )
    
    # Ajouter une WFE globale
    microlens_with_wfe.set_global_wfe(WaveFrontError(
        surface_roughness_nm=5.0,
        parallelism_arcsec=2.0,
        zernike_coefficients={
            (2, 0): 10.0,   # Defocus
            (2, 2): 5.0,    # Astigmatisme
        },
        seed=42,
    ))
    
    print(f"  WFE globale appliquée:")
    print(f"    - Rugosité: {microlens_with_wfe.global_wfe.surface_roughness_nm:.1f} nm RMS")
    print(f"    - Parallélisme: {microlens_with_wfe.global_wfe.parallelism_arcsec:.1f} arcsec")
    print(f"    - Zernike: {microlens_with_wfe.global_wfe.zernike_coefficients}")
    
    # Créer une grille pour les calculs
    try:
        from MathAndPhysicsTools import create_grid
        grid_x, grid_y = create_grid(5.0, 256)
    except:
        x = y = np.linspace(-2.5, 2.5, 256)
        grid_x, grid_y = np.meshgrid(x, y)
    
    # Calculer la phase avec et sans WFE globale
    phase_with_wfe = microlens_with_wfe.get_total_phase_map(grid_x, grid_y)
    phase_without_wfe = microlens_with_wfe.get_total_phase_map(grid_x, grid_y, include_global_wfe=False)
    
    # Calculer PV et RMS
    pv_without = np.max(phase_without_wfe) - np.min(phase_without_wfe)
    rms_without = np.std(phase_without_wfe)
    pv_with = np.max(phase_with_wfe) - np.min(phase_with_wfe)
    rms_with = np.std(phase_with_wfe)
    
    print(f"  Sans WFE: PV={pv_without:.1f} nm, RMS={rms_without:.1f} nm")
    print(f"  Avec WFE: PV={pv_with:.1f} nm, RMS={rms_with:.1f} nm")
    
    # Visualisation avec colormap "Jet" pour la phase
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = axes[0].imshow(
        phase_without_wfe,
        extent=[-2.5, 2.5, -2.5, 2.5],
        cmap='Jet'
    )
    axes[0].set_title("Phase sans WFE globale\nPV={:.1f} nm, RMS={:.1f} nm".format(pv_without, rms_without))
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    plt.colorbar(im1, ax=axes[0], label="Phase (nm)")
    
    im2 = axes[1].imshow(
        phase_with_wfe,
        extent=[-2.5, 2.5, -2.5, 2.5],
        cmap='Jet'
    )
    axes[1].set_title("Phase avec WFE globale\nPV={:.1f} nm, RMS={:.1f} nm".format(pv_with, rms_with))
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("y (mm)")
    plt.colorbar(im2, ax=axes[1], label="Phase (nm)")
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/example11_wfe_comparison.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  ✓ Image sauvegardée: example11_wfe_comparison.png")
    
    # 2.2 Ajouter une composante supplémentaire à la WFE globale
    print("\n2.2 Ajout d'une composante supplémentaire à la WFE globale")
    microlens_with_wfe.add_global_wfe_component(
        surface_roughness_nm=2.0,
        zernike_coefficients={(3, 1): 5.0},  # Coma
    )
    
    phase_with_additional = microlens_with_wfe.get_total_phase_map(grid_x, grid_y)
    pv_additional = np.max(phase_with_additional) - np.min(phase_with_additional)
    rms_additional = np.std(phase_with_additional)
    
    print(f"  Avec WFE supplémentaire: PV={pv_additional:.1f} nm, RMS={rms_additional:.1f} nm")
    
    # Vérification : la phase doit avoir changé
    assert not np.allclose(phase_with_additional, phase_with_wfe), \
        "La phase doit changer après ajout d'une composante WFE"
    print(f"  ✓ Vérification: la phase a changé après ajout de composante")
    
    # =========================================================================
    # 3. DILATATION THERMIQUE / THERMAL EXPANSION
    # =========================================================================
    print("\n--- 3. Dilatation thermique ---")
    
    # 3.1 Calcul des informations de déformation
    print("\n3.1 Calcul des informations de déformation thermique (ΔT = +50 K)")
    microlens_thermal = create_microlens_array(
        name="Microlentilles thermiques",
        pitch_mm=1.0,
        num_elements_x=3,
        num_elements_y=3,
        focal_length_mm=10.0,
        material_name="BK7",
        wavelength_nm=wavelength_nm,
    )
    
    # Calculer la déformation sans l'appliquer
    deformation_info = microlens_thermal.get_thermal_deformation_info(293.15 + 50)
    
    print(f"  Informations de déformation pour ΔT = +50 K:")
    print(f"    Δpitch: {deformation_info['delta_pitch_mm']:.6f} mm")
    print(f"    Nouveau pitch: {deformation_info['new_pitch_mm']:.6f} mm")
    print(f"    Δdiamètre: {deformation_info['delta_diameter_mm']:.6f} mm")
    print(f"    Nouveau diamètre: {deformation_info['new_diameter_mm']:.6f} mm")
    print(f"    Δcentre-à-centre: {deformation_info['delta_center_to_center_mm']:.6f} mm")
    
    # 3.2 Application de la déformation thermique
    print("\n3.2 Application de la déformation thermique (ΔT = +50 K)")
    initial_pitch = microlens_thermal.pitch_mm
    initial_diameter = microlens_thermal.element_diameter_mm
    initial_ctc = microlens_thermal.center_to_center_mm
    initial_positions = [(mo.position_mm[0], mo.position_mm[1]) for mo in microlens_thermal.micro_optics]
    
    microlens_thermal.apply_thermal_deformation(293.15 + 50)
    
    new_pitch = microlens_thermal.pitch_mm
    new_diameter = microlens_thermal.element_diameter_mm
    new_ctc = microlens_thermal.center_to_center_mm
    new_positions = [(mo.position_mm[0], mo.position_mm[1]) for mo in microlens_thermal.micro_optics]
    
    print(f"  Avant: pitch={initial_pitch:.6f} mm, diamètre={initial_diameter:.6f} mm, ctc={initial_ctc:.6f} mm")
    print(f"  Après:  pitch={new_pitch:.6f} mm, diamètre={new_diameter:.6f} mm, ctc={new_ctc:.6f} mm")
    
    # Vérifications
    assert new_pitch > initial_pitch, "Le pitch doit augmenter avec la température"
    assert new_diameter > initial_diameter, "Le diamètre doit augmenter avec la température"
    assert new_ctc > initial_ctc, "La distance centre-à-centre doit augmenter"
    assert not all(np.allclose(ip, np) for ip, np in zip(initial_positions, new_positions)), \
        "Les positions doivent changer après dilatation thermique"
    print(f"  ✓ Toutes les vérifications passées")
    
    # Visualisation après déformation
    microlens_thermal.visualize(
        grid_size_mm=5.0,
        num_points=256,
        save_path=f'{output_dir}/example11_after_thermal_deformation.png'
    )
    print(f"  ✓ Image sauvegardée: example11_after_thermal_deformation.png")
    
    # 3.3 Comparaison avant/après dilatation
    print("\n3.3 Comparaison avant/après dilatation")
    
    # Recalculer les positions initiales pour la visualisation
    microlens_before = create_microlens_array(
        name="Avant déformation",
        pitch_mm=1.0,
        num_elements_x=3,
        num_elements_y=3,
        focal_length_mm=10.0,
        material_name="BK7",
        wavelength_nm=wavelength_nm,
    )
    
    # Calculer la phase avant et après
    phase_before = microlens_before.get_total_phase_map(grid_x, grid_y)
    phase_after = microlens_thermal.get_total_phase_map(grid_x, grid_y)
    
    pv_before = np.max(phase_before) - np.min(phase_before)
    rms_before = np.std(phase_before)
    pv_after = np.max(phase_after) - np.min(phase_after)
    rms_after = np.std(phase_after)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = axes[0].imshow(
        phase_before,
        extent=[-2.5, 2.5, -2.5, 2.5],
        cmap='Jet'
    )
    axes[0].set_title(f"Phase à 20°C\nPV={pv_before:.1f} nm, RMS={rms_before:.1f} nm")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    plt.colorbar(im1, ax=axes[0], label="Phase (nm)")
    
    im2 = axes[1].imshow(
        phase_after,
        extent=[-2.5, 2.5, -2.5, 2.5],
        cmap='Jet'
    )
    axes[1].set_title(f"Phase à 70°C\nPV={pv_after:.1f} nm, RMS={rms_after:.1f} nm")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("y (mm)")
    plt.colorbar(im2, ax=axes[1], label="Phase (nm)")
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/example11_thermal_expansion_comparison.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  ✓ Image sauvegardée: example11_thermal_expansion_comparison.png")
    
    # =========================================================================
    # 4. MATRICE DE MICROTROUS / MICROHOLE ARRAY
    # =========================================================================
    print("\n--- 4. Matrice de microtrous ---")
    
    # 4.1 Création d'une matrice de microtrous
    print("\n4.1 Matrice de microtrous (5x5)")
    microhole_array = create_microhole_array(
        name="Microtrous",
        pitch_mm=0.5,          # Distance bord-à-bord
        num_elements_x=5,
        num_elements_y=5,
        hole_diameter_mm=0.2,  # Diamètre des trous
        wavelength_nm=wavelength_nm,
    )
    
    print(f"  {microhole_array}")
    print(f"  Diamètre des trous: {microhole_array.hole_diameter_mm:.2f} mm")
    print(f"  Distance centre-à-centre: {microhole_array.center_to_center_mm:.2f} mm")
    
    # Visualisation
    microhole_array.visualize(
        grid_size_mm=3.0,
        num_points=256,
        save_path=f'{output_dir}/example11_microhole_array.png'
    )
    print(f"  ✓ Image sauvegardée: example11_microhole_array.png")
    
    # 4.2 Figure de diffraction
    print("\n4.2 Figure de diffraction à 10 mm")
    try:
        from Propagation import Propagation
        PROPAGATION_AVAILABLE = True
    except ImportError:
        PROPAGATION_AVAILABLE = False
    
    if PROPAGATION_AVAILABLE:
        distance_mm = 10.0
        diffraction_pattern = microhole_array.get_diffraction_pattern(
            distance_mm=distance_mm,
            grid_size_mm=5.0,
            num_points=256,
        )
        
        pv_diff = np.max(diffraction_pattern) - np.min(diffraction_pattern)
        rms_diff = np.std(diffraction_pattern)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(diffraction_pattern, cmap='hot', extent=[-2.5, 2.5, -2.5, 2.5])
        plt.title(f"Figure de diffraction à {distance_mm} mm\nPV={pv_diff:.2f}, RMS={rms_diff:.2f}")
        plt.colorbar(label="Intensité (a.u.)")
        plt.savefig(f'{output_dir}/example11_microhole_diffraction.png', dpi=150, bbox_inches='tight')
        plt.close('all')
        print(f"  ✓ Image sauvegardée: example11_microhole_diffraction.png")
    else:
        print(f"  ⚠️  Propagation module not available. Skipping diffraction pattern.")
    
    # =========================================================================
    # 5. MATRICE DE MICROPRISMES / MICROPRISM ARRAY
    # =========================================================================
    print("\n--- 5. Matrice de microprismes ---")
    
    # 5.1 Création d'une matrice de microprismes
    print("\n5.1 Matrice de microprismes (4x4)")
    microprism_array = create_microprism_array(
        name="Microprismes",
        pitch_mm=1.0,
        num_elements_x=4,
        num_elements_y=4,
        apex_angle_deg=60.0,  # Angle au sommet
        material_name="BK7",
        wavelength_nm=wavelength_nm,
    )
    
    print(f"  {microprism_array}")
    print(f"  Angle au sommet: {microprism_array.apex_angle_deg}°")
    
    # Visualisation
    microprism_array.visualize(
        grid_size_mm=5.0,
        num_points=256,
        save_path=f'{output_dir}/example11_microprism_array.png'
    )
    print(f"  ✓ Image sauvegardée: example11_microprism_array.png")
    
    # 5.2 Carte de déviation
    print("\n5.2 Carte de déviation")
    deviation_map = microprism_array.get_deviation_map(grid_x, grid_y)
    
    pv_dev = np.max(deviation_map) - np.min(deviation_map)
    rms_dev = np.std(deviation_map)
    mean_dev = np.mean(deviation_map[deviation_map != 0]) if np.any(deviation_map != 0) else 0.0
    
    plt.figure(figsize=(8, 6))
    plt.imshow(deviation_map, cmap='Jet', extent=[-2.5, 2.5, -2.5, 2.5])
    plt.title(f"Carte de déviation\nPV={pv_dev:.2f}°, RMS={rms_dev:.2f}°, Mean={mean_dev:.2f}°")
    plt.colorbar(label="Déviation (°)")
    plt.savefig(f'{output_dir}/example11_microprism_deviation.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  ✓ Image sauvegardée: example11_microprism_deviation.png")
    
    # =========================================================================
    # 6. APPLICATION À UN FAISCEAU / BEAM APPLICATION
    # =========================================================================
    print("\n--- 6. Application à un faisceau ---")
    
    # 6.1 Faisceau incident gaussien
    print("\n6.1 Faisceau incident gaussien")
    beam = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=5.0,
        energy=1.0,
        num_points=256,
    )
    electric_field = beam.generate_electric_field(method="gaussian", sigma_mm=1.0)
    beam.electric_field = electric_field
    beam.intensity = beam.compute_intensity_from_electric_field(electric_field)
    beam.phase = beam.extract_phase_from_electric_field(electric_field)
    
    # Calculer PV et RMS du faisceau initial
    pv_initial, rms_initial = beam.compute_pv_rms(beam.phase)
    print(f"  Faisceau initial: PV={pv_initial:.2f} nm, RMS={rms_initial:.2f} nm")
    
    # Sauvegarder l'intensité initiale avec colormap "hot"
    plt.figure(figsize=(8, 6))
    plt.imshow(beam.intensity, cmap='hot', extent=[-2.5, 2.5, -2.5, 2.5])
    plt.title(f"Faisceau initial - Intensité\nPV={pv_initial:.2f} nm, RMS={rms_initial:.2f} nm")
    plt.colorbar(label="Intensité (a.u.)")
    plt.savefig(f'{output_dir}/example11_initial_beam.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  ✓ Image sauvegardée: example11_initial_beam.png")
    
    # 6.2 Application de la matrice de microlentilles
    print("\n6.2 Application de la matrice de microlentilles")
    beam_after_microlenses = microlens_array_joined.apply_to_beam(beam)
    
    pv_after, rms_after = beam_after_microlenses.compute_pv_rms(beam_after_microlenses.phase)
    print(f"  Faisceau après microlentilles: PV={pv_after:.2f} nm, RMS={rms_after:.2f} nm")
    
    # Sauvegarder l'intensité et la phase
    plt.figure(figsize=(8, 6))
    plt.imshow(beam_after_microlenses.intensity, cmap='hot', extent=[-2.5, 2.5, -2.5, 2.5])
    plt.title(f"Après microlentilles - Intensité\nPV={pv_after:.2f} nm, RMS={rms_after:.2f} nm")
    plt.colorbar(label="Intensité (a.u.)")
    plt.savefig(f'{output_dir}/example11_beam_after_microlenses_intensity.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    plt.figure(figsize=(8, 6))
    plt.imshow(beam_after_microlenses.phase, cmap='Jet', extent=[-2.5, 2.5, -2.5, 2.5])
    plt.title(f"Après microlentilles - Phase\nPV={pv_after:.2f} nm, RMS={rms_after:.2f} nm")
    plt.colorbar(label="Phase (nm)")
    plt.savefig(f'{output_dir}/example11_beam_after_microlenses_phase.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  ✓ Images sauvegardées: example11_beam_after_microlenses_*.png")
    
    # 6.3 Application de la matrice de microtrous
    print("\n6.3 Application de la matrice de microtrous")
    
    # Réinitialiser le faisceau
    beam2 = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=5.0,
        energy=1.0,
        num_points=256,
    )
    electric_field2 = beam2.generate_electric_field(method="gaussian", sigma_mm=1.0)
    beam2.electric_field = electric_field2
    beam2.intensity = beam2.compute_intensity_from_electric_field(electric_field2)
    beam2.phase = beam2.extract_phase_from_electric_field(electric_field2)
    
    beam_after_holes = microhole_array.apply_to_beam(beam2)
    
    pv_holes, rms_holes = beam_after_holes.compute_pv_rms(beam_after_holes.phase)
    print(f"  Faisceau après microtrous: PV={pv_holes:.2f} nm, RMS={rms_holes:.2f} nm")
    
    plt.figure(figsize=(8, 6))
    plt.imshow(beam_after_holes.intensity, cmap='hot', extent=[-2.5, 2.5, -2.5, 2.5])
    plt.title(f"Après microtrous - Intensité\nPV={pv_holes:.2f} nm, RMS={rms_holes:.2f} nm")
    plt.colorbar(label="Intensité (a.u.)")
    plt.savefig(f'{output_dir}/example11_beam_after_holes.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  ✓ Image sauvegardée: example11_beam_after_holes.png")
    
    # 6.4 Application avec affichage automatique
    print("\n6.4 Application avec affichage automatique")
    
    microlens_auto = create_microlens_array(
        name="Microlentilles auto",
        pitch_mm=1.0,
        num_elements_x=3,
        num_elements_y=3,
        focal_length_mm=10.0,
        wavelength_nm=wavelength_nm,
        display=True,
        display_dir=os.path.join(output_dir, 'auto_display'),
    )
    
    beam_auto = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=5.0,
        energy=1.0,
        num_points=256,
    )
    electric_field_auto = beam_auto.generate_electric_field(method="gaussian", sigma_mm=1.0)
    beam_auto.electric_field = electric_field_auto
    beam_auto.intensity = beam_auto.compute_intensity_from_electric_field(electric_field_auto)
    beam_auto.phase = beam_auto.extract_phase_from_electric_field(electric_field_auto)
    
    beam_after_auto = microlens_auto.apply_to_beam(beam_auto)
    print(f"  ✓ Images sauvegardées dans: {microlens_auto.display_dir}")
    
    # =========================================================================
    # 7. MATRICE PERSONNALISÉE (MIXTE) / CUSTOM ARRAY
    # =========================================================================
    print("\n--- 7. Matrice personnalisée (mixte) ---")
    
    # 7.1 Matrice avec microlentille au centre et microtrous autour
    print("\n7.1 Matrice mixte: microlentille au centre, microtrous autour")
    
    custom_array = MicroOpticsArray(
        name="Matrice mixte",
        pitch_mm=1.0,
        num_elements_x=3,
        num_elements_y=3,
        micro_optic_type=MicroOpticType.CUSTOM,
        wavelength_nm=wavelength_nm,
    )
    
    # Remplacer les éléments
    from Optiques import IdealLens, DiffractionHole, OpticSpecifications
    
    for mo in custom_array.micro_optics:
        i, j = mo.index
        x, y = mo.position_mm
        
        if i == 1 and j == 1:  # Centre
            # Microlentille au centre
            specs = OpticSpecifications(
                diameter_mm=custom_array.element_diameter_mm,
                thickness_mm=1.0,
                material_name="Fused_Silica",
                aperture_shape=ApertureShape.CIRCULAR,
            )
            mo.optic = IdealLens(
                name=f"Lentille centrale ({i},{j})",
                focal_length_mm=10.0,
                diameter_mm=custom_array.element_diameter_mm,
                material_name="Fused_Silica",
                specifications=specs,
                position_mm=(x, y, 0.0),
                wavelength_nm=wavelength_nm,
            )
        else:
            # Microtrou autour
            specs = OpticSpecifications(
                diameter_mm=custom_array.element_diameter_mm * 0.8,  # Plus petit
                thickness_mm=0.0,
                material_name="air",
                aperture_shape=ApertureShape.CIRCULAR,
            )
            mo.optic = DiffractionHole(
                name=f"Trou ({i},{j})",
                diameter_mm=custom_array.element_diameter_mm * 0.8,
                material_name="air",
                specifications=specs,
                position_mm=(x, y, 0.0),
                wavelength_nm=wavelength_nm,
            )
    
    # Visualisation
    custom_array.visualize(
        grid_size_mm=4.0,
        num_points=256,
        save_path=f'{output_dir}/example11_custom_array.png'
    )
    print(f"  ✓ Image sauvegardée: example11_custom_array.png")
    
    # 7.2 Application de la matrice mixte
    print("\n7.2 Application de la matrice mixte")
    beam_custom = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=5.0,
        energy=1.0,
        num_points=256,
    )
    electric_field_custom = beam_custom.generate_electric_field(method="gaussian", sigma_mm=1.0)
    beam_custom.electric_field = electric_field_custom
    beam_custom.intensity = beam_custom.compute_intensity_from_electric_field(electric_field_custom)
    beam_custom.phase = beam_custom.extract_phase_from_electric_field(electric_field_custom)
    
    beam_after_custom = custom_array.apply_to_beam(beam_custom)
    
    pv_custom, rms_custom = beam_after_custom.compute_pv_rms(beam_after_custom.phase)
    print(f"  Faisceau après matrice mixte: PV={pv_custom:.2f} nm, RMS={rms_custom:.2f} nm")
    
    plt.figure(figsize=(8, 6))
    plt.imshow(beam_after_custom.phase, cmap='Jet', extent=[-2.5, 2.5, -2.5, 2.5])
    plt.title(f"Après matrice mixte - Phase\nPV={pv_custom:.2f} nm, RMS={rms_custom:.2f} nm")
    plt.colorbar(label="Phase (nm)")
    plt.savefig(f'{output_dir}/example11_beam_after_custom_array.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  ✓ Image sauvegardée: example11_beam_after_custom_array.png")
    
    # =========================================================================
    # 8. EXEMPLE COMPLET: SYSTÈME SHACK-HARTMANN SIMPLIFIÉ
    # =========================================================================
    print("\n--- 8. Exemple complet: Système Shack-Hartmann simplifié ---")
    
    # 8.1 Création d'une matrice de microlentilles pour Shack-Hartmann
    print("\n8.1 Matrice de microlentilles pour Shack-Hartmann")
    shack_hartmann = create_microlens_array(
        name="Shack-Hartmann",
        pitch_mm=0.5,          # Pas de 0.5 mm (bord-à-bord)
        num_elements_x=10,
        num_elements_y=10,
        focal_length_mm=20.0,  # Distance focale de 20 mm
        array_pattern=ArrayPattern.SQUARE,
        material_name="Fused_Silica",
        wavelength_nm=wavelength_nm,
        edge_to_edge_spacing_mm=0.0,  # Éléments joints
    )
    
    print(f"  {shack_hartmann}")
    print(f"  Nombre de microlentilles: {len(shack_hartmann.micro_optics)}")
    print(f"  Largeur totale: {shack_hartmann.total_width_mm:.2f} mm")
    print(f"  Hauteur totale: {shack_hartmann.total_height_mm:.2f} mm")
    
    # 8.2 Ajouter une WFE globale (pour simuler des erreurs du système optique)
    print("\n8.2 Ajout d'une WFE globale (simulation d'erreurs système)")
    shack_hartmann.set_global_wfe(WaveFrontError(
        surface_roughness_nm=1.0,
        parallelism_arcsec=0.5,
        zernike_coefficients={
            (2, 0): 5.0,   # Defocus
            (2, 2): 3.0,   # Astigmatisme
            (3, 1): 2.0,   # Coma
        },
    ))
    
    print(f"  WFE globale appliquée:")
    print(f"    - Rugosité: {shack_hartmann.global_wfe.surface_roughness_nm:.1f} nm RMS")
    print(f"    - Parallélisme: {shack_hartmann.global_wfe.parallelism_arcsec:.1f} arcsec")
    print(f"    - Zernike: {shack_hartmann.global_wfe.zernike_coefficients}")
    
    # 8.3 Appliquer une déformation thermique
    print("\n8.3 Déformation thermique (ΔT = +30 K)")
    shack_hartmann.apply_thermal_deformation(293.15 + 30)
    
    print(f"  Après déformation:")
    print(f"    Nouveau pitch: {shack_hartmann.pitch_mm:.6f} mm")
    print(f"    Nouveau diamètre: {shack_hartmann.element_diameter_mm:.6f} mm")
    
    # 8.4 Visualisation après déformation
    shack_hartmann.visualize(
        grid_size_mm=6.0,
        num_points=256,
        save_path=f'{output_dir}/example11_shack_hartmann_deformed.png'
    )
    print(f"  ✓ Image sauvegardée: example11_shack_hartmann_deformed.png")
    
    # 8.5 Application à un faisceau
    print("\n8.5 Application à un faisceau (Shack-Hartmann)")
    beam_sh = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=6.0,
        energy=1.0,
        num_points=256,
    )
    electric_field_sh = beam_sh.generate_electric_field(method="gaussian", sigma_mm=2.0)
    beam_sh.electric_field = electric_field_sh
    beam_sh.intensity = beam_sh.compute_intensity_from_electric_field(electric_field_sh)
    beam_sh.phase = beam_sh.extract_phase_from_electric_field(electric_field_sh)
    
    beam_after_sh = shack_hartmann.apply_to_beam(beam_sh)
    
    pv_sh, rms_sh = beam_after_sh.compute_pv_rms(beam_after_sh.phase)
    print(f"  Faisceau après Shack-Hartmann: PV={pv_sh:.2f} nm, RMS={rms_sh:.2f} nm")
    
    plt.figure(figsize=(8, 6))
    plt.imshow(beam_after_sh.phase, cmap='Jet', extent=[-3.0, 3.0, -3.0, 3.0])
    plt.title(f"Shack-Hartmann - Phase après matrice\nPV={pv_sh:.2f} nm, RMS={rms_sh:.2f} nm")
    plt.colorbar(label="Phase (nm)")
    plt.savefig(f'{output_dir}/example11_shack_hartmann_beam.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  ✓ Image sauvegardée: example11_shack_hartmann_beam.png")
    
    # 8.6 Calcul de la phase dans le plan focal
    print("\n8.6 Phase dans le plan focal")
    focal_plane_phase = shack_hartmann.get_focal_plane_phase(grid_x, grid_y)
    
    pv_focal, rms_focal = np.max(focal_plane_phase) - np.min(focal_plane_phase), np.std(focal_plane_phase)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(focal_plane_phase, cmap='Jet', extent=[-2.5, 2.5, -2.5, 2.5])
    plt.title(f"Shack-Hartmann - Phase dans le plan focal\nPV={pv_focal:.2f} nm, RMS={rms_focal:.2f} nm")
    plt.colorbar(label="Phase (nm)")
    plt.savefig(f'{output_dir}/example11_shack_hartmann_focal_plane.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  ✓ Image sauvegardée: example11_shack_hartmann_focal_plane.png")
    
    # =========================================================================
    # 9. RÉSUMÉ DES INFORMATIONS D'ESPACEMENT
    # =========================================================================
    print("\n--- 9. Résumé des informations d'espacement ---")
    
    arrays = [
        ("Microlentilles jointes", microlens_array_joined),
        ("Microlentilles espacées", microlens_array_spaced),
        ("Microtrous", microhole_array),
        ("Microprismes", microprism_array),
    ]
    
    for name, array in arrays:
        spacing_info = array.get_spacing_info()
        print(f"\n  {name}:")
        print(f"    Espacement bord-à-bord: {spacing_info['edge_to_edge_mm']:.2f} mm")
        print(f"    Distance centre-à-centre: {spacing_info['center_to_center_mm']:.2f} mm")
        print(f"    Diamètre des éléments: {spacing_info['element_diameter_mm']:.2f} mm")
        print(f"    Type d'espacement: {spacing_info['spacing_type']}")
    
    # =========================================================================
    # 10. FIN DE L'EXEMPLE
    # =========================================================================
    print("\n" + "="*80)
    print("Example 11 terminé avec succès !")
    print(f"Les images ont été sauvegardées dans {output_dir}/")
    print("="*80)
    print("\nRésumé des fonctionnalités démontrées:")
    print("  ✓ Espacement bord-à-bord entre les éléments")
    print("  ✓ Application de WFE globale sur toute la matrice")
    print("  ✓ Dilatation thermique avec effet sur les positions des micro-éléments")
    print("  ✓ Création de matrices de microlentilles, microtrous et microprismes")
    print("  ✓ Intégration avec Optiques.py, Material_Behaviour.py et Beam.py")
    print("  ✓ Visualisation avec colormap 'Jet' pour la phase et 'hot' pour l'intensité")
    print("  ✓ Affichage du PV et du RMS pour chaque image")
    print("="*80)


if __name__ == "__main__":
    os.makedirs('examples/output', exist_ok=True)
    run_microstructure_example()

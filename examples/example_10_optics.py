"""
Example 10: Optics Generation and Application
FR: Exemple d'utilisation du module Optiques.py.
    Démonstration de la génération d'optiques diverses (lentilles, miroirs, beamsplitters),
    de l'application de ces optiques à un faisceau, et de la simulation de systèmes optiques complets.

EN: Example of using Optiques.py module.
    Demonstrates generation of various optics (lenses, mirrors, beamsplitters),
    application of these optics to a beam, and simulation of complete optical systems.

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Optiques import (
    IdealLens, SimpleLens, DoubleLens, Mirror, Beamsplitter, Window, AsphericLens,
    OpticalSystem, WaveFrontError, OpticSpecifications, LensType, MirrorType,
    BeamsplitterType, create_optic, create_lens_from_preset, OpticType
)
from Beam import Beam
from Propagation import Propagation
from Visualization import plot_intensity, plot_phase, plot_beam_map


def run_optics_example():
    """FR: Exécute l'exemple d'optiques."""
    print("\n" + "="*80)
    print("Example 10: Optics Generation and Application")
    print("="*80)
    
    wavelength_nm = 633.0
    beam_diameter_mm = 10.0
    num_points = 256
    
    # =========================================================================
    # 1. Création de différentes optiques / Creating Various Optics
    # =========================================================================
    print("\n--- 1. Création de différentes optiques ---")
    
    # 1.1 Lentille idéale (paraxiale)
    print("\n1.1 Lentille idéale (f = 100 mm)")
    ideal_lens = IdealLens(
        name="Lentille idéale",
        focal_length_mm=100.0,
        diameter_mm=beam_diameter_mm,
        wavelength_nm=wavelength_nm,
    )
    print(f"  Type: {type(ideal_lens).__name__}")
    print(f"  Distance focale: {ideal_lens.focal_length_mm} mm")
    print(f"  Diamètre: {ideal_lens.specifications.diameter_mm} mm")
    
    # 1.2 Lentille simple plan-convexe
    print("\n1.2 Lentille simple plan-convexe (R = 100 mm)")
    simple_lens = SimpleLens(
        name="Lentille plan-convexe",
        radius_of_curvature_mm=100.0,
        diameter_mm=beam_diameter_mm,
        thickness_mm=5.0,
        lens_type=LensType.PLAN_CONVEX,
        material_name="BK7",
        wavelength_nm=wavelength_nm,
    )
    print(f"  Type: {type(simple_lens).__name__}")
    print(f"  Rayon de courbure: {simple_lens.radius_of_curvature_mm} mm")
    print(f"  Matériau: {simple_lens.specifications.material_name}")
    print(f"  Indice de réfraction: {simple_lens.material.get_refractive_index(wavelength_nm):.4f}")
    
    # 1.3 Lentille double biconvexe
    print("\n1.3 Lentille double biconvexe (R1 = 100 mm, R2 = -100 mm)")
    double_lens = DoubleLens(
        name="Lentille biconvexe",
        radius_of_curvature_1_mm=100.0,
        radius_of_curvature_2_mm=-100.0,
        diameter_mm=beam_diameter_mm,
        thickness_mm=5.0,
        lens_type=LensType.BICONVEX,
        material_name="BK7",
        wavelength_nm=wavelength_nm,
    )
    print(f"  Type: {type(double_lens).__name__}")
    print(f"  Rayons de courbure: R1={double_lens.radius_of_curvature_1_mm} mm, R2={double_lens.radius_of_curvature_2_mm} mm")
    
    # 1.4 Miroir sphérique
    print("\n1.4 Miroir sphérique (R = 200 mm)")
    spherical_mirror = Mirror(
        name="Miroir sphérique",
        diameter_mm=beam_diameter_mm,
        mirror_type=MirrorType.SPHERICAL,
        radius_of_curvature_mm=200.0,  # f = 100 mm
        material_name="Aluminum",
        coating_reflectivity=0.95,
        wavelength_nm=wavelength_nm,
    )
    print(f"  Type: {type(spherical_mirror).__name__}")
    print(f"  Rayon de courbure: {spherical_mirror.radius_of_curvature_mm} mm")
    print(f"  Distance focale: {spherical_mirror.focal_length_mm} mm")
    print(f"  Réflectivité: {spherical_mirror.coating_reflectivity:.2f}")
    
    # 1.5 Séparateur de faisceau
    print("\n1.5 Séparateur de faisceau (50/50)")
    beamsplitter = Beamsplitter(
        name="Beamsplitter",
        diameter_mm=beam_diameter_mm,
        beamsplitter_type=BeamsplitterType.PLATE,
        transmission_ratio=0.5,
        reflection_ratio=0.5,
        material_name="BK7",
        wavelength_nm=wavelength_nm,
    )
    print(f"  Type: {type(beamsplitter).__name__}")
    print(f"  Transmission: {beamsplitter.transmission_ratio:.2f}")
    print(f"  Réflexion: {beamsplitter.reflection_ratio:.2f}")
    
    # 1.6 Fenêtre optique
    print("\n1.6 Fenêtre optique (Fused Silica, e = 2 mm)")
    window = Window(
        name="Fenêtre",
        diameter_mm=beam_diameter_mm,
        thickness_mm=2.0,
        material_name="Fused_Silica",
        wavelength_nm=wavelength_nm,
    )
    print(f"  Type: {type(window).__name__}")
    print(f"  Épaisseur: {window.specifications.thickness_mm} mm")
    print(f"  Indice de réfraction: {window.material.get_refractive_index(wavelength_nm):.4f}")
    
    # 1.7 Lentille asphérique
    print("\n1.7 Lentille asphérique (R = 50 mm, k = -0.5)")
    aspheric_lens = AsphericLens(
        name="Lentille asphérique",
        radius_of_curvature_mm=50.0,
        diameter_mm=beam_diameter_mm,
        thickness_mm=5.0,
        conic_constant=-0.5,
        aspheric_coefficients={4: 1e-6, 6: -1e-9},
        material_name="Fused_Silica",
        wavelength_nm=wavelength_nm,
    )
    print(f"  Type: {type(aspheric_lens).__name__}")
    print(f"  Rayon de courbure: {aspheric_lens.radius_of_curvature_mm} mm")
    print(f"  Constante conique: {aspheric_lens.conic_constant}")
    
    # =========================================================================
    # 2. Visualisation des cartes de phase / Phase Maps Visualization
    # =========================================================================
    print("\n--- 2. Visualisation des cartes de phase ---")
    
    # Créer une grille pour la visualisation
    grid_x, grid_y = np.meshgrid(
        np.linspace(-beam_diameter_mm/2, beam_diameter_mm/2, num_points),
        np.linspace(-beam_diameter_mm/2, beam_diameter_mm/2, num_points)
    )
    
    optics_to_plot = [
        ("Lentille idéale", ideal_lens),
        ("Lentille plan-convexe", simple_lens),
        ("Lentille biconvexe", double_lens),
        ("Miroir sphérique", spherical_mirror),
        ("Fenêtre", window),
        ("Lentille asphérique", aspheric_lens),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes_flat = axes.flatten()
    
    for idx, (title, optic) in enumerate(optics_to_plot):
        phase_map = optic.get_phase_map(grid_x, grid_y)
        
        im = axes_flat[idx].imshow(
            phase_map,
            extent=[-beam_diameter_mm/2, beam_diameter_mm/2, -beam_diameter_mm/2, beam_diameter_mm/2],
            cmap='coolwarm'
        )
        axes_flat[idx].set_title(title)
        axes_flat[idx].set_xlabel("x (mm)")
        axes_flat[idx].set_ylabel("y (mm)")
        plt.colorbar(im, ax=axes_flat[idx], label="Phase (nm)")
    
    plt.tight_layout()
    plt.savefig('examples/output/example10_optics_phase_maps.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # =========================================================================
    # 3. Erreurs de front d'onde (WFE) / WaveFront Errors (WFE)
    # =========================================================================
    print("\n--- 3. Erreurs de front d'onde (WFE) ---")
    
    # 3.1 Rugosité de surface
    print("\n3.1 Rugosité de surface (RMS = 10 nm)")
    wfe_roughness = WaveFrontError(
        surface_roughness_nm=10.0,
        seed=42,
    )
    
    # 3.2 Parallélisme
    print("\n3.2 Parallélisme (10 secondes d'arc)")
    wfe_parallelism = WaveFrontError(
        parallelism_arcsec=10.0,
    )
    
    # 3.3 Aberrations (Zernike)
    print("\n3.3 Aberrations (Defocus = 50 nm RMS, Astigmatisme = 30 nm RMS)")
    wfe_zernike = WaveFrontError(
        zernike_coefficients={
            (2, 0): 50.0,   # Defocus
            (2, 2): 30.0,   # Astigmatisme
        }
    )
    
    # 3.4 Combinaison
    print("\n3.4 Combinaison (rugosité + parallélisme + aberrations)")
    wfe_combined = WaveFrontError(
        surface_roughness_nm=5.0,
        parallelism_arcsec=5.0,
        zernike_coefficients={(2, 0): 25.0},
        seed=42,
    )
    
    # Visualisation des WFE
    wfe_list = [
        ("Rugosité de surface", wfe_roughness),
        ("Parallélisme", wfe_parallelism),
        ("Aberrations Zernike", wfe_zernike),
        ("Combinaison", wfe_combined),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes_flat = axes.flatten()
    
    for idx, (title, wfe) in enumerate(wfe_list):
        phase_map = wfe.generate_phase_map(grid_x, grid_y, wavelength_nm, seed=42)
        
        im = axes_flat[idx].imshow(
            phase_map,
            extent=[-beam_diameter_mm/2, beam_diameter_mm/2, -beam_diameter_mm/2, beam_diameter_mm/2],
            cmap='coolwarm'
        )
        axes_flat[idx].set_title(title)
        axes_flat[idx].set_xlabel("x (mm)")
        axes_flat[idx].set_ylabel("y (mm)")
        plt.colorbar(im, ax=axes_flat[idx], label="Phase (nm)")
        
        # Calculer PV et RMS
        pv = np.max(phase_map) - np.min(phase_map)
        rms = np.std(phase_map)
        axes_flat[idx].text(
            0.05, 0.95, f"PV={pv:.1f} nm, RMS={rms:.1f} nm",
            transform=axes_flat[idx].transAxes,
            ha='left', va='top', bbox=dict(facecolor='white', alpha=0.8)
        )
    
    plt.tight_layout()
    plt.savefig('examples/output/example10_wfe_maps.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # =========================================================================
    # 4. Application des optiques à un faisceau / Applying Optics to a Beam
    # =========================================================================
    print("\n--- 4. Application des optiques à un faisceau ---")
    
    # 4.1 Créer un faisceau gaussien
    print("\n4.1 Faisceau gaussien initial")
    beam = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=beam_diameter_mm,
        energy=1.0,
        num_points=num_points,
    )
    electric_field = beam.generate_electric_field(method="gaussian", sigma_mm=2.0)
    beam.electric_field = electric_field
    beam.intensity = beam.compute_intensity_from_electric_field(electric_field)
    beam.phase = beam.extract_phase_from_electric_field(electric_field)
    
    pv_initial, rms_initial = beam.compute_pv_rms(beam.phase)
    print(f"  Faisceau initial: PV={pv_initial:.2f} nm, RMS={rms_initial:.2f} nm")
    
    # Visualisation du faisceau initial
    plot_intensity(beam.intensity, beam_diameter_mm, title="Faisceau initial - Intensité")
    plot_phase(beam.phase, beam_diameter_mm, title="Faisceau initial - Phase")
    plt.savefig('examples/output/example10_initial_beam.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # 4.2 Appliquer une lentille idéale
    print("\n4.2 Application d'une lentille idéale (f = 100 mm)")
    beam_after_lens = ideal_lens.apply_to_beam(beam)
    
    pv_lens, rms_lens = beam_after_lens.compute_pv_rms(beam_after_lens.phase)
    print(f"  Faisceau après lentille: PV={pv_lens:.2f} nm, RMS={rms_lens:.2f} nm")
    
    # Visualisation
    plot_phase(beam_after_lens.phase, beam_diameter_mm, title="Phase après lentille idéale")
    plt.savefig('examples/output/example10_beam_after_ideal_lens.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # 4.3 Appliquer une lentille avec WFE
    print("\n4.3 Application d'une lentille avec erreurs de front d'onde")
    
    # Créer une lentille avec WFE
    lens_with_wfe = IdealLens(
        name="Lentille avec WFE",
        focal_length_mm=100.0,
        diameter_mm=beam_diameter_mm,
        wavelength_nm=wavelength_nm,
        wfe=WaveFrontError(
            surface_roughness_nm=5.0,
            parallelism_arcsec=2.0,
            zernike_coefficients={(2, 0): 20.0},
            seed=42,
        ),
    )
    
    beam_after_wfe_lens = lens_with_wfe.apply_to_beam(beam)
    
    pv_wfe, rms_wfe = beam_after_wfe_lens.compute_pv_rms(beam_after_wfe_lens.phase)
    print(f"  Faisceau après lentille avec WFE: PV={pv_wfe:.2f} nm, RMS={rms_wfe:.2f} nm")
    
    # Visualisation
    plot_phase(beam_after_wfe_lens.phase, beam_diameter_mm, title="Phase après lentille avec WFE")
    plt.savefig('examples/output/example10_beam_after_wfe_lens.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # =========================================================================
    # 5. Tilt et décentrement / Tilt and Decentering
    # =========================================================================
    print("\n--- 5. Tilt et décentrement ---")
    
    # 5.1 Lentille avec tilt
    print("\n5.1 Lentille avec tilt (5 degrés)")
    tilted_lens = IdealLens(
        name="Lentille inclinée",
        focal_length_mm=100.0,
        diameter_mm=beam_diameter_mm,
        wavelength_nm=wavelength_nm,
        tilt_deg=(5.0, 0.0),  # Inclinaison de 5° autour de l'axe x
    )
    
    # Calculer la phase avec tilt
    phase_tilted = tilted_lens.get_full_phase_map(grid_x, grid_y)
    
    # Visualisation
    plot_phase(phase_tilted, beam_diameter_mm, title="Phase avec tilt (5°)")
    plt.savefig('examples/output/example10_phase_with_tilt.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # 5.2 Lentille avec décentrement
    print("\n5.2 Lentille avec décentrement (2 mm)")
    decentered_lens = IdealLens(
        name="Lentille décentrée",
        focal_length_mm=100.0,
        diameter_mm=beam_diameter_mm,
        wavelength_nm=wavelength_nm,
        decentering_mm=(2.0, 0.0),  # Décentrement de 2 mm en x
    )
    
    # Calculer le masque d'aperture
    mask = decentered_lens.get_aperture_mask(grid_x, grid_y)
    
    # Visualisation du masque
    plot_beam_map(
        mask,
        beam_diameter_mm,
        title="Masque d'aperture avec décentrement",
        label="Masque",
        cmap="gray"
    )
    plt.savefig('examples/output/example10_aperture_mask_decentered.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # 5.3 Lentille avec tilt et décentrement
    print("\n5.3 Lentille avec tilt et décentrement")
    tilted_decentered_lens = IdealLens(
        name="Lentille inclinée et décentrée",
        focal_length_mm=100.0,
        diameter_mm=beam_diameter_mm,
        wavelength_nm=wavelength_nm,
        tilt_deg=(3.0, 2.0),
        decentering_mm=(1.0, -1.0),
    )
    
    phase_combined = tilted_decentered_lens.get_full_phase_map(grid_x, grid_y)
    mask_combined = tilted_decentered_lens.get_aperture_mask(grid_x, grid_y)
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = axes[0].imshow(
        phase_combined,
        extent=[-beam_diameter_mm/2, beam_diameter_mm/2, -beam_diameter_mm/2, beam_diameter_mm/2],
        cmap='coolwarm'
    )
    axes[0].set_title("Phase avec tilt et décentrement")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    plt.colorbar(im1, ax=axes[0], label="Phase (nm)")
    
    im2 = axes[1].imshow(
        mask_combined,
        extent=[-beam_diameter_mm/2, beam_diameter_mm/2, -beam_diameter_mm/2, beam_diameter_mm/2],
        cmap='gray'
    )
    axes[1].set_title("Masque d'aperture")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("y (mm)")
    plt.colorbar(im2, ax=axes[1], label="Masque")
    
    plt.tight_layout()
    plt.savefig('examples/output/example10_tilt_decenter_combined.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # =========================================================================
    # 6. Système optique complet / Complete Optical System
    # =========================================================================
    print("\n--- 6. Système optique complet ---")
    
    # 6.1 Créer un système optique (simulation d'un Shack-Hartmann simplifié)
    print("\n6.1 Système optique : Lentille + Miroir + Fenêtre")
    
    system = OpticalSystem()
    
    # Ajouter une lentille à z = 50 mm
    lens1 = system.add_lens(
        name="Lentille 1",
        focal_length_mm=100.0,
        diameter_mm=beam_diameter_mm,
        material_name="BK7",
        position_z_mm=50.0,
    )
    
    # Ajouter un miroir à z = 150 mm (incline à 45°)
    mirror1 = system.add_mirror(
        name="Miroir",
        diameter_mm=beam_diameter_mm,
        mirror_type=MirrorType.FLAT,
        position_z_mm=150.0,
        tilt_deg=(45.0, 0.0),
    )
    
    # Ajouter une fenêtre à z = 200 mm
    window1 = system.add_element(
        Window(
            name="Fenêtre",
            diameter_mm=beam_diameter_mm,
            thickness_mm=2.0,
            material_name="Fused_Silica",
            position_mm=(0.0, 0.0, 200.0),
            wavelength_nm=wavelength_nm,
        )
    )
    
    # Trier les éléments par position
    system.sort_elements_by_position()
    
    print(f"  Nombre d'éléments: {len(system.elements)}")
    for element in system.elements:
        print(f"    - {element.name} à z = {element.position_mm[2]:.1f} mm")
    
    # 6.2 Propager un faisceau à travers le système
    print("\n6.2 Propagation d'un faisceau à travers le système")
    
    # Créer un nouveau faisceau
    beam_system = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=beam_diameter_mm,
        energy=1.0,
        num_points=num_points,
    )
    electric_field_system = beam_system.generate_electric_field(method="gaussian", sigma_mm=2.0)
    beam_system.electric_field = electric_field_system
    beam_system.intensity = beam_system.compute_intensity_from_electric_field(electric_field_system)
    beam_system.phase = beam_system.extract_phase_from_electric_field(electric_field_system)
    
    # Propager le faisceau à travers le système (sans propagation entre les éléments pour l'instant)
    final_beam = system.propagate_beam(beam_system, initial_position_mm=0.0, use_propagation=False)
    
    pv_final, rms_final = final_beam.compute_pv_rms(final_beam.phase)
    print(f"  Faisceau final: PV={pv_final:.2f} nm, RMS={rms_final:.2f} nm")
    
    # Visualisation de la phase finale
    plot_phase(final_beam.phase, beam_diameter_mm, title="Phase finale après le système optique")
    plt.savefig('examples/output/example10_final_beam_after_system.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # 6.3 Système avec propagation
    print("\n6.3 Système avec propagation entre les éléments")
    
    # Créer un système plus simple pour la propagation
    simple_system = OpticalSystem()
    
    # Ajouter une lentille à z = 100 mm
    simple_system.add_lens(
        name="Lentille",
        focal_length_mm=100.0,
        diameter_mm=beam_diameter_mm,
        position_z_mm=100.0,
    )
    
    # Créer un nouveau faisceau
    beam_prop = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=beam_diameter_mm,
        energy=1.0,
        num_points=128,  # Réduire pour accélérer la propagation
    )
    electric_field_prop = beam_prop.generate_electric_field(method="gaussian", sigma_mm=2.0)
    beam_prop.electric_field = electric_field_prop
    beam_prop.intensity = beam_prop.compute_intensity_from_electric_field(electric_field_prop)
    beam_prop.phase = beam_prop.extract_phase_from_electric_field(electric_field_prop)
    
    # Propager le faisceau à travers le système (avec propagation)
    try:
        final_beam_prop = simple_system.propagate_beam(beam_prop, initial_position_mm=0.0, use_propagation=True)
        
        pv_prop, rms_prop = final_beam_prop.compute_pv_rms(final_beam_prop.phase)
        print(f"  Faisceau final (avec propagation): PV={pv_prop:.2f} nm, RMS={rms_prop:.2f} nm")
        
        # Visualisation
        plot_phase(final_beam_prop.phase, beam_diameter_mm, title="Phase après propagation et lentille")
        plt.savefig('examples/output/example10_beam_with_propagation.png', dpi=150, bbox_inches='tight')
        plt.close('all')
    except Exception as e:
        print(f"  ⚠️  Propagation échouée: {e}")
    
    # =========================================================================
    # 7. Comparaison des matériaux / Material Comparison
    # =========================================================================
    print("\n--- 7. Comparaison des matériaux ---")
    
    materials = ["Fused_Silica", "BK7", "SF5"]
    focal_length_mm = 100.0
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, material_name in enumerate(materials):
        # Créer une lentille idéale avec le matériau
        lens = IdealLens(
            name=f"Lentille {material_name}",
            focal_length_mm=focal_length_mm,
            diameter_mm=beam_diameter_mm,
            material_name=material_name,
            wavelength_nm=wavelength_nm,
        )
        
        # Calculer la phase
        phase_map = lens.get_phase_map(grid_x, grid_y)
        
        # Afficher
        im = axes[idx].imshow(
            phase_map,
            extent=[-beam_diameter_mm/2, beam_diameter_mm/2, -beam_diameter_mm/2, beam_diameter_mm/2],
            cmap='coolwarm'
        )
        axes[idx].set_title(f"{material_name} (n={lens.material.get_refractive_index(wavelength_nm):.4f})")
        axes[idx].set_xlabel("x (mm)")
        axes[idx].set_ylabel("y (mm)")
        plt.colorbar(im, ax=axes[idx], label="Phase (nm)")
        
        # Calculer PV et RMS
        pv, rms = lens.compute_pv_rms(phase_map)
        if hasattr(lens, 'compute_pv_rms'):
            pv, rms = lens.compute_pv_rms(phase_map)
        else:
            pv = np.max(phase_map) - np.min(phase_map)
            rms = np.std(phase_map)
        axes[idx].text(
            0.05, 0.95, f"PV={pv:.1f} nm, RMS={rms:.1f} nm",
            transform=axes[idx].transAxes,
            ha='left', va='top', bbox=dict(facecolor='white', alpha=0.8)
        )
    
    plt.tight_layout()
    plt.savefig('examples/output/example10_material_comparison.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # =========================================================================
    # 8. Génération de lentilles à partir de configurations prédéfinies
    # =========================================================================
    print("\n--- 8. Génération de lentilles à partir de configurations prédéfinies ---")
    
    presets = ["ideal", "plan_convex_standard", "biconvex_standard"]
    
    for preset_name in presets:
        print(f"\n  Configuration: {preset_name}")
        lens = create_lens_from_preset(
            preset_name=preset_name,
            diameter_mm=beam_diameter_mm,
            focal_length_mm=100.0,
            material_name="BK7",
        )
        print(f"    Type: {type(lens).__name__}")
        print(f"    Nom: {lens.name}")
        
        if hasattr(lens, 'focal_length_mm'):
            print(f"    Distance focale: {lens.focal_length_mm} mm")
        if hasattr(lens, 'radius_of_curvature_mm'):
            print(f"    Rayon de courbure: {lens.radius_of_curvature_mm} mm")
        if hasattr(lens, 'radius_of_curvature_1_mm'):
            print(f"    Rayons de courbure: R1={lens.radius_of_curvature_1_mm} mm, R2={lens.radius_of_curvature_2_mm} mm")
    
    # =========================================================================
    # 9. Masquage des zones non couvertes / Masking Uncovered Areas
    # =========================================================================
    print("\n--- 9. Masquage des zones non couvertes ---")
    
    # 9.1 Lentille plus petite que le faisceau
    print("\n9.1 Lentille plus petite que le faisceau")
    small_lens = IdealLens(
        name="Petite lentille",
        focal_length_mm=100.0,
        diameter_mm=5.0,  # Plus petite que le faisceau (10 mm)
        wavelength_nm=wavelength_nm,
    )
    
    # Calculer le masque
    mask_small = small_lens.get_aperture_mask(grid_x, grid_y)
    
    # Calculer la phase (sera nulle à l'extérieur du masque)
    phase_small = small_lens.get_full_phase_map(grid_x, grid_y)
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = axes[0].imshow(
        mask_small,
        extent=[-beam_diameter_mm/2, beam_diameter_mm/2, -beam_diameter_mm/2, beam_diameter_mm/2],
        cmap='gray'
    )
    axes[0].set_title("Masque d'aperture (lentille 5 mm)")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    plt.colorbar(im1, ax=axes[0], label="Masque")
    
    im2 = axes[1].imshow(
        phase_small,
        extent=[-beam_diameter_mm/2, beam_diameter_mm/2, -beam_diameter_mm/2, beam_diameter_mm/2],
        cmap='coolwarm'
    )
    axes[1].set_title("Phase (masquée)")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("y (mm)")
    plt.colorbar(im2, ax=axes[1], label="Phase (nm)")
    
    plt.tight_layout()
    plt.savefig('examples/output/example10_masking_uncovered_areas.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # 9.2 Application à un faisceau
    print("\n9.2 Application à un faisceau (lentille plus petite)")
    
    beam_small = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=beam_diameter_mm,
        energy=1.0,
        num_points=num_points,
    )
    electric_field_small = beam_small.generate_electric_field(method="gaussian", sigma_mm=2.0)
    beam_small.electric_field = electric_field_small
    beam_small.intensity = beam_small.compute_intensity_from_electric_field(electric_field_small)
    beam_small.phase = beam_small.extract_phase_from_electric_field(electric_field_small)
    
    # Appliquer la petite lentille
    beam_after_small_lens = small_lens.apply_to_beam(beam_small)
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = axes[0].imshow(
        beam_small.intensity,
        extent=[-beam_diameter_mm/2, beam_diameter_mm/2, -beam_diameter_mm/2, beam_diameter_mm/2],
        cmap='viridis'
    )
    axes[0].set_title("Intensité initiale")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    plt.colorbar(im1, ax=axes[0], label="Intensité (a.u.)")
    
    im2 = axes[1].imshow(
        beam_after_small_lens.intensity,
        extent=[-beam_diameter_mm/2, beam_diameter_mm/2, -beam_diameter_mm/2, beam_diameter_mm/2],
        cmap='viridis'
    )
    axes[1].set_title("Intensité après petite lentille")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("y (mm)")
    plt.colorbar(im2, ax=axes[1], label="Intensité (a.u.)")
    
    plt.tight_layout()
    plt.savefig('examples/output/example10_beam_after_small_lens.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # =========================================================================
    # 10. Exemple complet : Système Shack-Hartmann simplifié
    # =========================================================================
    print("\n--- 10. Exemple complet : Système Shack-Hartmann simplifié ---")
    
    # Créer un système Shack-Hartmann
    sh_system = OpticalSystem()
    
    # Matrice de microlentilles (simulée par une seule lentille pour l'exemple)
    sh_system.add_lens(
        name="Matrice de microlentilles",
        focal_length_mm=10.0,
        diameter_mm=beam_diameter_mm,
        material_name="Fused_Silica",
        position_z_mm=50.0,
    )
    
    # Lentille de champ
    sh_system.add_lens(
        name="Lentille de champ",
        focal_length_mm=50.0,
        diameter_mm=beam_diameter_mm,
        material_name="BK7",
        position_z_mm=100.0,
    )
    
    # Capteur (simulé par une fenêtre)
    sh_system.add_element(
        Window(
            name="Capteur",
            diameter_mm=beam_diameter_mm,
            thickness_mm=0.5,
            material_name="Silicon",
            position_mm=(0.0, 0.0, 150.0),
            wavelength_nm=wavelength_nm,
        )
    )
    
    print(f"\nSystème Shack-Hartmann : {len(sh_system.elements)} éléments")
    for element in sh_system.elements:
        print(f"  - {element.name} à z = {element.position_mm[2]:.1f} mm")
    
    # Propager un faisceau à travers le système
    sh_beam = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=beam_diameter_mm,
        energy=1.0,
        num_points=128,
    )
    electric_field_sh = sh_beam.generate_electric_field(method="gaussian", sigma_mm=2.0)
    sh_beam.electric_field = electric_field_sh
    sh_beam.intensity = sh_beam.compute_intensity_from_electric_field(electric_field_sh)
    sh_beam.phase = sh_beam.extract_phase_from_electric_field(electric_field_sh)
    
    try:
        sh_final_beam = sh_system.propagate_beam(sh_beam, initial_position_mm=0.0, use_propagation=True)
        
        pv_sh, rms_sh = sh_final_beam.compute_pv_rms(sh_final_beam.phase)
        print(f"\nFaisceau final Shack-Hartmann: PV={pv_sh:.2f} nm, RMS={rms_sh:.2f} nm")
        
        # Visualisation
        plot_intensity(sh_final_beam.intensity, beam_diameter_mm, title="Intensité finale (Shack-Hartmann)")
        plot_phase(sh_final_beam.phase, beam_diameter_mm, title="Phase finale (Shack-Hartmann)")
        plt.savefig('examples/output/example10_shack_hartmann_system.png', dpi=150, bbox_inches='tight')
        plt.close('all')
    except Exception as e:
        print(f"  ⚠️  Propagation Shack-Hartmann échouée: {e}")
    
    print("\n" + "="*80)
    print("Example 10 terminé avec succès !")
    print("Les images ont été sauvegardées dans examples/output/")
    print("="*80)


if __name__ == "__main__":
    import os
    os.makedirs('examples/output', exist_ok=True)
    run_optics_example()

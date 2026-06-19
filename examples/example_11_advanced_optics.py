"""
Example 11: Advanced Optics - Doublets, Prisms, and Various Shapes
FR: Exemple d'utilisation avancée du module Optiques.py.
    Démonstration des doublets collés, prismes, formes géométriques variées,
    et affichage automatique des cartes d'intensité et de phase.

EN: Example of advanced use of Optiques.py module.
    Demonstrates cemented doublets, prisms, various geometric shapes,
    and automatic display of intensity and phase maps.

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Optiques import (
    IdealLens, SimpleLens, DoubleLens, CementedDoublet, Mirror, Beamsplitter, Window, Prism,
    OpticalSystem, WaveFrontError, OpticSpecifications, LensType, MirrorType,
    BeamsplitterType, PrismType, ShapeType, create_optic, create_lens_from_preset,
    create_doublet_from_preset, OpticType
)
from Beam import Beam
from Propagation import Propagation
from Visualization import plot_intensity, plot_phase, plot_beam_map


def run_advanced_optics_example():
    """FR: Exécute l'exemple d'optiques avancées."""
    print("\n" + "="*80)
    print("Example 11: Advanced Optics - Doublets, Prisms, and Various Shapes")
    print("="*80)
    
    wavelength_nm = 633.0
    beam_diameter_mm = 10.0
    num_points = 256
    
    # =========================================================================
    # 1. Doublets collés / Cemented Doublets
    # =========================================================================
    print("\n--- 1. Doublets collés ---")
    
    # 1.1 Doublet achromatique standard
    print("\n1.1 Doublet achromatique standard (BK7 + SF5)")
    doublet = create_doublet_from_preset(
        preset_name="achromatic_standard",
        diameter_mm=beam_diameter_mm,
        focal_length_mm=100.0,
        material1_name="BK7",
        material2_name="SF5",
    )
    print(f"  Type: {type(doublet).__name__}")
    print(f"  Lentille 1: {doublet.lens1.name} ({doublet.lens1.material_name})")
    print(f"  Lentille 2: {doublet.lens2.name} ({doublet.lens2.material_name})")
    print(f"  Rayon d'interface: {doublet.interface_radius_mm} mm")
    
    # Calculer la phase
    grid_x, grid_y = np.meshgrid(
        np.linspace(-beam_diameter_mm/2, beam_diameter_mm/2, num_points),
        np.linspace(-beam_diameter_mm/2, beam_diameter_mm/2, num_points)
    )
    phase_doublet = doublet.get_phase_map(grid_x, grid_y)
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = axes[0].imshow(
        phase_doublet,
        extent=[-beam_diameter_mm/2, beam_diameter_mm/2, -beam_diameter_mm/2, beam_diameter_mm/2],
        cmap='coolwarm'
    )
    axes[0].set_title("Phase du doublet achromatique")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    plt.colorbar(im1, ax=axes[0], label="Phase (nm)")
    
    # Masque d'aperture
    mask_doublet = doublet.get_aperture_mask(grid_x, grid_y)
    im2 = axes[1].imshow(
        mask_doublet,
        extent=[-beam_diameter_mm/2, beam_diameter_mm/2, -beam_diameter_mm/2, beam_diameter_mm/2],
        cmap='gray'
    )
    axes[1].set_title("Masque d'aperture")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("y (mm)")
    plt.colorbar(im2, ax=axes[1], label="Masque")
    
    plt.tight_layout()
    plt.savefig('examples/output/example11_doublet_phase.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # 1.2 Doublet symétrique
    print("\n1.2 Doublet symétrique (biconvexe + biconcave)")
    doublet_sym = create_doublet_from_preset(
        preset_name="achromatic_symmetrical",
        diameter_mm=beam_diameter_mm,
        focal_length_mm=100.0,
        material1_name="BK7",
        material2_name="SF5",
    )
    print(f"  Type: {type(doublet_sym).__name__}")
    print(f"  Lentille 1: {doublet_sym.lens1.name} ({doublet_sym.lens1.material_name})")
    print(f"  Lentille 2: {doublet_sym.lens2.name} ({doublet_sym.lens2.material_name})")
    
    # =========================================================================
    # 2. Prismes / Prisms
    # =========================================================================
    print("\n--- 2. Prismes ---")
    
    # 2.1 Prisme à angle droit
    print("\n2.1 Prisme à angle droit (60°)")
    prism_right = Prism(
        name="Prisme à angle droit",
        diameter_mm=beam_diameter_mm,
        prism_type=PrismType.RIGHT_ANGLE,
        apex_angle_deg=60.0,
        base_length_mm=beam_diameter_mm,
        height_mm=beam_diameter_mm,
        material_name="BK7",
        wavelength_nm=wavelength_nm,
    )
    print(f"  Type: {type(prism_right).__name__}")
    print(f"  Angle au sommet: {prism_right.apex_angle_deg}°")
    print(f"  Matériau: {prism_right.material_name}")
    
    phase_prism_right = prism_right.get_phase_map(grid_x, grid_y)
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = axes[0].imshow(
        phase_prism_right,
        extent=[-beam_diameter_mm/2, beam_diameter_mm/2, -beam_diameter_mm/2, beam_diameter_mm/2],
        cmap='coolwarm'
    )
    axes[0].set_title("Phase du prisme à angle droit")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    plt.colorbar(im1, ax=axes[0], label="Phase (nm)")
    
    mask_prism_right = prism_right.get_aperture_mask(grid_x, grid_y)
    im2 = axes[1].imshow(
        mask_prism_right,
        extent=[-beam_diameter_mm/2, beam_diameter_mm/2, -beam_diameter_mm/2, beam_diameter_mm/2],
        cmap='gray'
    )
    axes[1].set_title("Masque d'aperture (triangulaire)")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("y (mm)")
    plt.colorbar(im2, ax=axes[1], label="Masque")
    
    plt.tight_layout()
    plt.savefig('examples/output/example11_prism_right_angle.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # 2.2 Prisme équilatéral
    print("\n2.2 Prisme équilatéral (60°)")
    prism_equilateral = Prism(
        name="Prisme équilatéral",
        diameter_mm=beam_diameter_mm,
        prism_type=PrismType.EQUILATERAL,
        apex_angle_deg=60.0,
        base_length_mm=beam_diameter_mm,
        height_mm=beam_diameter_mm,
        material_name="Fused_Silica",
        wavelength_nm=wavelength_nm,
    )
    print(f"  Type: {type(prism_equilateral).__name__}")
    print(f"  Angle au sommet: {prism_equilateral.apex_angle_deg}°")
    print(f"  Matériau: {prism_equilateral.material_name}")
    
    phase_prism_equilateral = prism_equilateral.get_phase_map(grid_x, grid_y)
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = axes[0].imshow(
        phase_prism_equilateral,
        extent=[-beam_diameter_mm/2, beam_diameter_mm/2, -beam_diameter_mm/2, beam_diameter_mm/2],
        cmap='coolwarm'
    )
    axes[0].set_title("Phase du prisme équilatéral")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    plt.colorbar(im1, ax=axes[0], label="Phase (nm)")
    
    mask_prism_equilateral = prism_equilateral.get_aperture_mask(grid_x, grid_y)
    im2 = axes[1].imshow(
        mask_prism_equilateral,
        extent=[-beam_diameter_mm/2, beam_diameter_mm/2, -beam_diameter_mm/2, beam_diameter_mm/2],
        cmap='gray'
    )
    axes[1].set_title("Masque d'aperture (triangulaire)")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("y (mm)")
    plt.colorbar(im2, ax=axes[1], label="Masque")
    
    plt.tight_layout()
    plt.savefig('examples/output/example11_prism_equilateral.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # =========================================================================
    # 3. Facteurs de forme différents / Various Shapes
    # =========================================================================
    print("\n--- 3. Facteurs de forme différents ---")
    
    shapes = [
        (ShapeType.CIRCULAR, {}, "Circulaire"),
        (ShapeType.RECTANGULAR, {"width_mm": 12.0, "height_mm": 6.0}, "Rectangulaire 12×6 mm"),
        (ShapeType.SQUARE, {"side_mm": 8.0}, "Carré 8×8 mm"),
        (ShapeType.ELLIPTICAL, {"semi_major_axis_mm": 6.0, "semi_minor_axis_mm": 4.0}, "Elliptique 12×8 mm"),
        (ShapeType.HEXAGONAL, {"radius_mm": 5.0}, "Hexagonal R=5 mm"),
        (ShapeType.OCTAGONAL, {"radius_mm": 5.0}, "Octagonal R=5 mm"),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes_flat = axes.flatten()
    
    for idx, (shape_type, dimensions, title) in enumerate(shapes):
        lens = IdealLens(
            name=f"Lens {shape_type.value}",
            focal_length_mm=100.0,
            diameter_mm=beam_diameter_mm,
            shape_type=shape_type,
            shape_dimensions=dimensions,
        )
        
        mask = lens.get_aperture_mask(grid_x, grid_y)
        
        im = axes_flat[idx].imshow(
            mask,
            extent=[-beam_diameter_mm/2, beam_diameter_mm/2, -beam_diameter_mm/2, beam_diameter_mm/2],
            cmap='gray'
        )
        axes_flat[idx].set_title(title)
        axes_flat[idx].set_xlabel("x (mm)")
        axes_flat[idx].set_ylabel("y (mm)")
        plt.colorbar(im, ax=axes_flat[idx], label="Masque")
        
        # Calculer l'aire effective
        effective_area = np.sum(mask) * (beam_diameter_mm / num_points)**2
        print(f"  {title}: aire effective = {effective_area:.2f} mm²")
    
    plt.tight_layout()
    plt.savefig('examples/output/example11_various_shapes.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # =========================================================================
    # 4. Affichage automatique des cartes d'intensité et de phase
    # =========================================================================
    print("\n--- 4. Affichage automatique des cartes d'intensité et de phase ---")
    
    # Créer un faisceau
    beam = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=beam_diameter_mm,
        num_points=num_points,
    )
    electric_field = beam.generate_electric_field(method="gaussian", sigma_mm=2.0)
    beam.electric_field = electric_field
    beam.intensity = beam.compute_intensity_from_electric_field(electric_field)
    beam.phase = beam.extract_phase_from_electric_field(electric_field)
    
    # Créer un système optique
    system = OpticalSystem()
    system.add_lens(
        name="Lentille 1",
        focal_length_mm=100.0,
        diameter_mm=beam_diameter_mm,
        position_z_mm=50.0,
    )
    system.add_element(
        Prism(
            name="Prisme",
            diameter_mm=beam_diameter_mm,
            prism_type=PrismType.RIGHT_ANGLE,
            apex_angle_deg=60.0,
            position_z_mm=100.0,
        )
    )
    system.add_element(
        CementedDoublet(
            name="Doublet",
            lens1=create_lens_from_preset("plan_convex_standard", beam_diameter_mm, 100.0),
            lens2=create_lens_from_preset("plan_concave_standard", beam_diameter_mm, 100.0),
            interface_radius_mm=100.0,
            position_z_mm=150.0,
        )
    )
    
    # Propager le faisceau
    final_beam = system.propagate_beam(beam, initial_position_mm=0.0, use_propagation=True)
    
    # Affichage automatique des cartes d'intensité et de phase
    print("\nAffichage automatique des cartes d'intensité et de phase:")
    
    # Utiliser les fonctions de Visualization.py
    plot_intensity(beam.intensity, beam_diameter_mm, title="Intensité initiale")
    plt.savefig('examples/output/example11_intensity_initial.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    plot_phase(beam.phase, beam_diameter_mm, title="Phase initiale")
    plt.savefig('examples/output/example11_phase_initial.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    plot_intensity(final_beam.intensity, beam_diameter_mm, title="Intensité finale")
    plt.savefig('examples/output/example11_intensity_final.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    plot_phase(final_beam.phase, beam_diameter_mm, title="Phase finale")
    plt.savefig('examples/output/example11_phase_final.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # Affichage combiné
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    im1 = axes[0, 0].imshow(
        beam.intensity,
        extent=[-beam_diameter_mm/2, beam_diameter_mm/2, -beam_diameter_mm/2, beam_diameter_mm/2],
        cmap='viridis'
    )
    axes[0, 0].set_title("Intensité initiale")
    axes[0, 0].set_xlabel("x (mm)")
    axes[0, 0].set_ylabel("y (mm)")
    plt.colorbar(im1, ax=axes[0, 0], label="Intensité (a.u.)")
    
    im2 = axes[0, 1].imshow(
        beam.phase,
        extent=[-beam_diameter_mm/2, beam_diameter_mm/2, -beam_diameter_mm/2, beam_diameter_mm/2],
        cmap='coolwarm'
    )
    axes[0, 1].set_title("Phase initiale")
    axes[0, 1].set_xlabel("x (mm)")
    axes[0, 1].set_ylabel("y (mm)")
    plt.colorbar(im2, ax=axes[0, 1], label="Phase (nm)")
    
    im3 = axes[1, 0].imshow(
        final_beam.intensity,
        extent=[-beam_diameter_mm/2, beam_diameter_mm/2, -beam_diameter_mm/2, beam_diameter_mm/2],
        cmap='viridis'
    )
    axes[1, 0].set_title("Intensité finale")
    axes[1, 0].set_xlabel("x (mm)")
    axes[1, 0].set_ylabel("y (mm)")
    plt.colorbar(im3, ax=axes[1, 0], label="Intensité (a.u.)")
    
    im4 = axes[1, 1].imshow(
        final_beam.phase,
        extent=[-beam_diameter_mm/2, beam_diameter_mm/2, -beam_diameter_mm/2, beam_diameter_mm/2],
        cmap='coolwarm'
    )
    axes[1, 1].set_title("Phase finale")
    axes[1, 1].set_xlabel("x (mm)")
    axes[1, 1].set_ylabel("y (mm)")
    plt.colorbar(im4, ax=axes[1, 1], label="Phase (nm)")
    
    plt.tight_layout()
    plt.savefig('examples/output/example11_combined_display.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # =========================================================================
    # 5. Tests d'intégration avec les fonctions de propagation
    # =========================================================================
    print("\n--- 5. Tests d'intégration avec les fonctions de propagation ---")
    
    # 5.1 Propagation à travers un doublet
    print("\n5.1 Propagation à travers un doublet collé")
    
    doublet_system = OpticalSystem()
    doublet_system.add_element(
        create_doublet_from_preset(
            preset_name="achromatic_standard",
            diameter_mm=beam_diameter_mm,
            focal_length_mm=100.0,
            material1_name="BK7",
            material2_name="SF5",
            position_z_mm=50.0,
        )
    )
    
    beam_doublet = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=beam_diameter_mm,
        num_points=128,
    )
    electric_field_doublet = beam_doublet.generate_electric_field(method="gaussian", sigma_mm=2.0)
    beam_doublet.electric_field = electric_field_doublet
    beam_doublet.intensity = beam_doublet.compute_intensity_from_electric_field(electric_field_doublet)
    beam_doublet.phase = beam_doublet.extract_phase_from_electric_field(electric_field_doublet)
    
    final_beam_doublet = doublet_system.propagate_beam(beam_doublet, initial_position_mm=0.0, use_propagation=True)
    
    pv_doublet, rms_doublet = final_beam_doublet.compute_pv_rms(final_beam_doublet.phase)
    print(f"  Faisceau après doublet: PV={pv_doublet:.2f} nm, RMS={rms_doublet:.2f} nm")
    
    # 5.2 Propagation à travers un prisme
    print("\n5.2 Propagation à travers un prisme")
    
    prism_system = OpticalSystem()
    prism_system.add_element(
        Prism(
            name="Prisme",
            diameter_mm=beam_diameter_mm,
            prism_type=PrismType.RIGHT_ANGLE,
            apex_angle_deg=60.0,
            position_z_mm=50.0,
        )
    )
    
    beam_prism = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=beam_diameter_mm,
        num_points=128,
    )
    electric_field_prism = beam_prism.generate_electric_field(method="gaussian", sigma_mm=2.0)
    beam_prism.electric_field = electric_field_prism
    beam_prism.intensity = beam_prism.compute_intensity_from_electric_field(electric_field_prism)
    beam_prism.phase = beam_prism.extract_phase_from_electric_field(electric_field_prism)
    
    final_beam_prism = prism_system.propagate_beam(beam_prism, initial_position_mm=0.0, use_propagation=True)
    
    pv_prism, rms_prism = final_beam_prism.compute_pv_rms(final_beam_prism.phase)
    print(f"  Faisceau après prisme: PV={pv_prism:.2f} nm, RMS={rms_prism:.2f} nm")
    
    # 5.3 Système complet avec plusieurs optiques
    print("\n5.3 Système complet avec plusieurs optiques")
    
    complex_system = OpticalSystem()
    complex_system.add_lens(
        name="Lentille 1",
        focal_length_mm=100.0,
        diameter_mm=beam_diameter_mm,
        position_z_mm=50.0,
    )
    complex_system.add_element(
        Prism(
            name="Prisme",
            diameter_mm=beam_diameter_mm,
            prism_type=PrismType.RIGHT_ANGLE,
            apex_angle_deg=60.0,
            position_z_mm=100.0,
        )
    )
    complex_system.add_element(
        create_doublet_from_preset(
            preset_name="achromatic_standard",
            diameter_mm=beam_diameter_mm,
            focal_length_mm=100.0,
            position_z_mm=150.0,
        )
    )
    complex_system.add_mirror(
        name="Miroir",
        diameter_mm=beam_diameter_mm,
        mirror_type=MirrorType.FLAT,
        position_z_mm=200.0,
    )
    
    beam_complex = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=beam_diameter_mm,
        num_points=128,
    )
    electric_field_complex = beam_complex.generate_electric_field(method="gaussian", sigma_mm=2.0)
    beam_complex.electric_field = electric_field_complex
    beam_complex.intensity = beam_complex.compute_intensity_from_electric_field(electric_field_complex)
    beam_complex.phase = beam_complex.extract_phase_from_electric_field(electric_field_complex)
    
    final_beam_complex = complex_system.propagate_beam(beam_complex, initial_position_mm=0.0, use_propagation=True)
    
    pv_complex, rms_complex = final_beam_complex.compute_pv_rms(final_beam_complex.phase)
    print(f"  Faisceau après système complet: PV={pv_complex:.2f} nm, RMS={rms_complex:.2f} nm")
    print(f"  Nombre d'éléments dans le système: {len(complex_system.elements)}")
    
    # Visualisation du système complet
    plot_intensity(final_beam_complex.intensity, beam_diameter_mm, title="Intensité finale (système complet)")
    plt.savefig('examples/output/example11_complex_system_intensity.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    plot_phase(final_beam_complex.phase, beam_diameter_mm, title="Phase finale (système complet)")
    plt.savefig('examples/output/example11_complex_system_phase.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # =========================================================================
    # 6. Résumé
    # =========================================================================
    print("\n" + "="*80)
    print("Example 11 terminé avec succès !")
    print("Les images ont été sauvegardées dans examples/output/")
    print("="*80)


if __name__ == "__main__":
    import os
    os.makedirs('examples/output', exist_ok=True)
    run_advanced_optics_example()

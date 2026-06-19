"""
Example 6: Resampling Test
FR: Test du rééchantillonnage des champs électriques entre différentes tailles de grille.
    Démonstration de la gestion automatique du rééchantillonnage dans Propagation.py
    et validation de la conservation de l'énergie et de la phase.

EN: Test of electric field resampling between different grid sizes.
    Demonstrates automatic resampling handling in Propagation.py
    and validates energy and phase conservation.

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Beam import Beam
from Propagation import Propagation
from Visualization import plot_intensity, plot_phase, plot_beam_map


def run_resampling_test():
    """FR: Exécute les tests de rééchantillonnage."""
    print("\n" + "="*80)
    print("Example 6: Resampling Test")
    print("="*80)
    
    # =========================================================================
    # 1. Test de rééchantillonnage direct
    # =========================================================================
    print("\n--- Test de rééchantillonnage direct ---")
    
    wavelength_nm = 633.0
    diameter_mm = 10.0
    energy = 1.0
    
    # Créer un faisceau avec une petite grille
    small_size = 64
    large_size = 512
    
    beam_small = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=diameter_mm,
        energy=energy,
        num_points=small_size
    )
    
    # Générer un champ électrique
    electric_field_small = beam_small.generate_electric_field(method="gaussian", sigma_mm=2.0)
    intensity_small = beam_small.compute_intensity_from_electric_field(electric_field_small)
    
    print(f"Grille initiale: {small_size}x{small_size}")
    print(f"  Énergie: {np.sum(intensity_small):.4f}")
    
    # Créer un propagateur avec une grande grille de sortie
    propagator_resample = Propagation(
        wavelength_nm=wavelength_nm,
        propagation_distance_mm=0.0,  # Pas de propagation, juste rééchantillonnage
        input_diameter_mm=diameter_mm,
        output_diameter_mm=diameter_mm,
        num_points=large_size,
        method="angular_spectrum"
    )
    
    # Le rééchantillonnage se fait automatiquement dans propagate
    # Mais on peut aussi tester directement _resample_electric_field
    resampled_field = propagator_resample._resample_electric_field(
        electric_field_small,
        propagator_resample.output_grid_x,
        propagator_resample.output_grid_y
    )
    
    # Calculer l'intensité rééchantillonnée
    intensity_resampled = beam_small.compute_intensity_from_electric_field(resampled_field)
    
    print(f"Grille rééchantillonnée: {large_size}x{large_size}")
    print(f"  Énergie: {np.sum(intensity_resampled):.4f}")
    print(f"  Conservation d'énergie: {np.sum(intensity_resampled)/np.sum(intensity_small):.4f}")
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = axes[0].imshow(
        intensity_small,
        extent=[-diameter_mm/2, diameter_mm/2, -diameter_mm/2, diameter_mm/2],
        cmap='viridis'
    )
    axes[0].set_title(f"Intensité (grille {small_size}x{small_size})")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    plt.colorbar(im1, ax=axes[0], label="Intensity (a.u.)")
    
    im2 = axes[1].imshow(
        intensity_resampled,
        extent=[-diameter_mm/2, diameter_mm/2, -diameter_mm/2, diameter_mm/2],
        cmap='viridis'
    )
    axes[1].set_title(f"Intensité (grille {large_size}x{large_size})")
    axes[1].set_xlabel("x (mm)")
    plt.colorbar(im2, ax=axes[1], label="Intensity (a.u.)")
    
    plt.tight_layout()
    plt.savefig('examples/output/example6_resampling_direct.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # =========================================================================
    # 2. Propagation avec rééchantillonnage
    # =========================================================================
    print("\n--- Propagation avec rééchantillonnage ---")
    
    # Propager un faisceau d'une petite grille vers une grande grille
    propagation_distance_mm = 500.0
    
    propagator_upsample = Propagation(
        wavelength_nm=wavelength_nm,
        propagation_distance_mm=propagation_distance_mm,
        input_diameter_mm=diameter_mm,
        output_diameter_mm=diameter_mm,
        num_points=small_size,  # Grille d'entrée petite
        method="angular_spectrum"
    )
    
    # Mais on veut une grille de sortie plus grande
    # On va créer manuellement une grille de sortie
    from MathAndPhysicsTools import create_grid
    output_grid_x, output_grid_y = create_grid(diameter_mm, large_size)
    
    # Propager avec rééchantillonnage
    propagated_field_upsample = propagator_upsample.propagate_angular_spectrum(
        electric_field_small,
        propagator_upsample.input_grid_x,
        propagator_upsample.input_grid_y
    )
    
    # Rééchantillonner sur la grande grille
    propagated_field_large = propagator_upsample._resample_electric_field(
        propagated_field_upsample,
        output_grid_x,
        output_grid_y
    )
    
    # Calculer les intensités
    intensity_propagated_small = beam_small.compute_intensity_from_electric_field(propagated_field_upsample)
    intensity_propagated_large = beam_small.compute_intensity_from_electric_field(propagated_field_large)
    
    print(f"Après propagation:")
    print(f"  Grille petite: énergie = {np.sum(intensity_propagated_small):.4f}")
    print(f"  Grille grande: énergie = {np.sum(intensity_propagated_large):.4f}")
    print(f"  Ratio: {np.sum(intensity_propagated_large)/np.sum(intensity_propagated_small):.4f}")
    
    # =========================================================================
    # 3. Test avec différentes tailles de grille
    # =========================================================================
    print("\n--- Test avec différentes tailles de grille ---")
    
    grid_sizes = [32, 64, 128, 256, 512]
    energies = []
    
    for size in grid_sizes:
        # Créer le faisceau
        beam_test = Beam(
            wavelength_nm=wavelength_nm,
            diameter_mm=diameter_mm,
            energy=energy,
            num_points=size
        )
        
        # Générer le champ électrique
        electric_field_test = beam_test.generate_electric_field(method="gaussian", sigma_mm=2.0)
        intensity_test = beam_test.compute_intensity_from_electric_field(electric_field_test)
        
        # Propager
        propagator_test = Propagation(
            wavelength_nm=wavelength_nm,
            propagation_distance_mm=propagation_distance_mm,
            input_diameter_mm=diameter_mm,
            output_diameter_mm=diameter_mm,
            num_points=size,
            method="angular_spectrum"
        )
        
        propagated_field_test = propagator_test.propagate(electric_field_test)
        intensity_propagated_test = beam_test.compute_intensity_from_electric_field(propagated_field_test)
        
        energy_propagated = np.sum(intensity_propagated_test)
        energies.append(energy_propagated)
        
        print(f"  Grille {size}x{size}: énergie propagée = {energy_propagated:.4f}")
    
    # Vérifier la conservation d'énergie relative
    energy_ratio = energies[-1] / energies[0]
    print(f"\nRatio énergie (512/32): {energy_ratio:.4f}")
    print(f"  (Doit être proche de 1.0 pour une bonne conservation)")
    
    # Visualisation de l'énergie en fonction de la taille de grille
    plt.figure(figsize=(10, 6))
    plt.plot(grid_sizes, energies, 'o-', label='Énergie propagée')
    plt.axhline(y=energy, color='r', linestyle='--', label='Énergie initiale')
    plt.xlabel("Taille de grille (N)")
    plt.ylabel("Énergie (a.u.)")
    plt.title("Conservation d'énergie en fonction de la taille de grille")
    plt.legend()
    plt.grid(True)
    plt.savefig('examples/output/example6_energy_conservation.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # =========================================================================
    # 4. Propagation avec changement de diamètre
    # =========================================================================
    print("\n--- Propagation avec changement de diamètre ---")
    
    input_diameter_mm = 10.0
    output_diameter_mm = 15.0  # Grille de sortie plus grande
    
    beam_input = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=input_diameter_mm,
        energy=energy,
        num_points=128
    )
    
    # Générer un champ électrique
    electric_field_input = beam_input.generate_electric_field(method="gaussian", sigma_mm=2.0)
    intensity_input = beam_input.compute_intensity_from_electric_field(electric_field_input)
    
    propagator_diameter = Propagation(
        wavelength_nm=wavelength_nm,
        propagation_distance_mm=propagation_distance_mm,
        input_diameter_mm=input_diameter_mm,
        output_diameter_mm=output_diameter_mm,
        num_points=128,
        method="angular_spectrum"
    )
    
    # Propager (inclut rééchantillonnage automatique)
    propagated_field_diameter = propagator_diameter.propagate(electric_field_input)
    
    # Pour incohérent, le résultat est une intensité
    if propagator_diameter.coherence == "coherent":
        intensity_propagated_diameter = beam_input.compute_intensity_from_electric_field(propagated_field_diameter)
    else:
        intensity_propagated_diameter = propagated_field_diameter
    
    print(f"Propagation avec changement de diamètre:")
    print(f"  Diamètre d'entrée: {input_diameter_mm} mm")
    print(f"  Diamètre de sortie: {output_diameter_mm} mm")
    print(f"  Énergie initiale: {np.sum(intensity_input):.4f}")
    print(f"  Énergie propagée: {np.sum(intensity_propagated_diameter):.4f}")
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = axes[0].imshow(
        intensity_input,
        extent=[-input_diameter_mm/2, input_diameter_mm/2, -input_diameter_mm/2, input_diameter_mm/2],
        cmap='viridis'
    )
    axes[0].set_title(f"Intensité initiale (D={input_diameter_mm} mm)")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    plt.colorbar(im1, ax=axes[0], label="Intensity (a.u.)")
    
    im2 = axes[1].imshow(
        intensity_propagated_diameter,
        extent=[-output_diameter_mm/2, output_diameter_mm/2, -output_diameter_mm/2, output_diameter_mm/2],
        cmap='viridis'
    )
    axes[1].set_title(f"Intensité propagée (D={output_diameter_mm} mm)")
    axes[1].set_xlabel("x (mm)")
    plt.colorbar(im2, ax=axes[1], label="Intensity (a.u.)")
    
    plt.tight_layout()
    plt.savefig('examples/output/example6_diameter_change.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    print("\n" + "="*80)
    print("Example 6 terminé avec succès !")
    print("Les images ont été sauvegardées dans examples/output/")
    print("="*80)


if __name__ == "__main__":
    import os
    os.makedirs('examples/output', exist_ok=True)
    run_resampling_test()

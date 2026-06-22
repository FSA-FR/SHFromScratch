"""
Script pour générer une image des tâches d'Airy après une matrice de microlentilles.
FR: Ce script crée un système Shack-Hartmann complet, simule le passage d'un faisceau
    à travers la matrice de microlentilles, et génère une image des tâches d'Airy
    telles que mesurées par la caméra.

EN: Script to generate an image of Airy spots after a microlens array.
    This script creates a complete Shack-Hartmann system, simulates the passage
    of a beam through the microlens array, and generates an image of the Airy spots
    as measured by the camera.

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from Shack_Hartmann import create_shack_hartmann, CentroidAlgorithm
from Beam import Beam
from Optiques import WaveFrontError


def generate_airy_spots_image():
    """FR: Génère une image des tâches d'Airy."""
    print("\n" + "="*80)
    print("Génération d'une image des tâches d'Airy")
    print("="*80)
    
    # Paramètres
    wavelength_nm = 633.0  # He-Ne laser
    output_dir = 'examples/output'
    os.makedirs(output_dir, exist_ok=True)
    
    # =========================================================================
    # 1. Système Shack-Hartmann
    # =========================================================================
    print("\n1. Création du système Shack-Hartmann...")
    sh = create_shack_hartmann(
        name="AirySpots_Demo",
        num_microlenses_x=10,
        num_microlenses_y=10,
        microlens_pitch_mm=0.4,  # Espacement bord-à-bord
        focal_length_mm=20.0,
        camera_num_pixels_x=1024,
        camera_num_pixels_y=1024,
        camera_pixel_size_um=5.0,
        wavelength_nm=wavelength_nm,
        centroid_algorithm=CentroidAlgorithm.WEIGHTED_CENTROID,
        display=False,
        display_dir=output_dir
    )
    
    print(f"  Matrice: {sh.microlens_array.num_elements_x}x{sh.microlens_array.num_elements_y} microlentilles")
    print(f"  Distance focale: {sh.focal_length_mm} mm")
    print(f"  Caméra: {sh.camera.specifications.num_pixels_x}x{sh.camera.specifications.num_pixels_y} pixels")
    print(f"  Taille des pixels: {sh.camera.pixel_width_mm*1e3:.1f} µm")
    
    # =========================================================================
    # 2. Faisceau gaussien
    # =========================================================================
    print("\n2. Création d'un faisceau gaussien...")
    beam = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=5.0,
        num_points=1024
    )
    
    # Générer un champ électrique gaussien
    electric_field = beam.generate_electric_field(
        method="gaussian",
        sigma_mm=1.0
    )
    beam.electric_field = electric_field
    beam.intensity = beam.compute_intensity_from_electric_field(electric_field)
    beam.phase = beam.extract_phase_from_electric_field(electric_field)
    
    print(f"  Diamètre du faisceau: {beam.diameter_mm} mm")
    print(f"  Sigma: 1.0 mm")
    
    # =========================================================================
    # 3. Simulation Shack-Hartmann
    # =========================================================================
    print("\n3. Simulation Shack-Hartmann...")
    sh.simulate(beam)
    
    print(f"  Tâches détectées: {len(sh.centroids)}")
    print(f"  Forme de l'image: {sh.spot_image.shape}")
    
    # =========================================================================
    # 4. Visualisation
    # =========================================================================
    print("\n4. Visualisation des tâches d'Airy...")
    
    # 4.1 Image brute des tâches
    print("\n4.1 Image brute des tâches d'Airy")
    plt.figure(figsize=(12, 10))
    
    extent = [
        -sh.camera.sensor_width_mm / 2,
        sh.camera.sensor_width_mm / 2,
        -sh.camera.sensor_height_mm / 2,
        sh.camera.sensor_height_mm / 2
    ]
    
    plt.imshow(sh.spot_image, cmap='hot', extent=extent, origin='lower')
    plt.title("Image des tâches d'Airy\n" +
             f"Faisceau gaussien (σ=1.0mm), λ={wavelength_nm}nm, f={sh.focal_length_mm}mm")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.colorbar(label="Intensité (ADU)")
    
    output_path = os.path.join(output_dir, "airy_spots_raw.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Image sauvegardée: {output_path}")
    
    # 4.2 Image avec les centroïdes marqués
    print("\n4.2 Image avec les centroïdes marqués")
    centroid_map = sh.get_centroid_map()
    
    plt.figure(figsize=(12, 10))
    plt.imshow(centroid_map, cmap='hot', extent=extent, origin='lower')
    plt.title("Tâches d'Airy avec centroïdes marqués\n" +
             f"Algorithme: {sh.centroid_algorithm.value}")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.colorbar(label="Intensité normalisée")
    
    output_path = os.path.join(output_dir, "airy_spots_with_centroids.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Image sauvegardée: {output_path}")
    
    # 4.3 Zoom sur une tâche individuelle
    print("\n4.3 Zoom sur une tâche individuelle")
    
    # Sélectionner la tâche centrale (index 5x5 dans une matrice 10x10)
    num_spots_x = sh.microlens_array.num_elements_x
    num_spots_y = sh.microlens_array.num_elements_y
    
    spot_size_pixels_x = sh.camera.specifications.num_pixels_x // num_spots_x
    spot_size_pixels_y = sh.camera.specifications.num_pixels_y // num_spots_y
    
    # Tâche centrale (indices 4,4 pour 0-based)
    center_i, center_j = num_spots_x // 2, num_spots_y // 2
    
    x_start = center_i * spot_size_pixels_x
    x_end = (center_i + 1) * spot_size_pixels_x
    y_start = center_j * spot_size_pixels_y
    y_end = (center_j + 1) * spot_size_pixels_y
    
    single_spot = sh.spot_image[y_start:y_end, x_start:x_end]
    
    # Position du centroïde de cette tâche
    spot_idx = center_j * num_spots_x + center_i
    centroid_x = sh.centroids[spot_idx, 0] - x_start
    centroid_y = sh.centroids[spot_idx, 1] - y_start
    
    plt.figure(figsize=(8, 8))
    plt.imshow(single_spot, cmap='hot', extent=[0, spot_size_pixels_x * sh.camera.pixel_width_mm,
                                                 0, spot_size_pixels_y * sh.camera.pixel_height_mm],
               origin='lower')
    plt.scatter([centroid_x * sh.camera.pixel_width_mm], 
                [centroid_y * sh.camera.pixel_height_mm],
                color='red', s=100, marker='x', label=f'Centroïde\n({centroid_x:.1f}, {centroid_y:.1f}) px')
    plt.title(f"Tâche d'Airy individuelle (centrale)\n" +
             f"Taille: {spot_size_pixels_x * sh.camera.pixel_width_mm:.2f}mm x " +
             f"{spot_size_pixels_y * sh.camera.pixel_height_mm:.2f}mm")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.legend()
    plt.colorbar(label="Intensité (ADU)")
    
    output_path = os.path.join(output_dir, "airy_spot_single_zoom.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Image sauvegardée: {output_path}")
    
    # 4.4 Profil de la tâche centrale
    print("\n4.4 Profil de la tâche centrale")
    
    # Profil horizontal (ligne médiane)
    y_mid = single_spot.shape[0] // 2
    x_profile = single_spot[y_mid, :]
    
    # Profil vertical (colonne médiane)
    x_mid = single_spot.shape[1] // 2
    y_profile = single_spot[:, x_mid]
    
    x_positions = np.arange(single_spot.shape[1]) * sh.camera.pixel_width_mm
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Profil horizontal
    axes[0].plot(x_positions, x_profile, 'b-', linewidth=2)
    axes[0].axvline(x=centroid_x * sh.camera.pixel_width_mm, color='r', linestyle='--',
                    label=f'Centroïde x={centroid_x:.1f} px')
    axes[0].set_title("Profil horizontal de la tâche centrale")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("Intensité (ADU)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Profil vertical
    axes[1].plot(x_positions, y_profile, 'b-', linewidth=2)
    axes[1].axvline(x=centroid_y * sh.camera.pixel_height_mm, color='r', linestyle='--',
                    label=f'Centroïde y={centroid_y:.1f} px')
    axes[1].set_title("Profil vertical de la tâche centrale")
    axes[1].set_xlabel("y (mm)")
    axes[1].set_ylabel("Intensité (ADU)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "airy_spot_profiles.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Image sauvegardée: {output_path}")
    
    # =========================================================================
    # 5. Faisceau avec aberrations
    # =========================================================================
    print("\n5. Simulation avec un faisceau aberré...")
    
    # Créer un faisceau avec des aberrations
    beam_aberrated = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=5.0,
        num_points=1024
    )
    
    electric_field_aberrated = beam_aberrated.generate_electric_field(
        method="gaussian",
        sigma_mm=1.0
    )
    
    # Ajouter des aberrations (Defocus + Astigmatisme)
    wfe = WaveFrontError(
        zernike_coefficients={
            (2, 0): 5.0,   # Defocus
            (2, 2): 3.0    # Astigmatisme
        },
        seed=42
    )
    
    # Appliquer les aberrations
    x = np.linspace(-beam_aberrated.diameter_mm/2, beam_aberrated.diameter_mm/2, beam_aberrated.num_points)
    y = np.linspace(-beam_aberrated.diameter_mm/2, beam_aberrated.diameter_mm/2, beam_aberrated.num_points)
    X, Y = np.meshgrid(x, y)
    
    phase_map_nm = wfe.generate_phase_map(X, Y, beam_aberrated.wavelength_nm)
    phase_rad = phase_map_nm * 2 * np.pi / beam_aberrated.wavelength_nm
    
    aberrated_field = np.abs(electric_field_aberrated) * np.exp(
        1j * (np.angle(electric_field_aberrated) + phase_rad)
    )
    beam_aberrated.electric_field = aberrated_field
    beam_aberrated.intensity = beam_aberrated.compute_intensity_from_electric_field(aberrated_field)
    
    # Simuler
    sh_aberrated = create_shack_hartmann(
        name="AirySpots_Aberrated",
        num_microlenses_x=10,
        num_microlenses_y=10,
        microlens_pitch_mm=0.4,
        focal_length_mm=20.0,
        camera_num_pixels_x=1024,
        camera_num_pixels_y=1024,
        camera_pixel_size_um=5.0,
        wavelength_nm=wavelength_nm,
        centroid_algorithm=CentroidAlgorithm.WEIGHTED_CENTROID,
        display=False,
        display_dir=output_dir
    )
    
    sh_aberrated.simulate(beam_aberrated)
    
    # Visualiser
    plt.figure(figsize=(12, 10))
    plt.imshow(sh_aberrated.spot_image, cmap='hot', extent=extent, origin='lower')
    plt.title("Tâches d'Airy avec aberrations\n" +
             f"Defocus + Astigmatisme, λ={wavelength_nm}nm, f={sh_aberrated.focal_length_mm}mm")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.colorbar(label="Intensité (ADU)")
    
    output_path = os.path.join(output_dir, "airy_spots_aberrated.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Image sauvegardée: {output_path}")
    
    # =========================================================================
    # 6. Comparaison avec/sans aberrations
    # =========================================================================
    print("\n6. Comparaison avec/sans aberrations...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Sans aberrations
    im1 = axes[0].imshow(sh.spot_image, cmap='hot', extent=extent, origin='lower')
    axes[0].set_title("Sans aberrations\nFaisceau gaussien parfait")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    plt.colorbar(im1, ax=axes[0], label="Intensité (ADU)")
    
    # Avec aberrations
    im2 = axes[1].imshow(sh_aberrated.spot_image, cmap='hot', extent=extent, origin='lower')
    axes[1].set_title("Avec aberrations\nDefocus + Astigmatisme")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("y (mm)")
    plt.colorbar(im2, ax=axes[1], label="Intensité (ADU)")
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "airy_spots_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Image sauvegardée: {output_path}")
    
    # =========================================================================
    # 7. Statistiques
    # =========================================================================
    print("\n7. Statistiques des tâches d'Airy...")
    
    # Sans aberrations
    stats_clean = {
        'mean_intensity': float(np.mean(sh.spot_image)),
        'max_intensity': float(np.max(sh.spot_image)),
        'std_intensity': float(np.std(sh.spot_image)),
        'num_spots': len(sh.centroids),
        'centroid_mean_error': float(np.mean(sh.centroid_errors))
    }
    
    # Avec aberrations
    stats_aberrated = {
        'mean_intensity': float(np.mean(sh_aberrated.spot_image)),
        'max_intensity': float(np.max(sh_aberrated.spot_image)),
        'std_intensity': float(np.std(sh_aberrated.spot_image)),
        'num_spots': len(sh_aberrated.centroids),
        'centroid_mean_error': float(np.mean(sh_aberrated.centroid_errors))
    }
    
    print(f"\n  Sans aberrations:")
    print(f"    Intensité moyenne: {stats_clean['mean_intensity']:.2f} ADU")
    print(f"    Intensité max: {stats_clean['max_intensity']:.2f} ADU")
    print(f"    Erreur moyenne sur centroïdes: {stats_clean['centroid_mean_error']:.4f} pixels")
    
    print(f"\n  Avec aberrations:")
    print(f"    Intensité moyenne: {stats_aberrated['mean_intensity']:.2f} ADU")
    print(f"    Intensité max: {stats_aberrated['max_intensity']:.2f} ADU")
    print(f"    Erreur moyenne sur centroïdes: {stats_aberrated['centroid_mean_error']:.4f} pixels")
    
    # =========================================================================
    # 8. Fin
    # =========================================================================
    print("\n" + "="*80)
    print("Génération des images des tâches d'Airy terminée !")
    print(f"Les images ont été sauvegardées dans {output_dir}/")
    print("="*80)
    print("\nImages générées:")
    print("  ✓ airy_spots_raw.png")
    print("  ✓ airy_spots_with_centroids.png")
    print("  ✓ airy_spot_single_zoom.png")
    print("  ✓ airy_spot_profiles.png")
    print("  ✓ airy_spots_aberrated.png")
    print("  ✓ airy_spots_comparison.png")
    print("="*80)


if __name__ == "__main__":
    generate_airy_spots_image()

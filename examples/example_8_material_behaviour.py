"""
Example 8: Material Behaviour
FR: Exemple d'utilisation du module Material_Behaviour.py.
    Démonstration du calcul de l'indice de réfraction, de l'expansion thermique,
    de la réflectance/transmittance, et de la variation de puissance optique
    pour différents matériaux optiques et mécaniques.

EN: Example of using Material_Behaviour.py module.
    Demonstrates calculation of refractive index, thermal expansion,
    reflectance/transmittance, and optical power variation
    for various optical and mechanical materials.

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Material_Behaviour import MaterialBehaviour, get_available_materials, Polarization


def run_material_behaviour_example():
    """FR: Exécute l'exemple de comportement des matériaux."""
    print("\n" + "="*80)
    print("Example 8: Material Behaviour")
    print("="*80)
    
    # =========================================================================
    # 1. Liste des matériaux disponibles
    # =========================================================================
    print("\n--- Matériaux disponibles ---")
    materials = get_available_materials()
    print(f"Matériaux optiques: {[m for m in materials if m in ['Fused_Silica', 'BK7', 'SF5', 'Silicon']]}")
    print(f"Matériaux mécaniques: {[m for m in materials if m in ['Steel', 'Aluminum', 'Invar', 'Copper']]}")
    
    # =========================================================================
    # 2. Indice de réfraction en fonction de la longueur d'onde
    # =========================================================================
    print("\n--- Indice de réfraction en fonction de la longueur d'onde ---")
    
    # Longueurs d'onde à tester (nm)
    wavelengths_nm = np.linspace(200, 2500, 100)
    
    # Matériaux optiques à tester
    optical_materials = ["Fused_Silica", "BK7", "SF5", "Silicon"]
    
    # Créer le plot
    plt.figure(figsize=(12, 8))
    
    for material_name in optical_materials:
        try:
            material = MaterialBehaviour(material_name)
            n_values = []
            for wavelength in wavelengths_nm:
                try:
                    n = material.get_refractive_index(wavelength)
                    n_values.append(n)
                except:
                    n_values.append(np.nan)
            
            plt.plot(wavelengths_nm, n_values, label=material_name, linewidth=2)
            print(f"  {material_name}: n({633}) = {material.get_refractive_index(633.0):.4f}")
        except Exception as e:
            print(f"  ⚠️  Erreur avec {material_name}: {e}")
    
    plt.xlabel("Longueur d'onde (nm)")
    plt.ylabel("Indice de réfraction (n)")
    plt.title("Indice de réfraction en fonction de la longueur d'onde")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(200, 2500)
    plt.ylim(1.0, 5.0)
    plt.savefig('examples/output/example8_refractive_index_vs_wavelength.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # =========================================================================
    # 3. Indice de réfraction en fonction de la température
    # =========================================================================
    print("\n--- Indice de réfraction en fonction de la température ---")
    
    # Températures à tester (K)
    temperatures_K = np.linspace(200, 500, 50)
    wavelength_nm = 633.0
    
    plt.figure(figsize=(12, 8))
    
    for material_name in ["Fused_Silica", "BK7", "SF5"]:
        material = MaterialBehaviour(material_name)
        n_values = []
        for temp in temperatures_K:
            n = material.get_refractive_index(wavelength_nm, temperature_K=temp)
            n_values.append(n)
        
        plt.plot(temperatures_K - 273.15, n_values, label=material_name, linewidth=2)  # Convertir en °C
        print(f"  {material_name}: n(633nm, 20°C) = {material.get_refractive_index(633.0, 293.15):.6f}")
        print(f"                    n(633nm, 100°C) = {material.get_refractive_index(633.0, 373.15):.6f}")
        print(f"                    dn/dT ≈ {(n_values[-1] - n_values[0]) / (temperatures_K[-1] - temperatures_K[0]) * 1e6:.2f} ppm/°C")
    
    plt.xlabel("Température (°C)")
    plt.ylabel("Indice de réfraction (n)")
    plt.title("Indice de réfraction en fonction de la température (λ = 633 nm)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('examples/output/example8_refractive_index_vs_temperature.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # =========================================================================
    # 4. Expansion thermique et dilatation
    # =========================================================================
    print("\n--- Expansion thermique et dilatation ---")
    
    # Matériaux à tester (optiques et mécaniques)
    materials_for_thermal = ["Fused_Silica", "BK7", "SF5", "Silicon", "Steel", "Aluminum", "Invar"]
    
    # Températures (K)
    temperatures_K = np.linspace(273.15, 373.15, 100)  # 0°C à 100°C
    initial_length_m = 0.1  # 10 cm
    
    plt.figure(figsize=(12, 8))
    
    for material_name in materials_for_thermal:
        material = MaterialBehaviour(material_name)
        cte = material.get_thermal_expansion_coefficient()
        
        # Calculer la dilatation relative ΔL/L
        delta_L_L = material.get_thermal_expansion(temperatures_K, reference_temperature_K=273.15)
        
        plt.plot(temperatures_K - 273.15, delta_L_L * 1e6, label=material_name, linewidth=2)  # ppm
        print(f"  {material_name}: CTE = {cte * 1e6:.2f} ppm/°C")
        print(f"                    ΔL/L (100°C) = {delta_L_L[-1] * 1e6:.2f} ppm")
        print(f"                    ΔL (10 cm) = {material.get_thermal_dilation(initial_length_m, 373.15) * 1000:.3f} mm")
    
    plt.xlabel("Température (°C)")
    plt.ylabel("Expansion thermique (ppm)")
    plt.title("Expansion thermique relative (ΔL/L) en fonction de la température")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('examples/output/example8_thermal_expansion.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # =========================================================================
    # 5. Réflectance en fonction de l'angle d'incidence
    # =========================================================================
    print("\n--- Réflectance en fonction de l'angle d'incidence ---")
    
    material = MaterialBehaviour("Fused_Silica")
    wavelength_nm = 633.0
    
    # Angles d'incidence (degrés)
    angles_deg = np.linspace(0, 85, 100)
    
    # Calculer la réflectance pour différentes polarisations
    R_s = []
    R_p = []
    R_unpolarized = []
    
    for angle in angles_deg:
        R_s.append(material.get_reflectance(wavelength_nm, angle, Polarization.S))
        R_p.append(material.get_reflectance(wavelength_nm, angle, Polarization.P))
        R_unpolarized.append(material.get_reflectance(wavelength_nm, angle, Polarization.NONE))
    
    plt.figure(figsize=(12, 8))
    plt.plot(angles_deg, R_s, label="Polarisation S", linewidth=2)
    plt.plot(angles_deg, R_p, label="Polarisation P", linewidth=2)
    plt.plot(angles_deg, R_unpolarized, label="Non polarisée", linewidth=2, linestyle='--')
    
    plt.xlabel("Angle d'incidence (degrés)")
    plt.ylabel("Réflectance (R)")
    plt.title("Réflectance en fonction de l'angle d'incidence (Fused Silica, λ = 633 nm)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.savefig('examples/output/example8_reflectance_vs_angle.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # Afficher les valeurs à quelques angles
    for angle in [0, 30, 45, 60, 80]:
        print(f"  Angle {angle}°: R_s={material.get_reflectance(633.0, angle, Polarization.S):.4f}, "
              f"R_p={material.get_reflectance(633.0, angle, Polarization.P):.4f}, "
              f"R_moy={material.get_reflectance(633.0, angle, Polarization.NONE):.4f}")
    
    # =========================================================================
    # 6. Transmittance en fonction de l'épaisseur
    # =========================================================================
    print("\n--- Transmittance en fonction de l'épaisseur ---")
    
    material = MaterialBehaviour("Fused_Silica")
    wavelength_nm = 633.0
    
    # Épaisseurs (mm)
    thicknesses_mm = np.linspace(0.1, 50, 100)
    
    # Calculer la transmittance
    T_values = []
    for thickness in thicknesses_mm:
        T = material.get_transmittance(wavelength_nm, thickness, angle_deg=0.0)
        T_values.append(T)
    
    plt.figure(figsize=(12, 8))
    plt.plot(thicknesses_mm, T_values, label="Fused Silica (λ = 633 nm)", linewidth=2)
    
    plt.xlabel("Épaisseur (mm)")
    plt.ylabel("Transmittance (T)")
    plt.title("Transmittance en fonction de l'épaisseur")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.savefig('examples/output/example8_transmittance_vs_thickness.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # Afficher quelques valeurs
    for thickness in [0.1, 1.0, 10.0, 50.0]:
        T = material.get_transmittance(wavelength_nm, thickness)
        print(f"  Épaisseur {thickness} mm: T = {T:.4f}")
    
    # =========================================================================
    # 7. Variation de puissance optique avec la température
    # =========================================================================
    print("\n--- Variation de puissance optique avec la température ---")
    
    materials_for_optical = ["Fused_Silica", "BK7", "SF5"]
    focal_length_mm = 100.0
    temperatures_K = np.linspace(273.15, 373.15, 50)  # 0°C à 100°C
    
    plt.figure(figsize=(12, 8))
    
    for material_name in materials_for_optical:
        material = MaterialBehaviour(material_name)
        delta_power = []
        
        for temp in temperatures_K:
            delta_p = material.get_optical_power_variation(
                focal_length_mm, temp, reference_temperature_K=273.15, wavelength_nm=633.0
            )
            delta_power.append(delta_p * 100)  # Convertir en %
        
        plt.plot(temperatures_K - 273.15, delta_power, label=material_name, linewidth=2)
        
        # Variation de la distance focale
        delta_f_0C = material.get_focal_length_variation(focal_length_mm, 273.15, 273.15)
        delta_f_100C = material.get_focal_length_variation(focal_length_mm, 373.15, 273.15)
        print(f"  {material_name} (f={focal_length_mm} mm):")
        print(f"    ΔP/P (100°C) = {delta_power[-1]:.4f}%")
        print(f"    Δf (100°C) = {delta_f_100C - delta_f_0C:.4f} mm")
    
    plt.xlabel("Température (°C)")
    plt.ylabel("Variation de puissance optique (ΔP/P, %)")
    plt.title("Variation de puissance optique avec la température (f = 100 mm, λ = 633 nm)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('examples/output/example8_optical_power_variation.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    
    # =========================================================================
    # 8. Comparaison des propriétés thermiques des matériaux
    # =========================================================================
    print("\n--- Comparaison des propriétés thermiques ---")
    
    materials_for_comparison = ["Fused_Silica", "BK7", "SF5", "Silicon", "Steel", "Aluminum", "Invar"]
    
    # Créer un tableau comparatif
    data = {
        "Matériau": [],
        "Type": [],
        "CTE (ppm/°C)": [],
        "Densité (kg/m³)": [],
        "Module de Young (GPa)": [],
        "Conductivité thermique (W/m·K)": [],
        "n (633 nm)": [],
    }
    
    for material_name in materials_for_comparison:
        material = MaterialBehaviour(material_name)
        info = material.get_material_info()
        
        data["Matériau"].append(material_name)
        data["Type"].append("Optique" if material.material_type.value == "optical" else "Mécanique")
        data["CTE (ppm/°C)"].append(f"{material.get_thermal_expansion_coefficient() * 1e6:.2f}")
        data["Densité (kg/m³)"].append(f"{material.density_kg_m3:.0f}")
        data["Module de Young (GPa)"].append(f"{material.young_modulus_Pa / 1e9:.1f}")
        data["Conductivité thermique (W/m·K)"].append(f"{material.thermal_conductivity_W_mK:.2f}")
        
        if material.material_type.value == "optical":
            try:
                n = material.get_refractive_index(633.0)
                data["n (633 nm)"].append(f"{n:.4f}")
            except:
                data["n (633 nm)"].append("N/A")
        else:
            data["n (633 nm)"].append("N/A")
    
    # Afficher le tableau
    print("\nTableau comparatif des matériaux :")
    print("-" * 120)
    print(f"{|'Matériau':<15}|{'Type':<10}|{'CTE (ppm/°C)':>15}|{'Densité (kg/m³)':>18}|{'Module de Young (GPa)':>20}|{'Conductivité (W/m·K)':>20}|{'n (633 nm)':>12}|")
    print("-" * 120)
    for i in range(len(data["Matériau"])):
        print(f"{|data['Matériau'][i]:<15}|{data['Type'][i]:<10}|{data['CTE (ppm/°C)'][i]:>15}|{data['Densité (kg/m³)'][i]:>18}|{data['Module de Young (GPa)'][i]:>20}|{data['Conductivité thermique (W/m·K)'][i]:>20}|{data['n (633 nm)'][i]:>12}|")
    print("-" * 120)
    
    # =========================================================================
    # 9. Exemple d'application : Lentille en BK7
    # =========================================================================
    print("\n--- Exemple d'application : Lentille en BK7 ---")
    
    material = MaterialBehaviour("BK7")
    
    # Paramètres de la lentille
    focal_length_mm = 50.0  # Distance focale initiale
    diameter_mm = 25.0      # Diamètre
    thickness_mm = 5.0      # Épaisseur
    temperature_range_C = [20.0, 50.0, 100.0]  # Températures à tester
    
    print(f"\nLentille en BK7 :")
    print(f"  Distance focale initiale: {focal_length_mm} mm")
    print(f"  Diamètre: {diameter_mm} mm")
    print(f"  Épaisseur: {thickness_mm} mm")
    
    for temp_C in temperature_range_C:
        temp_K = temp_C + 273.15
        
        # Calculer la nouvelle distance focale
        delta_f = material.get_focal_length_variation(focal_length_mm, temp_K, 293.15)
        new_focal_length = focal_length_mm + delta_f
        
        # Calculer la dilatation du diamètre
        delta_diameter = material.get_thermal_dilation(diameter_mm * 1e-3, temp_K, 293.15) * 1e3  # mm
        new_diameter = diameter_mm + delta_diameter
        
        # Calculer la dilatation de l'épaisseur
        delta_thickness = material.get_thermal_dilation(thickness_mm * 1e-3, temp_K, 293.15) * 1e3  # mm
        new_thickness = thickness_mm + delta_thickness
        
        print(f"\n  À {temp_C}°C :")
        print(f"    Distance focale: {new_focal_length:.4f} mm (Δf = {delta_f:+.4f} mm)")
        print(f"    Diamètre: {new_diameter:.4f} mm (ΔD = {delta_diameter:+.4f} mm)")
        print(f"    Épaisseur: {new_thickness:.4f} mm (Δt = {delta_thickness:+.4f} mm)")
    
    # =========================================================================
    # 10. Exemple d'application : Système optique avec plusieurs matériaux
    # =========================================================================
    print("\n--- Exemple d'application : Système optique avec plusieurs matériaux ---")
    
    # Supposons un système optique avec :
    # - Lentille 1 : BK7
    # - Lentille 2 : Fused Silica
    # - Support : Aluminium
    
    materials_system = {
        "Lentille 1 (BK7)": MaterialBehaviour("BK7"),
        "Lentille 2 (Fused Silica)": MaterialBehaviour("Fused_Silica"),
        "Support (Aluminium)": MaterialBehaviour("Aluminum"),
    }
    
    initial_positions_mm = {
        "Lentille 1 (BK7)": 0.0,
        "Lentille 2 (Fused Silica)": 50.0,
        "Support (Aluminium)": 100.0,
    }
    
    initial_focal_lengths_mm = {
        "Lentille 1 (BK7)": 100.0,
        "Lentille 2 (Fused Silica)": 150.0,
    }
    
    print("\nSystème optique initial (à 20°C) :")
    for component, position in initial_positions_mm.items():
        print(f"  {component}: position = {position:.2f} mm")
    for component, f in initial_focal_lengths_mm.items():
        print(f"  {component}: distance focale = {f:.2f} mm")
    
    # Calculer les positions et distances focales à 100°C
    new_temperature_K = 373.15  # 100°C
    reference_temperature_K = 293.15  # 20°C
    
    print("\nSystème optique à 100°C :")
    for component, material in materials_system.items():
        if component in initial_positions_mm:
            initial_position = initial_positions_mm[component]
            # Supposons que la position est déterminée par le support en aluminium
            if "Support" in component:
                # Le support se dilate
                delta_position = material.get_thermal_dilation(initial_position * 1e-3, new_temperature_K, reference_temperature_K) * 1e3
                new_position = initial_position + delta_position
                print(f"  {component}: position = {new_position:.2f} mm (Δ = {delta_position:+.2f} mm)")
            else:
                # Les lentilles se déplacent avec le support
                support_material = materials_system["Support (Aluminium)"]
                delta_position = support_material.get_thermal_dilation(initial_position * 1e-3, new_temperature_K, reference_temperature_K) * 1e3
                new_position = initial_position + delta_position
                print(f"  {component}: position = {new_position:.2f} mm (Δ = {delta_position:+.2f} mm)")
        
        if component in initial_focal_lengths_mm:
            initial_f = initial_focal_lengths_mm[component]
            delta_f = material.get_focal_length_variation(initial_f, new_temperature_K, reference_temperature_K)
            new_f = initial_f + delta_f
            print(f"  {component}: distance focale = {new_f:.2f} mm (Δ = {delta_f:+.2f} mm)")
    
    print("\n" + "="*80)
    print("Example 8 terminé avec succès !")
    print("Les images ont été sauvegardées dans examples/output/")
    print("="*80)


if __name__ == "__main__":
    import os
    os.makedirs('examples/output', exist_ok=True)
    run_material_behaviour_example()

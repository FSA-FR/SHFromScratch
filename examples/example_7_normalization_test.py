"""
Example 7: Normalization Test
FR: Test complet des normalisations et unités.
    Démonstration de la gestion des normalisations de phase (RMS, PV),
    des unités d'énergie (J, mJ, a.u.), de puissance (W, mW, a.u.),
    et d'intensité (W/m², W/cm², a.u.).

EN: Complete test of normalizations and units.
    Demonstrates phase normalization (RMS, PV),
    energy units (J, mJ, a.u.), power units (W, mW, a.u.),
    and intensity units (W/m², W/cm², a.u.).

Author: Vibe (Mistral AI)
Repository: https://github.com/FSA-FR/SHFromScratch
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Beam import Beam
from Propagation import Propagation
from Visualization import plot_intensity, plot_phase


def run_normalization_test():
    """FR: Exécute les tests de normalisation."""
    print("\n" + "="*80)
    print("Example 7: Normalization Test")
    print("="*80)
    
    wavelength_nm = 633.0
    diameter_mm = 10.0
    num_points = 256
    propagation_distance_mm = 500.0
    
    # =========================================================================
    # 1. Test des normalisations de phase
    # =========================================================================
    print("\n--- Test des normalisations de phase ---")
    
    beam = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=diameter_mm,
        energy=1.0,
        num_points=num_points
    )
    
    # Générer une intensité gaussienne
    intensity = beam.generate_intensity(method="gaussian", sigma_mm=2.0)
    
    # Tester les deux normalisations
    for norm in ["RMS", "PV"]:
        print(f"\nNormalisation: {norm}")
        
        # Générer une phase avec la normalisation spécifiée
        phase = beam.generate_phase(
            method="random_zernike",
            n_modes=10,
            max_amplitude_nm=100.0,
            normalization=norm
        )
        
        # Calculer PV et RMS
        pv, rms = beam.compute_pv_rms(phase)
        print(f"  Phase générée: PV={pv:.2f} nm, RMS={rms:.2f} nm")
        
        # Vérification de la normalisation
        if norm == "PV":
            expected = 100.0  # max_amplitude_nm
            error = abs(pv - expected)
            print(f"  Vérification PV: attendu={expected:.2f} nm, erreur={error:.4f} nm")
            assert error < 1.0, f"Normalisation PV échouée: erreur={error:.4f} nm"
        elif norm == "RMS":
            expected = 100.0  # max_amplitude_nm
            error = abs(rms - expected)
            print(f"  Vérification RMS: attendu={expected:.2f} nm, erreur={error:.4f} nm")
            assert error < 1.0, f"Normalisation RMS échouée: erreur={error:.4f} nm"
        
        # Visualisation
        plot_phase(phase, diameter_mm, title=f"Phase ({norm} normalization)")
        plt.savefig(f'examples/output/example7_phase_{norm}.png', dpi=150, bbox_inches='tight')
        plt.close('all')
    
    # =========================================================================
    # 2. Test des unités d'énergie
    # =========================================================================
    print("\n--- Test des unités d'énergie ---")
    
    energy_units = ["J", "mJ", "a.u."]
    energy_values = [0.001, 1.0, 1.0]  # 1 mJ = 0.001 J
    
    for unit, value in zip(energy_units, energy_values):
        print(f"\nUnité: {unit}, Valeur: {value}")
        
        beam_energy = Beam(
            wavelength_nm=wavelength_nm,
            diameter_mm=diameter_mm,
            energy=value,
            energy_unit=unit,
            num_points=num_points
        )
        
        # Vérifier la conversion
        energy_J = beam_energy.get_energy_in_unit("J")
        energy_mJ = beam_energy.get_energy_in_unit("mJ")
        energy_a_u = beam_energy.get_energy_in_unit("a.u.")
        
        print(f"  Énergie en J: {energy_J:.6f}")
        print(f"  Énergie en mJ: {energy_mJ:.6f}")
        print(f"  Énergie en a.u.: {energy_a_u:.6f}")
        
        # Vérification de la cohérence
        if unit == "J":
            assert abs(energy_J - value) < 1e-10, f"Conversion J échouée"
        elif unit == "mJ":
            assert abs(energy_mJ - value) < 1e-10, f"Conversion mJ échouée"
        
        # Générer un faisceau et vérifier l'intensité
        intensity_energy = beam_energy.generate_intensity(method="gaussian", sigma_mm=2.0)
        print(f"  Somme intensité: {np.sum(intensity_energy):.6f} (doit correspondre à l'énergie)")
    
    # =========================================================================
    # 3. Test des unités de puissance
    # =========================================================================
    print("\n--- Test des unités de puissance ---")
    
    pulse_duration_s = 1e-9  # 1 ns
    power_units = ["W", "mW", "a.u."]
    power_values = [1.0, 1000.0, 1.0]  # 1 W = 1000 mW
    
    for unit, value in zip(power_units, power_values):
        print(f"\nUnité: {unit}, Valeur: {value}")
        
        beam_power = Beam(
            wavelength_nm=wavelength_nm,
            diameter_mm=diameter_mm,
            power=value,
            power_unit=unit,
            pulse_duration_s=pulse_duration_s,
            num_points=num_points
        )
        
        # Vérifier la conversion
        power_W = beam_power.get_power_in_unit("W")
        power_mW = beam_power.get_power_in_unit("mW")
        power_a_u = beam_power.get_power_in_unit("a.u.")
        
        print(f"  Puissance en W: {power_W:.6f}")
        print(f"  Puissance en mW: {power_mW:.6f}")
        print(f"  Puissance en a.u.: {power_a_u:.6f}")
        
        # Vérification de la cohérence
        if unit == "W":
            assert abs(power_W - value) < 1e-10, f"Conversion W échouée"
        elif unit == "mW":
            assert abs(power_mW - value) < 1e-10, f"Conversion mW échouée"
        
        # Vérifier l'énergie calculée
        energy_J = beam_power.get_energy_in_unit("J")
        expected_energy_J = value * 1e-3 if unit == "mW" else value * pulse_duration_s
        print(f"  Énergie calculée: {energy_J:.6f} J (attendu: {expected_energy_J:.6f} J)")
    
    # =========================================================================
    # 4. Test des unités d'intensité
    # =========================================================================
    print("\n--- Test des unités d'intensité ---")
    
    intensity_units = ["W/m2", "W/cm2", "a.u."]
    
    for unit in intensity_units:
        print(f"\nUnité: {unit}")
        
        beam_intensity = Beam(
            wavelength_nm=wavelength_nm,
            diameter_mm=diameter_mm,
            energy=1.0,
            intensity_unit=unit,
            num_points=num_points
        )
        
        # Générer une intensité
        intensity_test = beam_intensity.generate_intensity(method="gaussian", sigma_mm=2.0)
        
        # Convertir l'intensité
        intensity_W_m2 = beam_intensity.get_intensity_in_unit("W/m2")
        intensity_W_cm2 = beam_intensity.get_intensity_in_unit("W/cm2")
        
        print(f"  Intensité moyenne en W/m²: {intensity_W_m2:.6e}")
        print(f"  Intensité moyenne en W/cm²: {intensity_W_cm2:.6e}")
        
        # Vérification de la conversion
        if unit == "W/m2":
            assert abs(intensity_W_m2 - np.mean(intensity_test)) < 1e-10, f"Conversion W/m2 échouée"
        elif unit == "W/cm2":
            assert abs(intensity_W_cm2 - np.mean(intensity_test)) < 1e-10, f"Conversion W/cm2 échouée"
    
    # =========================================================================
    # 5. Test de la conversion énergie-puissance-intensité
    # =========================================================================
    print("\n--- Test de la conversion énergie-puissance-intensité ---")
    
    # Créer un faisceau avec puissance en W
    beam_conversion = Beam(
        wavelength_nm=wavelength_nm,
        diameter_mm=diameter_mm,
        power=1.0,
        power_unit="W",
        pulse_duration_s=1e-3,  # 1 ms
        num_points=num_points
    )
    
    # Vérifier les conversions
    energy_J = beam_conversion.get_energy_in_unit("J")
    expected_energy_J = 1.0 * 1e-3  # 1 W * 1 ms = 0.001 J
    print(f"Énergie: {energy_J:.6f} J (attendu: {expected_energy_J:.6f} J)")
    assert abs(energy_J - expected_energy_J) < 1e-10, "Conversion puissance→énergie échouée"
    
    intensity_W_m2 = beam_conversion.get_intensity_in_unit("W/m2")
    radius_m = diameter_mm / 2000  # mm → m
    area_m2 = np.pi * radius_m**2
    power_W = 1.0
    expected_intensity = power_W / area_m2
    print(f"Intensité: {intensity_W_m2:.6e} W/m² (attendu: {expected_intensity:.6e} W/m²)")
    assert abs(intensity_W_m2 - expected_intensity) < 1e-6, "Conversion puissance→intensité échouée"
    
    # =========================================================================
    # 6. Impact de la normalisation sur la propagation
    # =========================================================================
    print("\n--- Impact de la normalisation sur la propagation ---")
    
    for norm in ["RMS", "PV"]:
        print(f"\nNormalisation: {norm}")
        
        beam_norm = Beam(
            wavelength_nm=wavelength_nm,
            diameter_mm=diameter_mm,
            energy=1.0,
            num_points=128
        )
        
        # Générer une phase avec normalisation
        phase_norm = beam_norm.generate_phase(
            method="random_zernike",
            n_modes=10,
            max_amplitude_nm=100.0,
            normalization=norm
        )
        
        intensity_norm = beam_norm.generate_intensity(method="gaussian", sigma_mm=2.0)
        electric_field_norm = beam_norm.generate_electric_field(
            intensity=intensity_norm,
            phase=phase_norm,
            method="from_intensity_phase"
        )
        
        # Propager
        propagator_norm = Propagation(
            wavelength_nm=wavelength_nm,
            propagation_distance_mm=propagation_distance_mm,
            input_diameter_mm=diameter_mm,
            num_points=128,
            method="angular_spectrum"
        )
        
        propagated_field_norm = propagator_norm.propagate(electric_field_norm)
        
        # Extraire la phase propagée
        propagated_phase_norm = beam_norm.extract_phase_from_electric_field(propagated_field_norm)
        pv_prop, rms_prop = beam_norm.compute_pv_rms(propagated_phase_norm)
        
        print(f"  Phase propagée: PV={pv_prop:.2f} nm, RMS={rms_prop:.2f} nm")
        
        # Visualisation
        plot_phase(
            propagated_phase_norm,
            diameter_mm,
            title=f"Propagated Phase ({norm} normalization)"
        )
        plt.savefig(f'examples/output/example7_propagated_phase_{norm}.png', dpi=150, bbox_inches='tight')
        plt.close('all')
    
    # =========================================================================
    # 7. Test des unités avec propagation
    # =========================================================================
    print("\n--- Test des unités avec propagation ---")
    
    for energy_unit in ["J", "mJ", "a.u."]:
        print(f"\nUnité d'énergie: {energy_unit}")
        
        beam_unit = Beam(
            wavelength_nm=wavelength_nm,
            diameter_mm=diameter_mm,
            energy=1.0,
            energy_unit=energy_unit,
            num_points=128
        )
        
        # Générer un champ électrique
        electric_field_unit = beam_unit.generate_electric_field(method="gaussian", sigma_mm=2.0)
        
        # Propager
        propagator_unit = Propagation(
            wavelength_nm=wavelength_nm,
            propagation_distance_mm=propagation_distance_mm,
            input_diameter_mm=diameter_mm,
            num_points=128,
            method="angular_spectrum"
        )
        
        propagated_field_unit = propagator_unit.propagate(electric_field_unit)
        
        # Calculer l'intensité propagée
        intensity_propagated_unit = beam_unit.compute_intensity_from_electric_field(propagated_field_unit)
        energy_propagated = np.sum(intensity_propagated_unit)
        
        print(f"  Énergie propagée: {energy_propagated:.4f} (a.u.)")
        print(f"  Énergie initiale: {beam_unit.energy:.4f} ({energy_unit})")
    
    print("\n" + "="*80)
    print("Example 7 terminé avec succès !")
    print("Les images ont été sauvegardées dans examples/output/")
    print("="*80)


if __name__ == "__main__":
    import os
    os.makedirs('examples/output', exist_ok=True)
    run_normalization_test()

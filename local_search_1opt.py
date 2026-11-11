import ast
import itertools
import sys
import numpy as np

import blackbox as bb
import constraints as cst
import data as d
import windfarm_setting as wf

def generate_candidate_grid(lb, ub, buildable_zone, step):
    """Génère la grille des points situés dans la zone constructible."""
    xs = np.arange(lb[0], ub[0], step)
    ys = np.arange(lb[1], ub[1], step)
    candidates = []
    for x in xs:
        for y in ys:
            if cst.placing_constraint([x], [y], buildable_zone) <= 1e-5:
                candidates.append((float(x), float(y)))
    return candidates

def local_search_1opt(param_file_name, x0_file_name, grid_step_factor=2.5, max_cycles=3):
    """Améliore une solution existante en déplaçant chaque éolienne une par une."""
    # 1. Chargement des données du site et du terrain
    nb_wt, D, hub_height, scale_factor, power_curve, boundary_file, exclusion_zone_file, wind_speed, wind_direction = d.read_param_file(param_file_name)
    fmGROSS, WS, WD, _, _ = wf.site_setting(power_curve, D, hub_height, wind_speed, wind_direction, "results")
    WS_BB, WD_BB = d.read_csv_wind_data(wind_speed, wind_direction)
    lb, ub, boundary_shapely, exclusion_zones_shapely = wf.terrain_setting(boundary_file, exclusion_zone_file, scale_factor=scale_factor)
    buildable_zone = cst.buildable_zone(boundary_shapely, exclusion_zones_shapely)

    min_dist = 2.0 * D
    step = D * grid_step_factor

    # 2. Lecture du x0.txt généré précédemment
    x0_path = f"{x0_file_name}/x0.txt"
    with open(x0_path, "r") as f:
        coords = ast.literal_eval(f.read().strip())

    placed_x = list(coords[0::2])
    placed_y = list(coords[1::2])

    # 3. Calcul de l'AEP de référence
    _, current_aep = bb.aep_func(placed_x, placed_y, fmGROSS, WS_BB, WD_BB)
    current_aep = float(current_aep)
    print(f"AEP initiale chargée depuis {x0_path} : {current_aep:.4f} GWh")

    # 4. Génération de la grille des emplacements candidats
    candidates = generate_candidate_grid(lb, ub, buildable_zone, step)
    print(f"Grille de recherche : {len(candidates)} candidats (pas = {step:.1f} m)")

    # 5. Boucle d'amélioration 1-opt
    for cycle in range(max_cycles):
        improved = False
        print(f"\n--- Cycle de recherche locale {cycle + 1}/{max_cycles} ---")

        for i in range(len(placed_x)):
            orig_x, orig_y = placed_x[i], placed_y[i]
            
            # On isole les 29 autres turbines
            other_x = placed_x[:i] + placed_x[i+1:]
            other_y = placed_y[:i] + placed_y[i+1:]

            best_candidate = (orig_x, orig_y)
            best_eap = current_aep

            for cx, cy in candidates:
                # Contrainte d'espacement avec les 29 turbines restantes
                too_close = any(np.hypot(cx - ox, cy - oy) < min_dist for ox, oy in zip(other_x, other_y))
                if too_close:
                    continue

                test_x = other_x + [cx]
                test_y = other_y + [cy]
                _, eap = bb.aep_func(test_x, test_y, fmGROSS, WS_BB, WD_BB)
                eap = float(eap)

                # Seuil minimal de gain pour justifier un déplacement
                if eap > best_eap + 1e-4:
                    best_eap = eap
                    best_candidate = (cx, cy)

            # Si un meilleur point a été trouvé pour cette turbine
            if best_candidate != (orig_x, orig_y):
                print(f"Éolienne {i + 1:02d} déplacée : ({orig_x:.1f}, {orig_y:.1f}) -> ({best_candidate[0]:.1f}, {best_candidate[1]:.1f}) | AEP : {current_aep:.4f} -> {best_eap:.4f} GWh")
                placed_x[i] = best_candidate[0]
                placed_y[i] = best_candidate[1]
                current_aep = best_eap
                improved = True

        if not improved:
            print("\nOptimum local atteint sur la grille : aucun gain supplémentaire possible.")
            break

    # 6. Écriture du nouveau x0.txt optimisé
    X0 = list(itertools.chain(*zip(placed_x, placed_y)))
    with open(x0_path, "w") as f:
        f.write(str(X0))

    print(f"\nSolution 1-opt enregistrée dans {x0_path}")
    print(f"AEP finale avant NOMAD : {current_aep:.4f} GWh")

if __name__ == '__main__':
    param_path = sys.argv[1]
    x0_dir = sys.argv[2]
    local_search_1opt(param_path, x0_dir)

#python local_search_1opt.py instances/1/param.txt instances/1


#####On passe de 54.0129 à 54.0736 pour environ 1/2 heures de calculs

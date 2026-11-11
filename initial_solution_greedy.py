import itertools
import sys
import numpy as np

import blackbox as bb
import constraints as cst
import data as d
import windfarm_setting as wf

def generate_candidate_grid(lb, ub, buildable_zone, step):
    """Génère une grille régulière de coordonnées situées dans la zone constructible."""
    xs = np.arange(lb[0], ub[0], step)
    ys = np.arange(lb[1], ub[1], step)
    candidates = []
    
    for x in xs:
        for y in ys:
            # Distance <= 0 signifie que le point est dans la zone constructible
            if cst.placing_constraint([x], [y], buildable_zone) <= 1e-5:
                candidates.append((float(x), float(y)))
    return candidates

def initial_sol_greedy(param_file_name, x0_file_name, grid_step_factor=2.5):
    """Place les éoliennes une par une pour maximiser l'AEP à chaque ajout."""
    # 1. Chargement des données du site et du terrain
    nb_wt, D, hub_height, scale_factor, power_curve, boundary_file, exclusion_zone_file, wind_speed, wind_direction = d.read_param_file(param_file_name)
    fmGROSS, WS, WD, _, _ = wf.site_setting(power_curve, D, hub_height, wind_speed, wind_direction, "results")
    WS_BB, WD_BB = d.read_csv_wind_data(wind_speed, wind_direction)
    lb, ub, boundary_shapely, exclusion_zones_shapely = wf.terrain_setting(boundary_file, exclusion_zone_file, scale_factor=scale_factor)
    buildable_zone = cst.buildable_zone(boundary_shapely, exclusion_zones_shapely)

    min_dist = 2.0 * D
    step = D * grid_step_factor

    # 2. Échantillonnage des points candidats admissibles
    print(f"Génération de la grille avec un pas de {step:.1f} m...")
    candidates = generate_candidate_grid(lb, ub, buildable_zone, step)
    print(f"{len(candidates)} points candidats trouvés dans la zone constructible.")

    if len(candidates) < nb_wt:
        raise ValueError(f"Pas assez de points candidats ({len(candidates)}) pour {nb_wt} éoliennes. Réduis grid_step_factor.")

    placed_x = []
    placed_y = []

    print(f"\nPlacement glouton de {nb_wt} éoliennes...")

    # 3. Sélection itérative de chaque éolienne
    for k in range(nb_wt):
        best_candidate = None
        best_eap = -1.0

        # Filtrage : garder uniquement les candidats à au moins 2D de toutes les turbines déjà posées
        valid_candidates = []
        for cx, cy in candidates:
            too_close = any(np.hypot(cx - px, cy - py) < min_dist for px, py in zip(placed_x, placed_y))
            if not too_close:
                valid_candidates.append((cx, cy))

        if not valid_candidates:
            raise RuntimeError(f"Aucun emplacement valide restant pour poser l'éolienne {k + 1}/{nb_wt}.")

        # Première turbine : sélection du premier candidat admissible
        if k == 0:
            best_candidate = valid_candidates[0]
            _, best_eap = bb.aep_func([best_candidate[0]], [best_candidate[1]], fmGROSS, WS_BB, WD_BB)
        else:
            # Évaluation PyWake de l'impact de sillage pour chaque candidat admissible
            for cx, cy in valid_candidates:
                test_x = placed_x + [cx]
                test_y = placed_y + [cy]
                _, eap = bb.aep_func(test_x, test_y, fmGROSS, WS_BB, WD_BB)
                if float(eap) > best_eap:
                    best_eap = float(eap)
                    best_candidate = (cx, cy)

        placed_x.append(best_candidate[0])
        placed_y.append(best_candidate[1])
        candidates.remove(best_candidate)
        print(f"Éolienne {k + 1:02d}/{nb_wt} posée en ({best_candidate[0]:.1f}, {best_candidate[1]:.1f}) -> AEP cumulée : {best_eap:.3f} GWh")

    # 4. Sauvegarde de la solution initiale pour NOMAD
    X0 = list(itertools.chain(*zip(placed_x, placed_y)))
    output_path = f"{x0_file_name}/x0.txt"
    with open(output_path, "w") as f:
        f.write(str(X0))

    print(f"\nSolution initiale sauvegardée avec succès dans : {output_path}")
    return placed_x, placed_y

if __name__ == '__main__':
    param_path = sys.argv[1]
    x0_dir = sys.argv[2]
    initial_sol_greedy(param_path, x0_dir)

###python initial_solution_greedy.py instances/1_greedy/param.txt instances/1_greedy
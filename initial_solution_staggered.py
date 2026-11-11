"""
Génération d'une solution initiale par motif géométrique en quinconce (Staggered Grid).
Recherche de la meilleure orientation d'angle et d'espacement pour maximiser l'AEP.
"""

import itertools
import math
import sys
import numpy as np

import blackbox as bb
import constraints as cst
import data as d
import windfarm_setting as wf


def generate_staggered_candidates(lb, ub, buildable_zone, step_x, step_y, angle_deg):
    """Génère un maillage régulier en quinconce tourné d'un angle donné."""
    cx = (lb[0] + ub[0]) / 2.0
    cy = (lb[1] + ub[1]) / 2.0
    diag = math.hypot(ub[0] - lb[0], ub[1] - lb[1])

    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    u_vals = np.arange(-diag / 2.0, diag / 2.0, step_x)
    v_vals = np.arange(-diag / 2.0, diag / 2.0, step_y)

    candidates = []
    for row_idx, v in enumerate(v_vals):
        # Décalage d'un demi-pas une ligne sur deux (quinconce)
        offset = (step_x / 2.0) if (row_idx % 2 == 1) else 0.0
        for u in u_vals + offset:
            x = cx + u * cos_a - v * sin_a
            y = cy + u * sin_a + v * cos_a

            if lb[0] <= x <= ub[0] and lb[1] <= y <= ub[1]:
                if cst.placing_constraint([x], [y], buildable_zone) <= 1e-5:
                    candidates.append((float(x), float(y)))

    return candidates


def prune_to_nb_wt(candidates, nb_wt, min_dist):
    """Élague les points excédentaires en éliminant les plus proches voisins."""
    pts = list(candidates)
    if len(pts) < nb_wt:
        return None

    while len(pts) > nb_wt:
        coords = np.array(pts)
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_matrix = np.hypot(diff[:, :, 0], diff[:, :, 1])
        np.fill_diagonal(dist_matrix, np.inf)

        min_dists = np.min(dist_matrix, axis=1)
        worst_idx = int(np.argmin(min_dists))
        pts.pop(worst_idx)

    # Vérification stricte de l'interdistance minimale
    coords = np.array(pts)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.hypot(diff[:, :, 0], diff[:, :, 1])
    np.fill_diagonal(dist_matrix, np.inf)

    if np.min(dist_matrix) < min_dist - 1e-5:
        return None

    return pts


def staggered_grid_optimization(param_file_name, x0_file_name):
    # 1. Chargement des données du site
    nb_wt, D, hub_height, scale_factor, power_curve, boundary_file, exclusion_zone_file, wind_speed, wind_direction = d.read_param_file(param_file_name)
    fmGROSS, WS, WD, _, _ = wf.site_setting(power_curve, D, hub_height, wind_speed, wind_direction, "results")
    WS_BB, WD_BB = d.read_csv_wind_data(wind_speed, wind_direction)
    lb, ub, boundary_shapely, exclusion_zones_shapely = wf.terrain_setting(boundary_file, exclusion_zone_file, scale_factor=scale_factor)
    buildable_zone = cst.buildable_zone(boundary_shapely, exclusion_zones_shapely)

    min_dist = 2.0 * D

    # 2. Paramètres de balayage géométrique
    # Rapport dy/dx = sqrt(3)/2 pour un compactage triangulaire parfait
    step_factors = [2.2, 2.5, 2.8, 3.2, 3.6, 4.0]
    angles = list(range(0, 180, 15))

    best_aep = -float('inf')
    best_layout = None
    best_params = None
    evaluated_count = 0

    print(f"Recherche de motif en quinconce pour {nb_wt} éoliennes...")

    # 3. Exploration des combinaisons d'angle et d'espacement
    for sf in step_factors:
        step_x = D * sf
        step_y = step_x * (math.sqrt(3.0) / 2.0)

        for angle in angles:
            raw_candidates = generate_staggered_candidates(lb, ub, buildable_zone, step_x, step_y, angle)
            selected_pts = prune_to_nb_wt(raw_candidates, nb_wt, min_dist)

            if selected_pts is None:
                continue

            # Évaluation PyWake de la configuration complète
            tx = [p[0] for p in selected_pts]
            ty = [p[1] for p in selected_pts]
            _, aep = bb.aep_func(tx, ty, fmGROSS, WS_BB, WD_BB)
            aep = float(aep)
            evaluated_count += 1

            if aep > best_aep:
                best_aep = aep
                best_layout = selected_pts
                best_params = (sf, angle)
                print(f"  Configuration retenue : pas={sf:.1f}D, angle={angle:03d}° | AEP : {best_aep:.4f} GWh")

    if best_layout is None:
        raise RuntimeError("Aucune grille admissible n'a pu contenir le nombre requis d'éoliennes.")

    # 4. Sauvegarde de la meilleure disposition dans x0.txt
    placed_x = [p[0] for p in best_layout]
    placed_y = [p[1] for p in best_layout]
    X0 = list(itertools.chain(*zip(placed_x, placed_y)))

    x0_path = f"{x0_file_name}/x0.txt"
    with open(x0_path, "w") as f:
        f.write(str(X0))

    print(f"\n--- Résultat optimal de la grille en quinconce ---")
    print(f"Configurations valides testées : {evaluated_count}")
    print(f"Paramètres optimaux : pas = {best_params[0]:.1f}D, orientation = {best_params[1]}°")
    print(f"AEP obtenue : {best_aep:.4f} GWh")
    print(f"Solution enregistrée dans : {x0_path}")


if __name__ == '__main__':
    param_path = sys.argv[1]
    x0_dir = sys.argv[2]
    staggered_grid_optimization(param_path, x0_dir)

## python initial_solution_staggered.py instances/1_staggered/param.txt instances/1_staggered


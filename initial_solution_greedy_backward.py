"""
Génération d'une solution initiale par glouton destructif (Backward Greedy).
Sature la zone constructible puis élimine itérativement les turbines les plus pénalisantes.
"""

import itertools
import math
import sys
import numpy as np

import blackbox as bb
import constraints as cst
import data as d
import windfarm_setting as wf


def pack_dense_turbines(lb, ub, buildable_zone, min_dist, target_count, D):
    """Génère une configuration dense admissible respectant strictement la distance 2D."""
    # Test de pas de grille décroissants pour garantir le nombre voulu d'éoliennes
    for step_factor in [2.3, 2.15, 2.05]:
        step = D * step_factor
        xs = np.arange(lb[0], ub[0], step)
        ys = np.arange(lb[1], ub[1], step)

        candidates = []
        for x in xs:
            for y in ys:
                if cst.placing_constraint([x], [y], buildable_zone) <= 1e-5:
                    candidates.append((float(x), float(y)))

        placed = []
        for cx, cy in candidates:
            if all(math.hypot(cx - px, cy - py) >= min_dist for px, py in placed):
                placed.append((cx, cy))
                if len(placed) == target_count:
                    return placed

    if len(placed) < target_count:
        print(f"Attention : seulement {len(placed)} turbines denses générées sur {target_count} souhaitées.")
    return placed


def backward_greedy(param_file_name, x0_file_name, extra_turbines=6):
    """Élimine une à une les éoliennes jusqu'à obtenir nb_wt turbines."""
    # 1. Chargement des données du site
    nb_wt, D, hub_height, scale_factor, power_curve, boundary_file, exclusion_zone_file, wind_speed, wind_direction = d.read_param_file(param_file_name)
    fmGROSS, WS, WD, _, _ = wf.site_setting(power_curve, D, hub_height, wind_speed, wind_direction, "results")
    WS_BB, WD_BB = d.read_csv_wind_data(wind_speed, wind_direction)
    lb, ub, boundary_shapely, exclusion_zones_shapely = wf.terrain_setting(boundary_file, exclusion_zone_file, scale_factor=scale_factor)
    buildable_zone = cst.buildable_zone(boundary_shapely, exclusion_zones_shapely)

    min_dist = 2.0 * D
    start_count = nb_wt + extra_turbines

    # 2. Remplissage initial sur-saturé
    print(f"Génération d'une configuration dense initiale de {start_count} éoliennes...")
    placed = pack_dense_turbines(lb, ub, buildable_zone, min_dist, start_count, D)

    if len(placed) < nb_wt:
        raise ValueError(f"Impossible de placer le minimum requis de {nb_wt} turbines sur ce terrain.")

    # 3. Boucle destructive itérative (Backward)
    print(f"\nDébut du Backward Greedy : élagage de {len(placed)} -> {nb_wt} éoliennes...")

    while len(placed) > nb_wt:
        current_len = len(placed)
        best_remaining_aep = -float('inf')
        worst_turbine_idx = -1

        print(f"\n--- Étape : passage de {current_len} à {current_len - 1} éoliennes ({current_len} candidats testés) ---")

        # Test du retrait de chaque éolienne individuellement
        for i in range(current_len):
            sub_layout = placed[:i] + placed[i + 1:]
            tx = [p[0] for p in sub_layout]
            ty = [p[1] for p in sub_layout]

            _, aep = bb.aep_func(tx, ty, fmGROSS, WS_BB, WD_BB)
            aep = float(aep)

            # On cherche le sous-ensemble qui conserve la production la plus haute
            if aep > best_remaining_aep:
                best_remaining_aep = aep
                worst_turbine_idx = i

        # Suppression définitive de la pire éolienne
        removed_pt = placed.pop(worst_turbine_idx)
        print(f"  -> Retrait éolienne {worst_turbine_idx + 1:02d} en ({removed_pt[0]:.1f}, {removed_pt[1]:.1f})")
        print(f"     AEP restante du parc : {best_remaining_aep:.4f} GWh")

    # 4. Évaluation et enregistrement de la solution finale à 30 éoliennes
    placed_x = [p[0] for p in placed]
    placed_y = [p[1] for p in placed]
    _, final_aep = bb.aep_func(placed_x, placed_y, fmGROSS, WS_BB, WD_BB)
    final_aep = float(final_aep)

    X0 = list(itertools.chain(*zip(placed_x, placed_y)))
    x0_path = f"{x0_file_name}/x0.txt"
    with open(x0_path, "w") as f:
        f.write(str(X0))

    print(f"\n--- Résultat final Backward Greedy ---")
    print(f"Nombre final d'éoliennes : {len(placed)}")
    print(f"AEP finale obtenue : {final_aep:.4f} GWh")
    print(f"Solution enregistrée dans : {x0_path}")


if __name__ == '__main__':
    param_path = sys.argv[1]
    x0_dir = sys.argv[2]
    backward_greedy(param_path, x0_dir)

###python initial_solution_greedy_backward.py instances/1_greedy_backward/param.txt instances/1_greedy_backward
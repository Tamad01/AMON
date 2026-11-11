import itertools
import math
import sys
import numpy as np
import pandas as pd

import blackbox as bb
import constraints as cst
import data as d
import windfarm_setting as wf

def get_dominant_wind_direction(wind_direction_file):
    """Extrait la direction du vent dominant (en degrés) depuis le fichier CSV."""
    try:
        df = pd.read_csv(wind_direction_file, sep=None, engine='python')
        col_dir = df.columns[0]
        col_freq = df.columns[1]
        dominant_wd = float(df.loc[df[col_freq].idxmax(), col_dir])
        return dominant_wd
    except Exception:
        data = np.genfromtxt(wind_direction_file, delimiter=',', skip_header=1)
        idx = np.argmax(data[:, 1])
        return float(data[idx, 0])

def generate_candidate_grid(lb, ub, buildable_zone, step):
    """Génère la grille des positions admissibles sur le terrain."""
    xs = np.arange(lb[0], ub[0], step)
    ys = np.arange(lb[1], ub[1], step)
    candidates = []
    for x in xs:
        for y in ys:
            if cst.placing_constraint([x], [y], buildable_zone) <= 1e-5:
                candidates.append((float(x), float(y)))
    return candidates

def upwind_greedy(param_file_name, x0_file_name, grid_step_factor=2.5):
    # 1. Chargement des paramètres du site
    nb_wt, D, hub_height, scale_factor, power_curve, boundary_file, exclusion_zone_file, wind_speed, wind_direction = d.read_param_file(param_file_name)
    fmGROSS, WS, WD, _, _ = wf.site_setting(power_curve, D, hub_height, wind_speed, wind_direction, "results")
    WS_BB, WD_BB = d.read_csv_wind_data(wind_speed, wind_direction)
    lb, ub, boundary_shapely, exclusion_zones_shapely = wf.terrain_setting(boundary_file, exclusion_zone_file, scale_factor=scale_factor)
    buildable_zone = cst.buildable_zone(boundary_shapely, exclusion_zones_shapely)

    min_dist = 2.0 * D
    step = D * grid_step_factor

    # 2. Identification du vent dominant et vecteur amont
    dominant_wd = get_dominant_wind_direction(wind_direction)
    rad = math.radians(dominant_wd)
    # En météo : theta est la direction d'où vient le vent (0° = Nord/+Y, 90° = Est/+X)
    ux = math.sin(rad)
    uy = math.cos(rad)
    print(f"Direction dominante du vent : {dominant_wd:.1f}°")

    # 3. Grille candidate
    candidates = generate_candidate_grid(lb, ub, buildable_zone, step)
    placed_x, placed_y = [], []
    current_aep = 0.0

    print(f"Placement glouton orienté vent de {nb_wt} éoliennes...")

    # 4. Placement itératif
    for k in range(nb_wt):
        valid_candidates = [
            (cx, cy) for cx, cy in candidates
            if all(np.hypot(cx - px, cy - py) >= min_dist for px, py in zip(placed_x, placed_y))
        ]

        if not valid_candidates:
            raise ValueError(f"Impossible de placer l'éolienne {k + 1} : terrain saturé.")

        if k == 0:
            # Turbine 1 : sélection du point le plus en amont face au vent (projection max)
            best_candidate = max(valid_candidates, key=lambda pt: pt[0] * ux + pt[1] * uy)
            test_x = [best_candidate[0]]
            test_y = [best_candidate[1]]
            _, best_aep = bb.aep_func(test_x, test_y, fmGROSS, WS_BB, WD_BB)
        else:
            best_candidate = None
            best_aep = -float('inf')

            for cx, cy in valid_candidates:
                test_x = placed_x + [cx]
                test_y = placed_y + [cy]
                _, aep = bb.aep_func(test_x, test_y, fmGROSS, WS_BB, WD_BB)
                aep = float(aep)

                if aep > best_aep:
                    best_aep = aep
                    best_candidate = (cx, cy)

        placed_x.append(best_candidate[0])
        placed_y.append(best_candidate[1])
        current_aep = float(best_aep)
        print(f"Éolienne {k + 1:02d}/{nb_wt} posée en ({best_candidate[0]:.1f}, {best_candidate[1]:.1f}) | AEP : {current_aep:.4f} GWh")

    # 5. Sauvegarde de la solution initiale optimisée
    X0 = list(itertools.chain(*zip(placed_x, placed_y)))
    x0_path = f"{x0_file_name}/x0.txt"
    with open(x0_path, "w") as f:
        f.write(str(X0))

    print(f"\nSolution initiale amont enregistrée dans {x0_path}")
    print(f"AEP de départ : {current_aep:.4f} GWh")

if __name__ == '__main__':
    param_path = sys.argv[1]
    x0_dir = sys.argv[2]
    upwind_greedy(param_path, x0_dir)

### python initial_solution_test2.py instances/1_bis/param.txt instances/1_bis
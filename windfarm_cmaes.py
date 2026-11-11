"""
Ce script exécute l'optimisation par stratégie d'évolution CMA-ES 
(Covariance Matrix Adaptation Evolution Strategy) sur les instances du parc éolien.
"""

import ast
import os
import sys
import time
import cma
import matplotlib.pyplot as plt
import numpy as np

import blackbox as bb
import constraints as cst
import data as d
import plotting_functions as plot_f
import windfarm_setting as wf


def CMAES_execution(param_file_name):
    """Exécute l'algorithme CMA-ES sur une instance donnée.

    Paramètres
    ----------
    param_file_name : str
        Chemin vers le fichier param.txt de l'instance.
    """
    # 1. Extraction propre du nom de l'instance et des chemins
    normalized_path = os.path.normpath(param_file_name)
    instance_name = normalized_path.split(os.sep)[-2]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_instance_dir = os.path.join(script_dir, 'instances_results', instance_name)
    os.makedirs(results_instance_dir, exist_ok=True)

    # 2. Chargement des paramètres du parc et du terrain
    nb_wt, D, hub_height, scale_factor, power_curve, boundary_file, exclusion_zone_file, wind_speed, wind_direction = d.read_param_file(param_file_name)
    nb_it = 400
    fmGROSS, WS, WD, max_index, max_ws = wf.site_setting(power_curve, D, hub_height, wind_speed, wind_direction, results_instance_dir + "/")
    WS_BB, WD_BB = d.read_csv_wind_data(wind_speed, wind_direction)
    lb, ub, boundary_shapely, exclusion_zones_shapely = wf.terrain_setting(boundary_file, exclusion_zone_file, scale_factor=scale_factor)
    buildable_zone = cst.buildable_zone(boundary_shapely, exclusion_zones_shapely)

    # 3. Chargement de la solution initiale (x0.txt)
    print(f"\n--------- Running CMA-ES instance {instance_name} ---------")
    x0_path = os.path.join("instances", instance_name, "x0.txt")
    with open(x0_path, "r") as file:
        content = file.read().splitlines()
    X0 = [float(coord) for coord in ast.literal_eval(content[0])]

    # 4. Fonction d'évaluation avec gestion stricte des pénalités
    def evaluate_layout(x_candidate):
        x_coords = list(x_candidate[0::2])
        y_coords = list(x_candidate[1::2])

        if not cst.checking_same_coords(x_coords, y_coords):
            return 1e12

        # Contrainte 1 : Zone constructible et zones d'exclusion
        sum_dist = cst.placing_constraint(x_coords, y_coords, buildable_zone)
        if sum_dist > 1e-5:
            return 1e9 + float(sum_dist) * 1e5

        # Contrainte 2 : Espacement minimal (2D)
        s_d = cst.spacing_constraint_min(x_coords, y_coords, D)
        if s_d > 1e-5:
            return 1e8 + float(s_d) * 1e5

        # Géométrie valide : calcul de l'AEP avec PyWake
        _, eap = bb.aep_func(x_coords, y_coords, fmGROSS, WS_BB, WD_BB)
        return -float(eap) * 1e6

    # 5. Configuration de CMA-ES
    lower_bounds = lb * nb_wt
    upper_bounds = ub * nb_wt
    # Écart-type réduit à 5 m pour un micro-ajustement local autour de x0
    sigma0 = 5.0

    cma_options = {
        'bounds': [lower_bounds, upper_bounds],
        'maxfevals': nb_it,
        'seed': 42,
        'verbose': -1,
        'verb_disp': 0,
        'verb_log': 0
    }

    es = cma.CMAEvolutionStrategy(X0, sigma0, cma_options)

    # 6. Boucle d'optimisation
    print("...Launching CMA-ES optimization...")
    t_start = time.time()

    eval_count = 0
    best_obj = float('inf')
    best_x = list(X0)
    best_eval = 1

    eval_history = []
    obj_history = []
    stats_lines = []

    init_obj = evaluate_layout(X0)
    if init_obj < 1e8:
        best_obj = init_obj
        print(f"Point initial x0 valide | AEP : {-best_obj * 1e-6:.4f} GWh")
    else:
        print("Point initial x0 avec contraintes actives.")

    while not es.stop():
        solutions = es.ask()
        fitness_list = []
        budget_reached = False

        for candidate in solutions:
            eval_count += 1
            cost = evaluate_layout(candidate)
            fitness_list.append(cost)

            if cost < best_obj:
                best_obj = cost
                best_x = list(candidate)
                best_eval = eval_count
                print(f"  [Eval {eval_count:04d}/{nb_it}] Nouveau record : {-best_obj * 1e-6:.4f} GWh")

            elapsed = time.time() - t_start
            eval_history.append(eval_count)
            obj_history.append(best_obj if best_obj < 1e8 else cost)
            stats_lines.append(f"{eval_count} {cost} 0.0 - {elapsed:.2f}\n")

            if eval_count >= nb_it:
                budget_reached = True
                break

        # Interrompt avant tell() si le budget s'arrête en milieu de génération
        if budget_reached:
            break

        es.tell(solutions, fitness_list)

    t_end = time.time()
    t_CMA = t_end - t_start
    print("...Ending CMA-ES optimization...")

    # 7. Sauvegarde des statistiques
    stat_file_name = os.path.join(results_instance_dir, f"cmaes_result_{instance_name}.0.txt")
    with open(stat_file_name, "w") as f_stat:
        f_stat.writelines(stats_lines)

    # 8. Courbe de convergence
    conv_plot_path = os.path.join(results_instance_dir, f"convergence_plot_{instance_name}.png")
    plt.clf()
    try:
        plot_f.plot_result_nomad(
            np.array(eval_history),
            np.array(obj_history),
            best_eval,
            best_obj,
            conv_plot_path
        )
    except Exception:
        plt.figure(figsize=(8, 5))
        plt.plot(eval_history, -np.array(obj_history) * 1e-6, color='tab:blue', lw=2)
        plt.xlabel("Évaluations")
        plt.ylabel("Meilleure AEP (GWh)")
        plt.title(f"Convergence CMA-ES - Instance {instance_name}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(conv_plot_path)
        plt.close()

    # 9. Tracé du layout final
    obj_function_value = -best_obj * 1e-6
    cg, eap = bb.aep_func(best_x[0::2], best_x[1::2], fmGROSS, max_ws, max_index * 10)
    layout_plot_path = os.path.join(results_instance_dir, f"layout_{instance_name}.png")

    plot_f.plot_terrain(
        best_x[0::2],
        best_x[1::2],
        "EAP",
        "GWh",
        obj_function_value,
        nb_wt,
        ub,
        boundary_shapely,
        exclusion_zones_shapely,
        max_index=max_index,
        max_ws=max_ws,
        cg=cg,
        plot_flow_map=False,
        save=True,
        save_name=layout_plot_path
    )

    # 10. Sauvegarde des coordonnées optimales
    x_best_path = os.path.join(results_instance_dir, "x_best.txt")
    with open(x_best_path, "w") as f_out:
        f_out.write(str(best_x))

    print(f"Best objective function value : {obj_function_value:.4f} GWh")
    print(f"CMA-ES execution time : {t_CMA:.2f} secondes")
    print(f"--------- Ending instance {instance_name} ---------")

    # Récapitulatif global
    output_summary = "instances_results/output_instances_cmaes.txt"
    with open(output_summary, "a") as f:
        f.write(f"{instance_name} {obj_function_value:.6f} {t_CMA:.2f}\n")


def run_instance_process():
    """Lance l'optimisation CMA-ES sur 1_greedy, 1_staggered et 1_wind."""
    summary_file = "instances_results/output_instances_cmaes.txt"
    if os.path.exists(summary_file):
        os.remove(summary_file)

    staggered_name = "1_staggerd" if os.path.exists("instances/1_staggerd") else "1_staggered"
    instances_to_run = ["1_greedy", staggered_name, "1_wind"]
    instance_failed = []

    for inst in instances_to_run:
        param_path = f"instances/{inst}/param.txt"
        if not os.path.exists(param_path):
            print(f"Fichier introuvable pour l'instance {inst} : {param_path}")
            instance_failed.append(inst)
            continue

        try:
            CMAES_execution(param_file_name=param_path)
        except Exception as e:
            print(f"Erreur sur l'instance {inst} : {e}")
            instance_failed.append(inst)

    if instance_failed:
        print("Instances échouées :", instance_failed)
    print("\n--------- Running instances completed ---------")


if __name__ == '__main__':
    run_instance_process()
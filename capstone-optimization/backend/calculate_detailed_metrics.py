import numpy as np
from run_publication_45_scenarios import execute_all_publication_benchmarks

results = execute_all_publication_benchmarks()

total_scenarios = sum(len(evals) for evals in results.values())
print(f"\n=========================================================================")
print(f" VERIFICATION: Total Scenarios Count = {total_scenarios} (45 Expected)")
print(f"=========================================================================\n")

for top_name, top_evals in results.items():
    n_scenarios = len(top_evals)
    init_peaks = np.array([ev['init_peak'] for ev in top_evals], dtype=float)
    pso_peaks = np.array([ev['pso_peak'] for ev in top_evals], dtype=float)
    qaoa_peaks = np.array([ev['qaoa_peak'] for ev in top_evals], dtype=float)

    # Reduction % formula: ((Initial Peak - Method Peak) / Initial Peak) * 100
    qaoa_red_pcts = ((init_peaks - qaoa_peaks) / init_peaks) * 100.0
    pso_red_pcts = ((init_peaks - pso_peaks) / init_peaks) * 100.0

    qaoa_median_red = float(np.median(qaoa_red_pcts))
    pso_median_red = float(np.median(pso_red_pcts))

    qaoa_min_red = float(np.min(qaoa_red_pcts))
    qaoa_max_red = float(np.max(qaoa_red_pcts))

    qaoa_lt_init = int(np.sum(qaoa_peaks < init_peaks - 1e-5))
    qaoa_lt_pso = int(np.sum(qaoa_peaks < pso_peaks - 1e-5))
    qaoa_eq_pso = int(np.sum(np.abs(qaoa_peaks - pso_peaks) <= 1e-5))
    qaoa_gt_pso = int(np.sum(qaoa_peaks > pso_peaks + 1e-5))

    print("=" * 105)
    print(f" TOPOLOGY: {top_name.upper()} ({n_scenarios} SCENARIOS VERIFIED)")
    print("=" * 105)
    print(f"{'Scenario Name':<42} | {'Init Peak':<10} | {'PSO Peak':<10} | {'QAOA Peak':<10} | {'PSO Red %':<11} | {'QAOA Red %':<11}")
    print("-" * 105)
    for idx, ev in enumerate(top_evals):
        print(f"{ev['scenario_name']:<42} | {init_peaks[idx]:<9.2f}% | {pso_peaks[idx]:<9.2f}% | {qaoa_peaks[idx]:<9.2f}% | {pso_red_pcts[idx]:<10.2f}% | {qaoa_red_pcts[idx]:<10.2f}%")

    print("\nRAW LISTS FOR MANUAL VERIFICATION:")
    print(f"  • 15 PSO Red% Values  : {[round(x, 2) for x in pso_red_pcts.tolist()]}")
    print(f"  • 15 QAOA Red% Values : {[round(x, 2) for x in qaoa_red_pcts.tolist()]}")

    print("\nCALCULATED METRICS:")
    print(f"  1. QAOA Median Peak-Reduction %       : {qaoa_median_red:.2f}%")
    print(f"  2. PSO Median Peak-Reduction %        : {pso_median_red:.2f}%")
    print(f"  3. QAOA Minimum Peak-Reduction %      : {qaoa_min_red:.2f}%")
    print(f"  4. QAOA Maximum Peak-Reduction %      : {qaoa_max_red:.2f}%")
    print(f"  5. Scenarios QAOA Peak < Initial Peak : {qaoa_lt_init} / {n_scenarios}")
    print(f"  6. Scenarios QAOA Peak < PSO Peak     : {qaoa_lt_pso} / {n_scenarios}")
    print(f"  7. Scenarios QAOA Peak == PSO Peak    : {qaoa_eq_pso} / {n_scenarios}")
    print(f"  8. Scenarios QAOA Peak > PSO Peak     : {qaoa_gt_pso} / {n_scenarios}\n")

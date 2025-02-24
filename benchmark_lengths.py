import subprocess
import csv
from datetime import datetime
import os
import numpy as np
import pandas as pd
import torch
# Set random seeds for reproducibility
seed = 100
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def run_benchmarks():
    # Sequence lengths to test (100, 316, 1000, 3162, 10000)
    kv_lengths = [int(x) for x in np.logspace(2, 5, num=12)]
    batch_size = 64
    
    # Models to benchmark
    models = ['CacheCompressed', 'CacheDecompressed', 'Absorbed_CacheCompressed', 
    'Absorbed_CacheCompressed_MoveElision', 'AbsorbedMaterialized_CacheCompressed_MoveElision']
    
    # Create results directory if it doesn't exist
    base_results_dir = "results"
    results_dir = os.path.join(base_results_dir, f"bsz-{batch_size}")
    os.makedirs(results_dir, exist_ok=True)

    # Create CSV files for each model
    for model in models:
        model_name = model.lower()
        results = []
        
        # Run benchmarks for this model
        for length in kv_lengths:
            print(f"\nRunning benchmark for {model} with sequence length {length}")
            cmd = [
                'python3',
                'mla/benchmark.py',
                model,
                str(length),
                f'--bsz={batch_size}',
                '--config=mla/config.json',
                '--min_run_time=1.0'
            ]
            
            # Run the benchmark and get the results
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.stdout.strip():
                # Parse the output string
                output_lines = result.stdout.strip().split('\n')
                parsed_result = {}
                for line in output_lines:
                    key, value = [x.strip() for x in line.split(':')]
                    parsed_result[key] = value

                # Create dictionary with desired columns
                result_dict = {
                    'bsz': batch_size,
                    'kv_length': length,
                    'Cache_Size': int(parsed_result['Cache_Size']),
                    'Mean': float(parsed_result['Mean']),
                    'Median': float(parsed_result['Median']),
                    'P25': float(parsed_result['P25']),
                    'P75': float(parsed_result['P75'])
                }
                results.append(result_dict)
            
            print(f"Completed benchmark for length {length}")
        
        # Convert results to DataFrame and save
        df = pd.DataFrame(results)
        csv_filename = os.path.join(results_dir, f'benchmark_results_{model_name}.csv')
        df.to_csv(csv_filename, index=False)
        print(f"Results saved to {csv_filename}")

if __name__ == "__main__":
    print("Starting benchmarks...")
    run_benchmarks()
    print("\nBenchmarks complete!") 
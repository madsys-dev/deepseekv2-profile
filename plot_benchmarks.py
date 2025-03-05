import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_benchmarks(batch_size=1):
    plt.style.use('default')
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Define results directory based on batch size
    results_dir = os.path.join('results', f'bsz-{batch_size}')
    
    # Read all CSV files from the batch-specific directory
    df_simple = pd.read_csv(os.path.join(results_dir, 'benchmark_results_simpleattention.csv'))
    df_simple_compressed = pd.read_csv(os.path.join(results_dir, 'benchmark_results_simplecompressedattention.csv'))
    df_simple_absorbed = pd.read_csv(os.path.join(results_dir, 'benchmark_results_simpleabsorbedattention.csv'))

    # Plot simple attention data
    plt.plot(df_simple['kv_length'], df_simple['Median'], 
            marker='o', 
            linewidth=2,
            markersize=8,
            color='blue',
            label='Uncompressed KV Cache MLA')
    
    # Add error bands for simple attention
    plt.fill_between(df_simple['kv_length'], 
                    df_simple['P25'], 
                    df_simple['P75'], 
                    alpha=0.2,
                    color='blue')
    
    # Plot simple compressed attention data
    plt.plot(df_simple_compressed['kv_length'], df_simple_compressed['Median'], 
            marker='s',  # square marker to differentiate
            linewidth=2,
            markersize=8,
            color='red',
            label='Compressed KV Cache MLA')
    
    # Add error bands for simple compressed attention
    plt.fill_between(df_simple_compressed['kv_length'], 
                    df_simple_compressed['P25'], 
                    df_simple_compressed['P75'], 
                    alpha=0.2,
                    color='red')

    # Plot simple absorbed attention data
    plt.plot(df_simple_absorbed['kv_length'], df_simple_absorbed['Median'], 
            marker='^',  # triangle marker to differentiate
            linewidth=2,
            markersize=8,
            color='green',
            label='Absorbed MLA')
    
    # Add error bands for simple absorbed attention
    plt.fill_between(df_simple_absorbed['kv_length'], 
                    df_simple_absorbed['P25'], 
                    df_simple_absorbed['P75'], 
                    alpha=0.2,
                    color='green')

    # Configure plot
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('KV Cache Length (log)', fontsize=12)
    plt.ylabel('Time (seconds) per layer (log)', fontsize=12)
    plt.title(f'Attention Performance Comparison batch size = {batch_size}', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    # Save plot in the batch-specific directory
    plot_filename = os.path.join('results', f'bsz-{batch_size}', 'benchmark_comparison.png')
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # You can specify different batch sizes here
    plot_benchmarks(batch_size=32)
    print(f"Plot saved") 
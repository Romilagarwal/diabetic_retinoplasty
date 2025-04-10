import os
import subprocess
import time

def run_script(script_name):
    """Run a Python script and print its output"""
    print(f"\n{'=' * 50}")
    print(f"Running {script_name}...")
    print(f"{'=' * 50}\n")
    
    start_time = time.time()
    result = subprocess.run(['python', script_name], capture_output=True, text=True)
    elapsed_time = time.time() - start_time
    
    print(f"Output from {script_name}:")
    print(result.stdout)
    
    if result.stderr:
        print(f"Errors from {script_name}:")
        print(result.stderr)
    
    print(f"\nCompleted in {elapsed_time:.2f} seconds")
    return result.returncode == 0

def main():
    """Execute all visualization scripts in sequence"""
    scripts = [
        'create_architecture_diagram.py',
        'create_performance_metrics.py',
        'create_explainability_visualizations.py',
        'create_summary_dashboard.py',
        'create_publication_tables.py'
    ]
    
    success_count = 0
    for script in scripts:
        if run_script(script):
            success_count += 1
    
    print(f"\n{'=' * 50}")
    print(f"Visualization generation complete: {success_count}/{len(scripts)} scripts succeeded")
    print(f"{'=' * 50}")

if __name__ == "__main__":
    # Make sure we're in the right directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
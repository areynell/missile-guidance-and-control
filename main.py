import os


from simulation import run_simulation
from visualization import plot_metrics, animate_trajectories, animate_6dof_missile

def main():
    # Ensure media directory exists for saving output animations/plots
    os.makedirs('media', exist_ok=True)

    # Run guidance and control simulation loop
    missile_log, target_log, intercepted = run_simulation()

    # Generate and display plots and 3D animations
    print("Generating post-flight interception metrics...")
    plot_metrics(missile_log, target_log)

    print("Generating trajectory animation...")
    animate_trajectories(missile_log, target_log)

    print("Generating 6-DOF attitude and force animation...")
    animate_6dof_missile(missile_log, target_log, length=5.0, diameter=0.41)

if __name__ == "__main__":
    main()
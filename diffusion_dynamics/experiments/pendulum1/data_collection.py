from diffusion_dynamics.systems import Pendulum, PendulumParams
from diffusion_dynamics.simulator import simulate_dynamical_system

import multiprocessing
import time

def worker(task_queue):
    while True:
        task = task_queue.get()  # Get a task from the queue
        if task is None:  # Sentinel value to signal termination
            break
        
        print(f"Worker {multiprocessing.current_process().name} processing: {task}")
        time.sleep(1)  # Simulate work
        print(f"Worker {multiprocessing.current_process().name} finished: {task}")

def generate_data(n_workers=10):
    num_workers = 4  # Number of worker processes
    task_queue = multiprocessing.Queue()

    # Start worker processes
    workers = []
    for _ in range(num_workers):
        p = multiprocessing.Process(target=worker, args=(task_queue,))
        p.start()
        workers.append(p)

if __name__ == "__main__":
    N_training_trajectories = 1000
    
    
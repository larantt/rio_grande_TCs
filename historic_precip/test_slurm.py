from dask.distributed import Client, LocalCluster
import logging
import time

# Setup logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Print current environment paths
import subprocess
logger.info(f"Current Python path: {subprocess.check_output(['which', 'python']).decode().strip()}")
logger.info(f"Current Python version: {subprocess.check_output(['python', '--version']).decode().strip()}")

# Simplified test script
if __name__ == "__main__":
    # Setting up a LocalCluster using all available cores (or you can specify a number of cores)
    cluster = LocalCluster(n_workers=10, threads_per_worker=1)
    client = Client(cluster)
    
    logger.info(f'Dask dashboard available at {client.dashboard_link}')
    
    # Give some time to set up and connect
    time.sleep(5)
    
    active_workers = client.scheduler_info()['workers']
    logger.info(f"Active workers: {active_workers}")
    
    # Check and log the status of the workers
    if not active_workers:
        logger.error("No workers found. Check error logs for details.")
        raise RuntimeError("No workers found. Check error logs for details.")
    
    # Test task to see if workers are starting correctly
    def test(x):
        return x + 1
    
    future = client.submit(test, 10)
    result = future.result()
    logger.info(f'Test result: {result}')
    
    # Close the client and cluster
    client.close()
    cluster.close()
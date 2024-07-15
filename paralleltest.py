import numpy as np
import multiprocessing as mp
import time

def matrix_multiplication(chunk_A, B):
    """ Perform matrix multiplication of chunk_A and B. """
    return np.dot(chunk_A, B)

if __name__ == '__main__':
    # Define matrices A and B
    A = np.random.rand(1000, 1000)
    B = np.random.rand(1000, 1000)
    
    # Without multiprocessing (sequential)
    start_time = time.time()
    result_sequential = np.dot(A, B)
    end_time = time.time()
    time_sequential = end_time - start_time
    
    print(f"Time taken without multiprocessing: {time_sequential:.4f} seconds")
    
    # With multiprocessing
    num_processes = mp.cpu_count()
    pool = mp.Pool(processes=num_processes)
    
    start_time = time.time()
    
    # Split matrix A into chunks
    chunk_size = len(A) // num_processes
    chunks_A = [A[i:i + chunk_size] for i in range(0, len(A), chunk_size)]
    
    # Perform matrix multiplication in parallel
    results = pool.starmap(matrix_multiplication, [(chunk_A, B) for chunk_A in chunks_A])
    
    # Close the pool of workers and wait for all processes to complete
    pool.close()
    pool.join()
    
    end_time = time.time()
    time_parallel = end_time - start_time
    
    print(f"Time taken with multiprocessing: {time_parallel:.4f} seconds")
    
    # Combine results from processes (assuming valid matrix split)
    result_parallel = np.concatenate(results)
    
    # Verify correctness by comparing results
    np.testing.assert_array_almost_equal(result_sequential, result_parallel, decimal=5)
    
    print("Matrix multiplication results are consistent.")


from src.helpers.main_process import process_input_directory
import time

start = time.time() 


if __name__ == "__main__":
    INPUT_DIRECTORY = "input"
    OUTPUT_DIRECTORY = "output"
    process_input_directory(INPUT_DIRECTORY, OUTPUT_DIRECTORY)
    print("\nOperation completed successfully. Check the 'output' directory for results.")
end = time.time()
print(f"Execution time: {end - start:.4f} seconds")
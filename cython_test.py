import ctypes
import os

if __name__ == "__main__":
    try:
        example_dll = ctypes.CDLL(os.path.abspath('src/funclustweight.dll'))
    except OSError as e:
        print(f"Error loading DLL: {e}")
        exit()
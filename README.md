# readmetest
This README provides instruction on how to run FunWeightClust in Visual Studio Code.

### Instructions
1. Install the [Conda Package Manager](https://www.anaconda.com/download). The Anaconda distribution will come with every needed dependency. Ensure you also have [Visual Studio Code](https://code.visualstudio.com).

2. To run FunWeightClust in a Jupyter Notebook and Conda, you must change the Kernel to the BASE version of Conda.
![Step 1](images/Jupyter%20Notebook%201.png)
![Step 2](images/Jupyter%20Notebook%202.png)
![Step 3](images/Jupyter%20Notebook%203.png)
![Step 4](images/Jupyter%20Notebook%204.png)

3. To run FunWeightClust in a Python Script, you must change the interpreter to the version of Python installed with the Conda intepreter.
![Step 1](images/Script%201.png)
![Step 2](images/Script%202.png)
![Step 3](images/Script%203.png)
![Step 4](images/Script%204.png)
![Step 5](images/Script%205.png)

4. Install and run the FunWeightClust package in R. (See https://github.com/popescuc71/funclustweight.git) After installing, drag and drop the funclustweight.so (Apple/Linux) or funclustweight.dll (Windows) file from the directory your package is installed in, into the root directory of the Python package

5. Install the scikit-fda package with the following command:
    conda install conda-forge::scikit-fda

### Conclusion
You can now run FunWeightClust in Visual Studio Code
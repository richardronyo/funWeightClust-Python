# TFunHDDC
tfunHDDC is an adapatation of funHDDC (Scmutz et al., 2018) that uses t-distributions for robust clustering in the presence of outliers. In addition, funHDDC is also available for use.

This version of tfunHDDC works with skewed distributions.

# Usage
The main function is `tfunHDDC`. It is designed to take a functional data object along with some parameters for the clustering process.
```
def tfunHDDC(datax, datay, K=np.arange(1,11), model='AKJBKQKDK', 
            modely = "VVV", known=None, threshold=0.1, itermax=200, dfstart=50., eps=1.e-6,init='random',
            criterion='bic', d_select='cattell', init_vector=None, 
            show=True, mini_nb=[5,10], min_individuals=4, mc_cores=1, nb_rep=2,
            keepAllRes=False,kmeans_control={'n_init':1, 'max_iter':10, 'algorithm':'lloyd'},  d_max=100, d_range=2, cmtol=1.e-10, cmmax=10, verbose=True,
            dfupdate='approx', dfconstr='no',   
            Numba=True):
```

Once clustering is done, a `TFunHDDC` object is returned, containing the parameters used for the clustering and the assigned clusters to each curve.

Prediction can also be done using the `TFunHDDC.predict()` function. This requires another `FDataBasis` object with the same number of basis functions as the `FDataBasis` originally clustered on.

The above also holds for funHDDC as well.
```
def funHDDC(data, K=np.arange(1,11), model='AKJBKQKDK', known=None, threshold=0.1, itermax=200, 
            eps=1.e-6, init='random', criterion='bic', d_select='cattell', 
            init_vector=None, show=True, mini_nb=[5,10], min_individuals=4,
            mc_cores=1, nb_rep=2, keepAllRes=True,
            kmeans_control={'n_init':1, 'max_iter':10, 'algorithm':'lloyd'}, d_max=100,
            d_range=2, verbose=True):
```
Instead of a `TFunHDDC` object, `funHDDC` returns a `FunHDDC` object. Note that prediction cannot be done with FunHDDC.

# Examples
There are various examples provided to illustrate how to use TFunHDDC and funHDDC. Examples  are taken from the paper from which TFunHDDC is based on (Anton, C., Smith, I. Model-based clustering of functional data via mixtures of t distributions. Adv Data Anal Classif (2023). https://doi.org/10.1007/s11634-023-00542-w)


fitAdelaideFD.py:
1016 curves are simulated with 6 splines for the basis functions. Curves are classified into 2 clusters.

fitFlourFD.py:
115 curves are simulated with 5 splines for the basis functions. Curves are classified into 3 clusters.

# How to use the test cases
1. Set up and run virtual environment using the following command
```
python -m venv venv
cd venv\Scripts
./activate
```

2. Navigate to root directory and download all the dependencies needed to run the model.
```
pip install -r requirements.txt
```

4. You can run the example files:
```
python fitAdelaideFD.py
```
OR
```
python fitFlourFD.py
```



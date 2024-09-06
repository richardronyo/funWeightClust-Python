import funweightclust as funweight
from fitAlzheimer import fitAlzheimerFD

if __name__ == "__main__":

    data = fitAlzheimerFD()
    fdobj = data['fdx']
    fdobjy = data['fdy']
    clm = data['groupd']

    model = "AkjBkQkDk"
    modely = "EII"
    model=model.upper()

    new_res = funweight.funweightclust(fdobj, fdobjy, K=2, model=model, modely=modely, init="kmeans", nb_rep=1, threshold=0.001)
    
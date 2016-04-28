import multiprocessing as mp
import numpy as np
import pandas as pd
import functools
import types
import time
import re


class Compteur(object):
    """ decorateur pour compter le nombre d'appels et le temps d'exécution.
        => valable pour décorer des fonctions ou des méthodes de classe
     """

    def __init__(self, fonc):
        """initialisation du décorateur"""
        self.fonc = fonc
        self.comptappel = 0  # compteur d'appels
        self.tempsexec = 0  # pour stocker les temps d'exécution
        # pour que les fonctions décorées gardent nom et docstring:
        functools.wraps(fonc)(self)

    def __get__(self, inst):
        """méthode nécessaire pour décorer les méthodes: avoir le bon 'self' """
        return types.MethodType(self, inst)

    def __call__(self, *args, **kwargs):
        """ méthode appelée à chaque appel de la fonction décorée """

        # instructions avant l'appel de la fonction décorée
        self.comptappel += 1  # incrémentation du compteur d'appels
        temps = time.clock()  # initialisation du compteur de temps

        # appel de la fonction décorée
        result = self.fonc(*args, **kwargs)

        # instructions après le retour d'appel de la fonction décorée
        self.tempsexec += time.clock()-temps  # ajout du temps d'exécution

        # fin d'appel
        return result

    def resultat(self):
        """retourne le résultat: compteur d'appel et temps moyen d'exécution"""
        c = self.comptappel
        t = self.tempsexec
        if c == 0:
            tm = 0
        else:
            tm = t/c
        return c, round(tm, 2), round(t, 2)


def try_divide(x, y):
    try:
        return float(x) / float(y)
    except ZeroDivisionError:
        return 0.0


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


# ========================== parallelisation =========================
def parallelize_func(df, funct, nbjob=8):
    # Parallelisation
    p = mp.Pool(processes=nbjob)
    split_df = np.array_split(df, nbjob)
    pool_results = p.map(funct, split_df)
    p.close()
    p.join()

    # merging parts processed by different processes
    df_concat = pd.concat(pool_results, axis=0)

    return df_concat

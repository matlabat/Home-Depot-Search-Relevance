def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import models_param as modp
import param as p
from hyperopt import fmin, tpe, STATUS_OK, Trials
import numpy as np
import pickle
import time


def load_X(ftpath):
    with open(ftpath, 'rb') as file:
        X_all_ft = pickle.load(file)
    X_train_ft = X_all_ft[:p.nbtrain]
    X_test_ft = X_all_ft[p.nbtrain:]
    del X_all_ft

    return X_train_ft, X_test_ft


def load_Y_id():
    with open(p.featpth + "\\df_all_base.pkl", "rb") as f:
        df_all = pickle.load(f)

    id_test = df_all.iloc[p.nbtrain:]['id'].values
    y_train_class = df_all.iloc[:p.nbtrain]['relevance'].map(p.dreltoclass).values
    y_train_val = df_all.iloc[:p.nbtrain]['relevance'].values
    del df_all

    return y_train_class.astype('int32'), y_train_val, id_test


def log(ftmodnm):
    log_handler = open("%s\\%s_%s_hpoptlog.csv" % (p.modellogpth,  time.strftime('%Y%m%d%H%M'), ftmodnm), 'w')
    headers = ['trial_counter', 'time', 'RMSE_mean', 'RMSE_std'] + [k for k, v in sorted(param.items())]
    log_handler.write(";".join(h for h in headers)+"\n")
    log_handler.flush()
    return log_handler


def hyperopt_wrapper(parameters, featmodinfo):
    global trial_counter
    trial_counter += 1

    # convert integer feat
    for f in modp.int_feat():
        if f in parameters:
            parameters[f] = int(parameters[f])

    ## evaluate performance
    print("Trial %d" % trial_counter)
    start = time.clock()
    RMSE_cv_mean, RMSE_cv_std = hyperopt_obj(parameters, featmodinfo)
    end = time.clock()
    elapsed = round((end - start) / 60.0, 2)
    print("in %s" % elapsed, " min")
    print("        Result")
    print("              RMSE_mean: %s" % RMSE_cv_mean)
    print("              RMSE_std: %s" % RMSE_cv_std)

    var_to_log = ["%d" % trial_counter, str(elapsed).replace(".", ","),
                  str(RMSE_cv_mean).replace(".", ","), str(RMSE_cv_std).replace(".", ",")]
    for k,v in sorted(parameters.items()):
        var_to_log.append(str(v).replace(".", ","))
    log_handler.write(";".join(v for v in var_to_log)+"\n")
    log_handler.flush()

    return {'loss': RMSE_cv_mean, 'attachments': {'std': RMSE_cv_std}, 'status': STATUS_OK}


def hyperopt_obj(parameters, feattmodinfo):
    [model, data] = feattmodinfo

    resultlist = []
    for skf in lstskf:
        for tr_id, cv_id in skf:
            dataskf = []
            for dataset in data:
                dataskf.append(dataset[tr_id])
                dataskf.append(dataset[cv_id])

            RMSE, pred = dmod[model](parameters, dataskf)
            resultlist.append(RMSE)

    RMSE_cv_mean = np.mean(np.asarray(resultlist))
    RMSE_cv_std = np.std(np.asarray(resultlist))

    return RMSE_cv_mean, RMSE_cv_std



if __name__ == "__main__":

    nbrun, nbfold = 2, 2
    y_class_tr_all, y_reg_tr_all, id_test = load_Y_id()
    del id_test
    lstskf = modp.create_lstskf(y_class_tr_all, nbrun, nbfold)

    dftmods = {}
    dft = modp.dfeats()
    dmodparam = modp.dmodelsparam()
    dmod = modp.dmodels()

    for feat in dft:
        ftpth = dft[feat][0]
        X_all, X_tst_all = load_X(ftpth)
        del X_tst_all

        for model in dmod:
            param = dmodparam[model]
            ftmodnm = feat + "_" + model
            dftmods[ftmodnm] = [feat, ftpth, model, param]
            data = [X_all, y_class_tr_all, y_reg_tr_all]
            # =========================== Search the best params ===========================
            print("------------------------------------------------------------------------")
            print("-------- Search the best params for %s --------" % ftmodnm)
            starttime = time.clock()
            log_handler = log(ftmodnm)
            trial_counter = 0
            ftmodinfo = [model, data]
            trials = Trials()
            objective = lambda p: hyperopt_wrapper(p, ftmodinfo)
            best_params = fmin(objective, param, algo=tpe.suggest, trials=trials, max_evals=param["max_evals"])

            for f in modp.int_feat():
                if f in best_params:
                    best_params[f] = int(best_params[f])
            elapsed = round((time.clock() - starttime) / 60.0, 2)
            print("************************************************************")
            print("Best params for %s in %.2f min" %(ftmodnm, elapsed))
            for k, v in best_params.items():
                print("        %s: %s" % (k, v))
            trial_RMSEs = np.asarray(trials.losses(), dtype=float)
            best_RMSE_mean = min(trial_RMSEs)
            ind = np.where(trial_RMSEs == best_RMSE_mean)[0][0]
            best_RMSE_std = trials.trial_attachments(trials.trials[ind])['std']
            print("RMSE stats")
            print("        Mean: %.6f\n        Std: %.6f" % (best_RMSE_mean, best_RMSE_std))
            print("        Trial: %s" % str(ind + 1))
            print("************************************************************")
            print()


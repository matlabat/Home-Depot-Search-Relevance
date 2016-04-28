import param as p
import models_param as mp
import train_model as tm
import xgboost as xgb
import pandas as pd
import time
import numpy as np

data_to_load = "cnt_cos_tf"
isbaggingpred = True
nbrun = 2
nbfold = 2


def param_cntcostf():
    return {
        'task': 'regression',
        'booster': 'gbtree',
        'objective': 'reg:linear',
        'eta': 0.024,
        'gamma': 2.2,
        'min_child_weight': 5,
        'max_depth': 12,
        'subsample': 0.6,
        'colsample_bytree': 0.69,
        'num_round': 350,
        'silent': 1,
        'seed': 44,
        }


def make_pred(data_to_load, param, baggingpred, nrun, nfold):

    y_cls_tr, y_reg_tr, id_test = tm.load_Y_id()
    X_tr, X_tst = tm.load_X(mp.dfeats()[data_to_load][0])

    if baggingpred:
        lstskf = mp.create_lstskf(y_cls_tr, nrun, nfold)
        sumpred = np.zeros(X_tst.shape[0])
        listRMSE = []
        for skf in lstskf:
            for tr_id, cv_id in skf:
                X_tr_tr = X_tr[tr_id]
                y_reg_tr_tr = y_reg_tr[tr_id]
                X_tr_cv = X_tr[cv_id]
                y_reg_tr_cv = y_reg_tr[cv_id]
                dtrain = xgb.DMatrix(X_tr_tr, label=y_reg_tr_tr)
                dval = xgb.DMatrix(X_tr_cv, label=y_reg_tr_cv)
                dtest = xgb.DMatrix(X_tst, label=np.ones(X_tst.shape[0]))
                bst = xgb.train(param, dtrain, param['num_round'])
                predtest = bst.predict(dtest)
                predval = bst.predict(dval)
                sumpred += predtest
                listRMSE.append(mp.getscoreRMSE(y_reg_tr_cv, predval))
        print("RMSE Score on CV: ", np.mean(np.asarray(listRMSE)))
        pred = sumpred / float(nrun * nfold)

    else:
        dtrain = xgb.DMatrix(X_tr, label=y_reg_tr)
        dtest = xgb.DMatrix(X_tst, label=np.ones(X_tst.shape[0]))
        bst = xgb.train(param, dtrain, param['num_round'])
        pred = bst.predict(dtest)

    pred[pred > 3] = 3
    pred[pred < 1] = 1

    return pred, id_test

if __name__ == "__main__":

    pred, id_test = make_pred(data_to_load, param_cntcostf(), isbaggingpred, nbrun, nbfold)
    file = "%s\\%s_%s_sub.csv" % (p.outpth,  time.strftime('%Y%m%d%H%M'), data_to_load)
    pd.DataFrame({"id": id_test, "relevance": pred}).to_csv(file, index=False)

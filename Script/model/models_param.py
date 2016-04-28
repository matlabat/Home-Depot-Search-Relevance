import numpy as np
from hyperopt import hp
import xgboost as xgb
from sklearn.decomposition import RandomizedPCA, TruncatedSVD
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale
from sklearn.cross_validation import StratifiedKFold
# import matplotlib.pyplot as plt
import param as p


# =========================== Parameter Space & Models for XGBoost ==============================
# ************************ Gradient Boosting Linear & Tree Regression ***************************
# regression with linear booster
def param_space_reg_xgb_linear():
    return {
        'task': 'regression',
        'booster': 'gblinear',
        'objective': 'reg:linear',
        'eta': hp.quniform('eta', 0.01, 1, 0.01),
        'lambda': hp.quniform('lambda', 0, 5, 0.05),
        'alpha': hp.quniform('alpha', 0, 0.5, 0.005),
        'lambda_bias': hp.quniform('lambda_bias', 0, 3, 0.1),
        'num_round': hp.quniform('num_round', min_num_round, max_num_round, num_round_step),
        'nthread': nb_job,
        'silent': 1,
        'seed': rdmseed,
        "max_evals": max_evals,
    }

# regression with tree booster
def param_space_reg_xgb_tree():
    return {
        'task': 'regression',
        'booster': 'gbtree',
        'objective': 'reg:linear',
        'eta': hp.quniform('eta', 0.02, 0.10, 0.008),
        'gamma': hp.quniform('gamma', 2, 4, 0.2),
        'min_child_weight': hp.quniform('min_child_weight', 5, 7, 1),
        'max_depth': hp.quniform('max_depth', 8, 12, 1),
        'subsample': hp.quniform('subsample', 0.5, 0.85, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.7, 1, 0.03),
        'num_round': hp.quniform('num_round', min_num_round, max_num_round, num_round_step),
        'silent': 1,
        'seed': rdmseed,
        "max_evals": max_evals,
    }

def reg__xgb_linear_tree(param, data):
    [X_tr, X_cv, y_class_tr, y_class_cv, y_reg_tr, y_reg_cv] = data
    dtrain = xgb.DMatrix(X_tr, label=y_reg_tr)
    dvalid = xgb.DMatrix(X_cv, label=y_reg_cv)
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    bst = xgb.train(param, dtrain, param['num_round'])  # , watchlist)
    pred = bst.predict(dvalid)
    RMSEScore = getscoreRMSE(y_reg_cv, pred)

    return RMSEScore, pred

# ************************** Pairwise Ranking with Linear BoosterRegression *********************
# pairwise ranking with linear booster
def param_space_rank_xgb_linear():
    return {
        'task': 'ranking',
        'booster': 'gblinear',
        'objective': 'rank:pairwise',
        'eta': hp.quniform('eta', 0.01, 1, 0.01),
        'lambda': hp.quniform('lambda', 0, 5, 0.05),
        'alpha': hp.quniform('alpha', 0, 0.5, 0.005),
        'lambda_bias': hp.quniform('lambda_bias', 0, 3, 0.1),
        'num_round': hp.quniform('num_round', min_num_round, max_num_round, num_round_step),
        'nthread': nb_job,
        'silent': 1,
        'seed': rdmseed,
        "max_evals": max_evals,
    }

def rank_xgb_linear(param, data):
    [X_tr, X_cv, y_class_tr, y_class_cv, y_reg_tr, y_reg_cv] = data
    hist = np.bincount(y_class_tr)
    cdf = np.cumsum(hist) / float(sum(hist))
    dtrain = xgb.DMatrix(X_tr, label=y_class_tr)
    dvalid = xgb.DMatrix(X_cv, label=y_class_cv)
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    bst = xgb.train(param, dtrain, param['num_round'])  # , watchlist)
    pred = bst.predict(dvalid)
    rank = pred.argsort()

    num = pred.shape[0]
    listbornerel = [[int(cdf[i]*num), int(cdf[i+1]*num), p.dclasstorel[i+1]] for i in range(7)]

    def assign_rel(x):
        for bornerel in reversed(listbornerel):
            if (x >= bornerel[0]) & (x < bornerel[1]):
                return float(bornerel[2])

    vect_assign_rel = np.vectorize(assign_rel)
    newpred = vect_assign_rel(rank)

    RMSEScore = getscoreRMSE(y_reg_cv, newpred)

    return RMSEScore, pred

# *************************** Gradient Boosting Linear Classifier *******************************
# softmax with linear booster
def param_space_clf_xgb_linear():
    return {
        'task': 'softmax',
        'booster': 'gblinear',
        'objective': 'multi:softprob',
        'eta': hp.quniform('eta', 0.01, 1, 0.01),
        'lambda': hp.quniform('lambda', 0, 5, 0.05),
        'alpha': hp.quniform('alpha', 0, 0.5, 0.005),
        'lambda_bias': hp.quniform('lambda_bias', 0, 3, 0.1),
        'num_round': hp.quniform('num_round', min_num_round, max_num_round, num_round_step),
        'num_class': 7,
        'nthread': nb_job,
        'silent': 1,
        'seed': rdmseed,
        "max_evals": max_evals,
    }

def clf_xgb_linear(param, data):
    [X_tr, X_cv, y_class_tr, y_class_cv, y_reg_tr, y_reg_cv] = data
    dtrain = xgb.DMatrix(X_tr, label=y_class_tr-1)
    dvalid = xgb.DMatrix(X_cv, label=y_class_cv-1)
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    bst = xgb.train(param, dtrain, param['num_round'])  # , watchlist)
    pred = bst.predict(dvalid)
    w = np.asarray(range(1, 8))
    pred = pred * w[np.newaxis, :]
    pred = np.sum(pred, axis=1)
    predreg = vectclassval_to_regval(pred)
    RMSEScore = getscoreRMSE(y_reg_cv, predreg)
    return RMSEScore, pred


# ========================== Parameter Space & Models for SKlearn ===============================
# ******************************** Random Forest Regressor **************************************
def param_space_reg_skl_rf():
    return {
        'task': 'reg_skl_rf',
        'n_estimators': hp.quniform("n_estimators", min_num_round, max_num_round, num_round_step),
        'max_features': hp.quniform("max_features", 0.05, 1.0, 0.05),
        'n_jobs': nb_job,
        'random_state': rdmseed,
        "max_evals": max_evals,
    }

def reg_skl_rf(param, data):
    [X_tr, X_cv, y_class_tr, y_class_cv, y_reg_tr, y_reg_cv] = data
    rf = RandomForestRegressor(n_estimators=param['n_estimators'],
                               max_features=param['max_features'],
                               n_jobs=param['n_jobs'],
                               random_state=param['random_state'])
    rf.fit(X_tr, y_reg_tr)
    pred = rf.predict(X_cv)
    RMSEScore = getscoreRMSE(y_reg_cv, pred)
    return RMSEScore, pred


# ********************************** Extra Tree Regressor ***************************************
def param_space_reg_skl_etr():
    return {
        'task': 'reg_skl_etr',
        'n_estimators': hp.quniform("n_estimators", min_num_round, max_num_round, num_round_step),
        'max_features': hp.quniform("max_features", 0.05, 1.0, 0.05),
        'n_jobs': nb_job,
        'random_state': rdmseed,
        "max_evals": max_evals,
    }

def reg_skl_etr(param, data):
    [X_tr, X_cv, y_class_tr, y_class_cv, y_reg_tr, y_reg_cv] = data
    etr = ExtraTreesRegressor(n_estimators=param['n_estimators'],
                              max_features=param['max_features'],
                              n_jobs=param['n_jobs'],
                              random_state=param['random_state'])
    etr.fit(X_tr, y_reg_tr)
    pred = etr.predict(X_cv)
    RMSEScore = getscoreRMSE(y_reg_cv, pred)
    return RMSEScore, pred


# ****************************** Gradient Boosting Regressor ************************************
def param_space_reg_skl_gbm():
    return {
        'task': 'reg_skl_gbm',
        'n_estimators': hp.quniform("n_estimators", min_num_round, max_num_round, num_round_step),
        'learning_rate': hp.quniform("learning_rate", 0.01, 0.5, 0.01),
        'max_features': hp.quniform("max_features", 0.05, 1.0, 0.05),
        'max_depth': hp.quniform('max_depth', 1, 15, 1),
        'subsample': hp.quniform('subsample', 0.5, 1, 0.1),
        'random_state': rdmseed,
        "max_evals": max_evals,
    }

def reg_skl_gbm(param, data):
    [X_tr, X_cv, y_class_tr, y_class_cv, y_reg_tr, y_reg_cv] = data
    gbm = GradientBoostingRegressor(n_estimators=param['n_estimators'],
                                    max_features=param['max_features'],
                                    learning_rate=param['learning_rate'],
                                    max_depth=param['max_depth'],
                                    subsample=param['subsample'],
                                    random_state=param['random_state'])
    gbm.fit(X_tr, y_reg_tr)
    pred = gbm.predict(X_cv)
    RMSEScore = getscoreRMSE(y_reg_cv, pred)
    return RMSEScore, pred


# ******************************* Support Vector Regression *************************************
def param_space_reg_skl_svr():
    return {
        'task': 'reg_skl_svr',
        'C': hp.loguniform("C", np.log(0.00000001), np.log(10)),
        'gamma': hp.loguniform("gamma", np.log(0.001), np.log(0.1)),
        'degree': hp.quniform('degree', 1, 5, 1),
        'epsilon': hp.loguniform("epsilon", np.log(0.001), np.log(0.1)),
        'kernel': hp.choice('kernel', ['rbf', 'poly', 'sigmoid']),
        "max_evals": max_evals,
    }

def reg_skl_svr(param, data):
    [X_tr, X_cv, y_class_tr, y_class_cv, y_reg_tr, y_reg_cv] = data
    svr = SVR(C=param['C'],
              gamma=param['gamma'],
              epsilon=param['epsilon'],
              degree=param['degree'],
              kernel=param['kernel'])
    svr.fit(X_tr, y_reg_tr)
    pred = svr.predict(X_cv)
    RMSEScore = getscoreRMSE(y_reg_cv, pred)
    return RMSEScore, pred


# ************************************ Ridge Regression *****************************************
def param_space_reg_skl_ridge():
    return {
        'task': 'reg_skl_ridge',
        'alpha': hp.loguniform("alpha", np.log(0.0000000001), np.log(10)),
        'random_state': rdmseed,
        "max_evals": max_evals,
    }

def reg_skl_ridge(param, data):
    [X_tr, X_cv, y_class_tr, y_class_cv, y_reg_tr, y_reg_cv] = data
    ridge = Ridge(alpha=param["alpha"], normalize=True)
    ridge.fit(X_tr, y_reg_tr)
    pred = ridge.predict(X_cv)
    RMSEScore = getscoreRMSE(y_reg_cv, pred)
    return RMSEScore, pred


# ************************************ Lasso Regression *****************************************
def param_space_reg_skl_lasso():
    return {
        'task': 'reg_skl_lasso',
        'alpha': hp.loguniform("alpha", np.log(0.00000001), np.log(10)),
        'random_state': rdmseed,
        "max_evals": max_evals,
    }

def reg_skl_lasso(param, data):
    [X_tr, X_cv, y_class_tr, y_class_cv, y_reg_tr, y_reg_cv] = data
    lasso = Lasso(alpha=param["alpha"], normalize=True)
    lasso.fit(X_tr, y_reg_tr)
    pred = lasso.predict(X_cv)
    RMSEScore = getscoreRMSE(y_reg_cv, pred)
    return RMSEScore, pred


# *********************************** Logistic Regression ***************************************
def param_space_clf_skl_lr():
    return {
        'task': 'clf_skl_lr',
        'C': hp.loguniform("C", np.log(0.00000001), np.log(10)),
        'random_state': rdmseed,
        "max_evals": max_evals,
    }

def clf_skl_lr(param, data):
    [X_tr, X_cv, y_class_tr, y_class_cv, y_reg_tr, y_reg_cv] = data
    lr = LogisticRegression(penalty="l2", dual=True, tol=1e-5,
                            C=param['C'], fit_intercept=True, intercept_scaling=1.0,
                            class_weight='auto', random_state=param['random_state'])
    lr.fit(X_tr, y_class_tr)
    pred = lr.predict_proba(X_cv)
    w = np.asarray(range(1, 8))
    pred = pred * w[np.newaxis,:]
    pred = np.sum(pred, axis=1)
    predreg = vectclassval_to_regval(pred)
    RMSEScore = getscoreRMSE(y_reg_cv, predreg)
    return RMSEScore, pred


# ===================================== Scoring & utilities =====================================
def classval_to_regval(yinclass):
    yinreg = 1 + round((yinclass-1)/3, 2)
    return yinreg

vectclassval_to_regval = np.vectorize(classval_to_regval)

def getscoreRMSE(y_pred, y_cv):
    RMSEScore = mean_squared_error(y_cv, y_pred)**0.5
    return RMSEScore


# ==================================== Decomposition Method =====================================
def decomp(wishedmodel, ftnm, X_tr, X_tst=None, verbose=False, nbfact=10):

    if wishedmodel == "PCA":
        model = RandomizedPCA(n_components=nbfact, random_state=2016).fit(X_tr)
    elif wishedmodel == "SVD":
        model = TruncatedSVD(n_components=nbfact, random_state=2016, n_iter=15).fit(X_tr)
    dimini = X_tr.shape[1]
    X_decomp_all = model.transform(X_tr)
    if X_tst is not None:
        X_decomp_all = np.concatenate((X_decomp_all, model.transform(X_tst)), axis=0)
    X_decomp_all = scale(np.matrix(X_decomp_all)).astype("float16")

    if verbose:
        expvar = str(model.explained_variance_ratio_.sum() * 100) + " %"
        print("%s explained variance for features %s with %s %s factors instead of %s intial features" %
              (expvar, ftnm, nbfact, wishedmodel, dimini))
        # plt.plot(model.explained_variance_ratio_)
        # plt.show()
    return X_decomp_all


# ======================================== Validation ===========================================
def create_lstskf(y_class_tr_all, nbrun=1, nbfold=3):
    rdmstate = 8
    listskfrun = []
    for run in range(nbrun):
        skf = StratifiedKFold(y_class_tr_all, n_folds=nbfold, shuffle=True, random_state=rdmstate)
        listskfrun.append(skf)
        rdmstate += 9
    return listskfrun


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ===============================================================================================
# ================================ Combine feats with models ====================================
# ===============================================================================================

min_num_round, max_num_round, num_round_step = 150, 350, 50
rdmseed, nb_job, max_evals = 44, 8, 200

# integer features
def int_feat():
    return ["num_round", "n_estimators", "max_depth", "degree", "hidden_units", "hidden_layers", "batch_size",
            "nb_epoch", "dim", "iter", "max_leaf_forest", "num_iteration_opt", "num_tree_search", "min_pop",
            "opt_interval", 'seed']

def dmodelsparam():
    return {
           "reg_xgb_linear": param_space_reg_xgb_linear(),
           "reg_xgb_tree": param_space_reg_xgb_tree(),
           "clf_xgb_linear": param_space_clf_xgb_linear(),
           "rank_xgb_linear": param_space_rank_xgb_linear(),
           "reg_skl_rf": param_space_reg_skl_rf(),
           "reg_skl_etr": param_space_reg_skl_etr(),
           "reg_skl_gbm": param_space_reg_skl_gbm(),
           "reg_skl_svr": param_space_reg_skl_svr(),
           "reg_skl_ridge": param_space_reg_skl_ridge(),
           "reg_skl_lasso": param_space_reg_skl_lasso(),
           "clf_skl_lr": param_space_clf_skl_lr(),
            }

def dfeats():
    return {
        # "count": [p.featpth + "\\X_all_count_feat_PCA_150.pkl"],
        # "cos_sim": [p.featpth + "\\X_cos_all.pkl"],
        # "tfidf_all": [p.featpth + "\\X_tfidf_all_dim_25.pkl"],
        "cnt_cos_tf": [p.featpth + "\\X_all_cnt_cos_tf.pkl"],
          }

def dmodels():
    return {
           # "reg_xgb_linear": reg__xgb_linear_tree,
           # "reg_xgb_tree": reg__xgb_linear_tree,
           # "clf_xgb_linear": clf_xgb_linear,
           # "reg_skl_rf": reg_skl_rf,
           # "reg_skl_etr": reg_skl_etr,
           "reg_skl_ridge": reg_skl_ridge,
           # "reg_skl_lasso": reg_skl_lasso,

           # Poor results or too long
           # "reg_skl_gbm": reg_skl_gbm,
           # "reg_skl_svr": reg_skl_svr,
           # "rank_xgb_linear": rank_xgb_linear,
           # "clf_skl_lr": clf_skl_lr,
           # "reg_keras_dnn": reg_keras_dnn,
    }





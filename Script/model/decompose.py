import pickle
import param as p
import models_param as mp


if __name__ == "__main__":
    # ======================================== PARAMETERS =========================================
    # *********************************************************************************************
    data = "X_all_count_feat"
    wishmethod = 'PCA'
    lstwishdim = [150]
    getstat = True
    transformtest = True
    write = False
    # *********************************************************************************************

    # ======================================= LOADING DATA ========================================
    print("Load file...")
    filebase = p.featpth + "\\" + data
    with open(filebase + ".pkl", "rb") as f:
        X_all_feat = pickle.load(f)

    X_train_feat = X_all_feat[:p.nbtrain]
    X_test_feat = X_all_feat[p.nbtrain:]
    del X_all_feat

    for dim in lstwishdim:
        print("start decomposition for %s dim %s..." % (wishmethod, dim))
        # with the wished method of decomposition and the wished dimension output
        if transformtest:
            # return a transformed matrix with a shape of ('train + test' obs, 'dim' features)
            X_decomp = mp.decomp(wishmethod, data, X_train_feat, X_tst=X_test_feat, verbose=getstat, nbfact=dim)
        else:
            # return a transformed matrix with a shape of ('train only' obs, 'dim' features)
            X_decomp = mp.decomp(wishmethod, data, X_train_feat, X_tst=None, verbose=getstat, nbfact=dim)
            filebase.replace("_all_", "_train_")

        if write:
            filetowrite = filebase + "_" + wishmethod + "_" + str(dim)
            with open(filetowrite + ".pkl", "wb") as f:
                pickle.dump(X_decomp, f, -1)

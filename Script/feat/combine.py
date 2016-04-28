import pickle
import numpy as np
import param as p


if __name__ == "__main__":
    with open(p.featpth + "\\X_all_count_feat_PCA_150.pkl", "rb") as f:
        X_all_cnt = pickle.load(f)
    with open(p.featpth + "\\X_cos_all.pkl", "rb") as f:
        X_all_cos = pickle.load(f)
    with open(p.featpth + "\\X_tfidf_all_dim_25.pkl", "rb") as f:
        X_all_tf = pickle.load(f)

    X_all_cnt_cos_tf = np.concatenate((X_all_cnt, X_all_cos, X_all_tf), axis=1)

    with open(p.featpth + "\\X_all_cnt_cos_tf.pkl", "wb") as f:
        pickle.dump(X_all_cnt_cos_tf, f, -1)

import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import scale
import param as p
import time



# ========================= Distance Metric ========================
def cosine_sim(x, y):
    try:
        d = cosine_similarity(x, y)
        d = d[0][0]
    except:
        print(x)
        print(y)
        d = 0.
    return d


# ============================= TF-IDF =============================
def get_tf_idf_df(dataframe, dim, rng):
    # construit une tf_idf_matrix a partir dun texte donnee et reduit la dimension avec truncated SVD
    tf = TfidfVectorizer(ngram_range=(1, rng), stop_words='english')
    tfidf_matrix = tf.fit_transform(dataframe)

    svd = TruncatedSVD(n_components=dim, random_state=44, n_iter=30)
    tfidf_reduc = svd.fit_transform(tfidf_matrix)

    print("variance ratio for %s dimension with %s ngram_range: " % (dim, rng), svd.explained_variance_ratio_.sum())
    res = pd.DataFrame(tfidf_reduc)

    return res


def get_total_tf_idf(df, nslice, dim, nrng):
    print("+++++ tf-idf for the entire dataframe")
    df_tf = pd.concat((df['query'], df['title'], df['brand'], df['descr']), axis=0)
    tfidf_total = get_tf_idf_df(df_tf, dim, nrng)

    tfidf_query_n1 = tfidf_total[:nslice].reset_index()
    tfidf_title_n1 = tfidf_total[nslice:nslice * 2].reset_index()
    tfidf_brand_n1 = tfidf_total[nslice * 2:nslice * 3].reset_index()
    tfidf_descr_n1 = tfidf_total[nslice * 3:].reset_index()

    return pd.concat((tfidf_query_n1, tfidf_title_n1, tfidf_brand_n1, tfidf_descr_n1), axis=1)


def get_cos_query_assoc_tf_idf(df, nslice, dim1, dim2, nrng1, nrng2):

    ddimrng = {nrng1: dim1, nrng2: dim2}
    dqft = {"title": ("query_title", "QT"),
            "brand": ("query_brand", "QB"),
            "descr": ("query_descr", "QD")}

    X_cos_sim_all = np.ones([nslice, ]).astype("float16")
    for feat in ["title", "brand", "descr"]:
        print("+++++ cosine sim tf-idf for the query_%s" % feat)
        df_assoc = pd.concat((df["query"], df[feat]), axis=0)
        for nrg in [nrng1, nrng2]:
            if nrg == 2 and feat == "descr":  # too costly for my RAM with this combination...
                continue
            starttime = time.clock()
            tfidfassoc = get_tf_idf_df(df_assoc, ddimrng[nrg], nrg)
            cos_sim = scale(np.asarray([cosine_sim(tfidfassoc[:nslice].iloc[[i]], tfidfassoc[nslice:].iloc[[i]])
                                       for i in range(0, nslice)])).astype("float16")
            X_cos_sim_all = np.vstack((X_cos_sim_all, cos_sim))
            elapsed = round((time.clock() - starttime) / 60.0, 2)

            print("Done for %s for range %s in %.2f min" % (dqft[feat][0], nrg, elapsed))

    X_cos_sim_all = X_cos_sim_all.T
    X_cos_sim_all = X_cos_sim_all[:, 1:]

    return X_cos_sim_all


if __name__ == "__main__":
    sample = None
    with open(p.featpth + "\\df_all_base.pkl", "rb") as f:
        df_all = pickle.load(f)[:sample]

    # ======================================== PARAMETERS =========================================
    numslice = df_all.shape[0]
    token_word = r'(?u)\b\w\w+\b'
    token_word2 = r"[a-z]{2,}|[0-9]+\.?/?,?[0-9]+|[0-9]+"
    token_digit = r"[0-9]+\.?/?,?[0-9]+|[0-9]+"
    token_all = r"[a-z]{2,}|[0-9]+\.?/?,?[0-9]+|[0-9]+"
    dimreduc0 = 25
    dimreduc1 = 150
    dimreduc2 = 200
    nrange1 = 1
    nrange2 = 2
    # ====================================== TFIDF TOTAL ========================================
    print("++++++++++++++++++++++++++++++ Get tfidf total...")
    starttime = time.clock()
    df_tfidf_all = get_total_tf_idf(df_all, numslice, dimreduc0, nrange1)
    X_tfidf_all = df_tfidf_all.as_matrix()
    for col in [i * dimreduc0 for i in range(3, -1, -1)]:
        X_tfidf_all = np.delete(X_tfidf_all, col, 1)
    print(X_tfidf_all.shape)
    X_tfidf_all = scale(X_tfidf_all).astype("float16")
    with open(p.featpth + "\\X_tfidf_all_dim_%s.pkl" % dimreduc0, "wb") as f:
        pickle.dump(X_tfidf_all, f, -1)
    elapsed = round((time.clock() - starttime) / 60.0, 2)
    print("Done in %.2f min" % elapsed)

    # ============= Cosine Similarity for TFIDF Association with 1 & 2 gram range ===============
    print("++++++++++++++++++++++++++++++ Get Cosine Similarity for query association tf-idf 1-gram & 2-gram...")
    starttime = time.clock()
    X_cos_all = get_cos_query_assoc_tf_idf(df_all, numslice, dimreduc1, dimreduc2, nrange1, nrange2)
    with open(p.featpth + "\\X_cos_all.pkl", "wb") as f:
        pickle.dump(X_cos_all, f, -1)

    elapsed = round((time.clock() - starttime) / 60.0, 2)
    print("Done in %.2f min" % elapsed)

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pickle
import time
import param as p
import numpy as np
from sklearn.preprocessing import scale
import re
import utilities as u


# =============================== gen n-gram ============================
def list_word(sentence):
    return [x for x in sentence.split(" ")]


def list_digits(sen):
    try:
        return [w for w in sen.split(" ") if re.match(r"[0-9]+\.?/?[0-9]+|[0-9]+", w)]
    except AttributeError:
        print(sen)
        return []


def getbigram(words, join_string, skip=0):

    assert type(words) == list
    lgth = len(words)
    if lgth > 1:
        lst = []
        for i in range(lgth-1):
            for k in range(1, skip+2):
                if i+k < lgth:
                    lst.append(join_string.join([words[i], words[i+k]]))
    else:
        # set it as unigram
        lst = words
    return lst


def gettrigram(words, join_string, skip=0):

    assert type(words) == list
    lgth = len(words)
    if lgth > 2:
        lst = []
        for i in range(lgth-2):
            for k1 in range(1, skip+2):
                for k2 in range(1, skip+2):
                    if i+k1 < lgth and i+k1+k2 < lgth:
                        lst.append(join_string.join([words[i], words[i+k1], words[i+k1+k2]]))
    else:
        # set it as bigram
        lst = getbigram(words, join_string, skip)
    return lst


def gen_gram(df):

    # ========================= variables over which we loop to create features =============================
    col_names_base = ["query", "title", "brand", "descr"]
    join_str = "_"

    # --------------------------------------- generate 1-gram ---------------------------------------------
    df["query_1gramdigitsk0"] = df["query"].map(list_digits)
    df["title_1gramdigitsk0"] = df["title"].map(list_digits)
    df["brand_1gramdigitsk0"] = df["brand"].map(list_digits)
    df["descr_1gramdigitsk0"] = df["descr"].map(list_digits)
    df["query_1gramsk0"] = df["query"].map(list_word)
    df["title_1gramsk0"] = df["title"].map(list_word)
    df["brand_1gramsk0"] = df["brand"].map(list_word)
    df["descr_1gramsk0"] = df["descr"].map(list_word)

    # --------------------------------------- generate n-gram ----------------------------------------------
    for col in col_names_base:
        df["%s_2gramdigitsk0" % col] = df["%s_1gramdigitsk0" % col].map(lambda x: getbigram(x, join_str))
        df["%s_2gramsk0" % col] = df["%s_1gramsk0" % col].map(lambda x: getbigram(x, join_str))
        df["%s_2gramsk1" % col] = df["%s_1gramsk0" % col].map(lambda x: getbigram(x, join_str, skip=1))
        df["%s_3gramsk0" % col] = df["%s_1gramsk0" % col].map(lambda x: gettrigram(x, join_str))
        df["%s_3gramsk1" % col] = df["%s_1gramsk0" % col].map(lambda x: gettrigram(x, join_str, skip=1))

    return df


# =============================== Counting  =============================
def count_digit(lword):
    return sum([1. for w in lword if re.match(r"[0-9]+\.?/?[0-9]+|[0-9]+", w)])


def count_common(obslst, trgtlst):
    return sum([1. for w in obslst if w in set(trgtlst)])


def str_whole_word(str1, str2, i_):
    cnt = 0
    if len(str2) < len(str1):
        return cnt
    while i_ < len(str2):
        i_ = str2.find(str1, i_)
        if i_ == -1:
            return cnt
        else:
            cnt += 1
            i_ += len(str1)
    return cnt


def str_common_word(str1, str2, pos=None):
    cnt = 0
    words = str1.split()
    if pos is not None and len(words) >= 1:
        words = [words[pos]]
    for word in words:
        if str2.find(word) >= 0:
            cnt += 1
    return cnt


# =============================== Position  =============================
def get_position_list(target, obs):
    pos_of_obs_in_target = [j for j, w in enumerate(obs, start=1) if w in target]
    # print(pos_of_obs_in_target)
    if len(pos_of_obs_in_target) == 0:
        pos_of_obs_in_target = [0]
    return pos_of_obs_in_target


# ============================ Distance Metric ==========================
def JaccardCoef(A, B):
    A, B = set(A), set(B)
    intersect = len(A.intersection(B))
    union = len(A.union(B))
    coef = u.try_divide(intersect, union)
    return coef


def DiceDist(A, B):
    A, B = set(A), set(B)
    intersect = len(A.intersection(B))
    union = len(A) + len(B)
    d = u.try_divide(2*intersect, union)
    return d


def compute_dist(A, B, dist="jaccard_coef"):
    if dist == "jaccard_coef":
        d = JaccardCoef(A, B)
    elif dist == "dice_dist":
        d = DiceDist(A, B)
    return d


# ============================= MAIN FUNCTION ===========================
def gen_feat(df):

    # ========================= variables over which we loop to create features ============================
    col_names_base = ["query", "title", "brand", "descr"]
    grams = ["1gramdigitsk0", "1gramsk0", "2gramdigitsk0", "2gramsk0", "3gramsk0", "2gramsk1", "3gramsk1"]

    # ================================= generate the n-gram dataframe ======================================
    df = gen_gram(df)

    # =============================== generate word counting features ======================================
    for col in col_names_base:
        for gram in grams:
            # word count
            df["count_of_%s_%s" % (col, gram)] = df[col+"_"+gram].map(lambda x: len(x))
            df["count_of_unique_%s_%s" % (col, gram)] = \
                df[col+"_"+gram].map(lambda x: len(set(x)))
            df["ratio_of_uniquew<q_%s_%s" % (col, gram)] = \
                np.vectorize(u.try_divide)(df["count_of_unique_%s_%s" % (col, gram)],
                                           df["count_of_%s_%s" % (col, gram)])

        # digit count
        df["count_of_digit_in_%s" % col] = df[col+"_1gramsk0"].map(count_digit)
        df["ratio_of_digit_in_%s" % col] = \
            np.vectorize(u.try_divide)(df["count_of_digit_in_%s" % col],
                                       df["count_of_%s_1gramsk0" % col])

    # =============================== intersect word counting features ======================================
    for gram in grams:
        for obsnm in col_names_base:
            for trgtnm in col_names_base:
                if trgtnm != obsnm:
                    df["count_of_%s_%s_in_%s" % (obsnm, gram, trgtnm)] = \
                        np.vectorize(count_common)(df[obsnm+"_"+gram], df[trgtnm+"_"+gram])
                    df["ratio_of_%s_%s_in_%s" % (obsnm, gram, trgtnm)] = \
                        np.vectorize(u.try_divide)(df["count_of_%s_%s_in_%s" % (obsnm, gram, trgtnm)],
                                                   df["count_of_%s_%s" % (obsnm, gram)])

            if obsnm != "query":
                df["%s_%s_in_query_div_query_%s" % (obsnm, gram, gram)] = \
                    np.vectorize(u.try_divide)(df["count_of_%s_%s_in_query" % (obsnm, gram)],
                                               df["count_of_query_%s" % gram])
                df["%s_%s_in_query_div_query_%s_in_%s" % (obsnm, gram, gram, obsnm)] = \
                    np.vectorize(u.try_divide)(df["count_of_%s_%s_in_query" % (obsnm, gram)],
                                               df["count_of_query_%s_in_%s" % (gram, obsnm)])

    # =============================== intersect word position features ======================================
    dstat = {"min": np.min, "mean": np.mean, "median": np.median, "max": np.max, "std": np.std}

    for gram in grams:
        for trgtnm in col_names_base:
            for obsnm in col_names_base:
                if trgtnm != obsnm:
                    pos = df.apply(lambda x: get_position_list(x[trgtnm+"_"+gram], x[obsnm+"_"+gram]), axis=1)

                    for stat in dstat:
                        df["pos_of_%s_%s_in_%s_%s" % (obsnm, gram, trgtnm, stat)] = pos.map(dstat[stat])

                    # stats feat on normalized_pos
                    for stat in dstat:
                        df["normalized_pos_of_%s_%s_in_%s_%s" % (obsnm, gram, trgtnm, stat)] = \
                            np.vectorize(u.try_divide)(df["pos_of_%s_%s_in_%s_%s" % (obsnm, gram, trgtnm, stat)],
                                                       df["count_of_%s_%s" % (obsnm, gram)])



    # ========================== generate jaccard coef and dice dist for n-gram ============================
    dists = ["jaccard_coef", "dice_dist"]
    grams = ["1gramdigitsk0", "1gramsk0", "2gramsk0"]
    for dist in dists:
        for gram in grams:
            for i in range(len(col_names_base)-1):
                for j in range(i+1,len(col_names_base)):
                    target_name = col_names_base[i]
                    obs_name = col_names_base[j]
                    df["%s_of_%s_between_%s_%s" % (dist, gram, target_name, obs_name)] = \
                        df.apply(
                            lambda x: compute_dist(x[target_name+"_"+gram], x[obs_name+"_"+gram], dist), axis=1)


    # # ============================ additive features (probable redundancy) ==================================

    for col in col_names_base:
        df['len_of_w_' + col] = df[col + '_1gramsk0'].map(lambda x: len(x))

    df['query_w_in_title'] = np.vectorize(str_common_word)(df["query"], df["title"])
    df['query_w_in_brand'] = np.vectorize(str_common_word)(df["query"], df["brand"])
    df['query_w_in_descr'] = np.vectorize(str_common_word)(df["query"], df["descr"])

    df['query_last_w_in_title'] = np.vectorize(str_common_word)(df["query"], df["title"], -1)
    df['query_last_w_in_brand'] = np.vectorize(str_common_word)(df["query"], df["brand"], -1)
    df['query_last_w_in_descr'] = np.vectorize(str_common_word)(df["query"], df["descr"], -1)
    df['query_first_w_in_title'] = np.vectorize(str_common_word)(df["query"], df["title"], 0)
    df['query_first_w_in_brand'] = np.vectorize(str_common_word)(df["query"], df["brand"], 0)
    df['query_first_w_in_descr'] = np.vectorize(str_common_word)(df["query"], df["descr"], 0)

    df["ratio_title"] = np.vectorize(u.try_divide)(df["query_w_in_title"], df["len_of_w_query"])
    df["ratio_brand"] = np.vectorize(u.try_divide)(df["query_w_in_brand"], df["len_of_w_query"])
    df["ratio_descr"] = np.vectorize(u.try_divide)(df["query_w_in_descr"], df["len_of_w_query"])

    for col in col_names_base:
        df[col + '_lencarac'] = df[col].map(lambda x: len(x))

    df["ratio_carac_title"] = np.vectorize(u.try_divide)(df["query_lencarac"], df["title_lencarac"])
    df["ratio_carac_brand"] = np.vectorize(u.try_divide)(df["query_lencarac"], df["brand_lencarac"])
    df["ratio_carac_descr"] = np.vectorize(u.try_divide)(df["query_lencarac"], df["descr_lencarac"])

    return df


if __name__ == "__main__":
    start = time.clock()
    sample = None
    # ======================================= LOADING DATA ========================================
    print("==================================================")
    print("Load all_file...")
    with open(p.featpth + "\\df_all_base.pkl", "rb") as f:
        df_all = pickle.load(f)[:sample]

    # ===================================== GENERATE FEATURES =====================================
    print("==================================================")
    print("Generate counting features...")

    nb_split = 20
    splits_df_all = np.array_split(df_all, nb_split)
    del df_all
    sumtime = 0
    for i in range(nb_split):
        start = time.clock()
        # MAIN OPERATION: parallelize the gen_feat function
        splits_df_all[i] = u.parallelize_func(splits_df_all[i], gen_feat, nbjob=8)
        feat_names = [name for name in splits_df_all[i].columns
                      if ("count" in name or "ratio" in name or "div" in name or "pos_of" in name or
                          "len_of" in name or "_in_" in name or "carac" in name or "between" in name)]
        splits_df_all[i] = splits_df_all[i][feat_names]
        splits_df_all[i] = splits_df_all[i].as_matrix().astype("float16")

        end = time.clock()
        print("Done: ", i, "in ", round((end - start) / 60.0, 2), " min")
        sumtime += (end - start)

    print("Done all in ", round((sumtime) / 60.0, 2), " min")

    X_all_feat = np.concatenate([splits_df_all[i] for i in range(nb_split)], axis=0)
    time.sleep(10)
    X_all_feat = scale(X_all_feat).astype("float16")
    X_all_feat = np.nan_to_num(X_all_feat)

    with open(p.featpth + "\\X_all_count_feat.pkl", "wb") as f:
        pickle.dump(X_all_feat, f, -1)

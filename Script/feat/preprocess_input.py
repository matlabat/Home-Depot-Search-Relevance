def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import collections
import multiprocessing as mp
import operator
import re
from math import log, pow
import nltk.stem.porter as pstem
import numpy as np
import pandas as pd
from utilities import Compteur
from dictionnary import get_dic
import time
import param as p
import string
import pickle

# ======================================================================
# -------------------------- Global Variable ---------------------------
encod = "ISO-8859-1"

# ======================================================================
# ------------------------- String Processing --------------------------
def strnum():
    return {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}

def listxhandle():
    return [
        ("hxh", "hub x hub"), ("'x", "' x"), ("ftx", "ft x"), (".x", ". x"), ("lx", "l x"),
        ("vx", " x"), ("wx", "w x"), ("x.", "x ."), ("hx", "h x")]

def listnormunite():
    return [
        ("(square|sq|sq\.) ?\.? ?(feet|foot|ft|ft\.)", r"\1 square feet "),
        ("(cubic|cu|cu\.) ?\.? ?(feet|foot|ft|ft\.)", r"\1 cubic feet "),
        ("(foot|feet|ft|ft\.|'')", r"\1 feet "), ("(inches|inch|inchs|in|in\.|')", r"\1 inch "),
        ("(pounds|pound|lbs|lb)", r"\1 pound "), ("(square|sq)", r"\1 square "),
        ("(cubic|cu)", r"\1 cubic "), ("(gallons|gallon|gal|gal\.|gl\.)", r"\1 gallon "),
        ("(ounces|ounce|oz)", r"\1 ounce "), ("(centimeters|cm)", r"\1 centimeter "),
        ("(milimeters|mm|mil)", r"\1 milimeter "), ("(°|degrees|degree|deg|d )", r"\1 degree "),
        ("( v|volts|volt)", r"\1 volt "), ("(watts|watt| w )", r"\1 watt "),
        ("(amperes|ampere|amps|amp|a )", r"\1 ampere ")]

def htmlcodes():
    return [("'", '&#39;'),	('"', '&quot;'), ('>', '&gt;'), ('<', '&lt;'), ('&', '&amp;')]

def normalize_space(s):
    while s.find("  ") > 0:
        s = s.replace("  ", " ")
    return s

def str_process(s):

    if isinstance(s, str):
        # minusculiser tous les mots
        s = s.lower()

        # Normalisation des nombres (écrire "one two three" en "1 2 3")
        s = " ".join([str(strnum()[z]) if z in strnum() else z for z in s.split(" ")])

        # normalisation des codes HTML
        for tupl in htmlcodes():
            s = re.sub(r""+tupl[1]+"", tupl[0], s)

        # normalisation des unités
        for tupl in listnormunite():
            s = re.sub(r"([0-9]+|\()( ?)"+tupl[0]+"\.? ", tupl[1], s)
        s = s.replace("'", " ")
        s = normalize_space(s)

        # suppression de la ponctuation à quelques exceptions près
        # "." caractérise des chiffres à virgules ou des abréviations d'unités
        # "/" caractérise une division, un rapport, une mesure
        # "$" à convertir avec le mot dollars
        # "'" caractérise souvent les "inch" ou les "feet" ('')
        # "*" donne souvent un rapport entre 2 mesures
        punct = "".join([i for i in string.punctuation if i not in [".", "/", "$", "'", "*"]])
        s = re.compile('[%s]' % re.escape(punct)).sub(' ', s)  # remplacement par un espace
        s = normalize_space(s)

        # 4/ séparation des nombres et des chiffres
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)

        # 5/ gestion des "x" collés pour signifier "par"
        for tuplx in listxhandle():
            s = s.replace(tuplx[0], tuplx[1])
        s = normalize_space(s)

        # normalisation des unités
        for tupl in listnormunite():
            s = re.sub(r"([0-9]+|\()( ?)"+tupl[0]+"\.?", tupl[1], s)
        s = s.replace("'", " ")
        s = normalize_space(s)

        # gestion du "."
        s = re.sub(r"( )(\.[0-9]+)", r" 0\2", s)
        s = re.sub(r"([a-z]+)(\.)([a-z]+)", r"\1 \3", s)
        s = re.sub(r"([a-z]+)(\.)", r"\1 ", s)
        s = re.sub(r"(\.)([a-z]+)", r" \2", s)
        s = normalize_space(s)

        # gestion du "/"
        s = re.sub(r"(^|/| )a ?/?c( |/|$)", r" alternating current ", s)
        s = re.sub(r"([a-z]+) ?/ ?([a-z]+)", r"\1 \2", s)
        s = re.sub(r"([a-z]+) ?/ ?", r"\1 ", s)
        s = re.sub(r" ?/ ?([a-z]+)", r" \1", s)
        s = normalize_space(s)

        # gestion du "$", du "*"
        s = re.sub(r"\$", r" dollar ", s)
        s = re.sub(r"\*", r" x ", s)
        s = re.sub(r" x ", r" xbi ", s)
        s = normalize_space(s)

        # gestion des lettres orphelines
        # concaténation des lettres (1 ou 2) devant des chiffres pour identifier des références de produits
        # s = re.sub(r" ((?!in|to|by|of|up|on|tv|no|or|bi|do|pc)\w{2}) ([0-9]+)", r"\1\2", s)
        # s = re.sub(r"([a-z]) ([0-9]+)", r"\1\2", s)
        # s = normalize_space(s)

    return s

def map_str_proc(pdserie):
    return pdserie.map(str_process)

@Compteur
def parallel_serie_str_proc(pdser):
    # Parallelisation
    nb_cpu = mp.cpu_count()
    p = mp.Pool(processes=nb_cpu)
    split_serie = np.array_split(pdser, nb_cpu)
    pool_results = p.map(map_str_proc, split_serie)
    p.close()
    p.join()

    # merging parts processed by different processes
    parts = pd.concat(pool_results, axis=0)

    return parts

# ======================================================================
# ---------------------------- Split words -----------------------------
def infer_spaces(word, dwcost):

    if word in dwcost.keys():
        return word

    suffixeexception = ['ion']
    mxwlen = max(len(x) for x in dwcost.keys())

    # Find the best match for the i first characters, assuming cost has been built for the i-1 first characters.
    # Returns a pair (match_cost, match_length).
    def best_match(i):
        candidates = enumerate(reversed(cost[max(0, i-mxwlen):i]))
        return min((c + dwcost.get(word[i - k - 1:i], 9e999), k + 1) for k, c in candidates)

    # Build the cost array.
    cost = [0]
    for i in range(1, len(word)+1):
        c, k = best_match(i)
        cost.append(c)

    # Backtrack to recover the minimal-cost string.
    out = []
    i = len(word)
    while i > 0:
        c, k = best_match(i)
        assert c == cost[i]
        out.append(word[i - k:i])
        i -= k

    if min([len(x) for x in out]) < 3 or out[0] in suffixeexception:
        return word
    else:
        return " ".join(reversed(out))

def split_w_in_sen(dwcost, sen):
    lsen = []
    for word in str(sen).split(" "):
        if re.match("[a-z]+", word) is not None \
                and len(word) >= 3:
            lsen.append(infer_spaces(word, dwcost))
        else:
            lsen.append(word)
    return " ".join(w for w in lsen)

def process_split_word(listwc, dword, wctodel, dwordcost):
    for wc in listwc:
        (wordini, countini) = wc
        newword = split_w_in_sen(dwordcost, wordini)
        if newword != wordini:
            for split in newword.split(" "):
                dword[split] += countini
            wctodel.append(wc)
            del dword[wordini]

@Compteur
def parallel_split_word(listwordsnum_ini):

    def create_subind(liste):
        wcntini = len(liste)
        start = 0.002
        iteration = 5
        listindref = []
        for i in range(iteration):
            mltpl = pow(3, i)
            endref = int(wcntini * start * mltpl)
            endtrain = endref * 3
            if i == iteration - 1:
                endtrain = None
            if i in [iteration - 2, iteration - 1]:  # on répète l'opération en fin de liste pour éliminer les
                listindref.append([endref, endtrain])  # concaténations à 3 mots

            listindref.append([endref, endtrain])

        return listindref

    listindref = create_subind(listwordsnum_ini)

    listwordsnum_work = listwordsnum_ini
    for [endref, endtrain] in listindref:
        dword = mp.Manager().dict()
        for wc in listwordsnum_work[:endtrain]:
            dword[wc[0]] = wc[1]
        lwordonlyref = [wc[0] for wc in listwordsnum_work[:endref]]
        dwordcost = dict((k, log((i+1)*log(len(lwordonlyref)))) for i, k in enumerate(lwordonlyref))
        wctodel = mp.Manager().list()

        nb_cpu = mp.cpu_count()
        jobs = []
        workser = listwordsnum_work[endref:endtrain]
        for i in range(nb_cpu):
            split = [workser[index] for index in range(i, len(workser), nb_cpu)]
            p = mp.Process(target=process_split_word, args=(split, dword, wctodel, dwordcost))
            jobs.append(p)
            p.start()

        for p in jobs: p.join()

        listwordsnum_work = [wc for wc in listwordsnum_work if wc not in wctodel]

    listwordsnum_work = [(wc[0], dword[wc[0]]) for wc in listwordsnum_work]
    listwordsnum_work.sort(key=operator.itemgetter(1))
    listwordsnum_work.reverse()

    return listwordsnum_work


# -----------------------------------------------------
def map_split(args):
    (pdserie, dwordcost) = args
    return pdserie.map((lambda x: split_w_in_sen(dwordcost, x)))

@Compteur
def para_split(lwordref, pdser):
    dwordcost = dict((str(k), log((i+1)*log(len(lwordref)))) for i, k in enumerate(lwordref))
    # Parallelisation
    nb_cpu = mp.cpu_count()
    p = mp.Pool(processes=nb_cpu)
    split_serie = np.array_split(pdser, nb_cpu)
    splserdico = [(split_serie[i], dwordcost) for i in range(nb_cpu)]
    pool_results = p.map(map_split, splserdico)
    p.close()
    p.join()

    # merging parts processed by different processes
    parts = pd.concat(pool_results, axis=0)
    return parts


# ======================================================================
# ------------------------- Eliminate Stop-Words -----------------------
def eliminate_sw(liststopword, sen):
    return " ".join(w for w in str(sen).split(" ") if w not in liststopword)

def map_elim_sw(args):
    (pdserie, liststopword) = args
    return pdserie.map((lambda x: eliminate_sw(liststopword, x)))

@Compteur
def para_elim_sw(lsw, pdser):
    # Parallelisation
    nb_cpu = mp.cpu_count()
    p = mp.Pool(processes=nb_cpu)
    split_serie = np.array_split(pdser, nb_cpu)
    splserdico = [(split_serie[i], lsw) for i in range(nb_cpu)]
    pool_results = p.map(map_elim_sw, splserdico)
    p.close()
    p.join()

    # merging parts processed by different processes
    parts = pd.concat(pool_results, axis=0)
    return parts


# ======================================================================
# ------------------------------ Stemming ------------------------------
stemmer = pstem.PorterStemmer()
def stem_sen(sen):
    return " ".join(stemmer.stem(w) for w in sen.split(" "))

def map_stem(pdserie):
    return pdserie.map(stem_sen)

@Compteur
def parallel_serie_stem(pdser):
    # Parallelisation
    nb_cpu = mp.cpu_count()
    p = mp.Pool(processes=nb_cpu)
    split_serie = np.array_split(pdser, nb_cpu)
    pool_results = p.map(map_stem, split_serie)
    p.close()
    p.join()

    # merging parts processed by different processes
    parts = pd.concat(pool_results, axis=0)

    return parts


# -----------------------------------------------------
def rem_unknwn(knwnstem, sen):
    newsen = " ".join(w for w in sen.split(" ") if w in knwnstem or re.match("[a-z]+", w) is None)
    if re.match("[a-z]+", newsen) is None:
        return sen
    else:
        return newsen

def map_rem_unknwn(args):
    (pdserie, knwnstem) = args
    return pdserie.map((lambda x: rem_unknwn(knwnstem, x)))

@Compteur
def para_rem_unknwn(knwnstem, pdser):
    # Parallelisation
    nb_cpu = mp.cpu_count()
    p = mp.Pool(processes=nb_cpu)
    split_serie = np.array_split(pdser, nb_cpu)
    splserdico = [(split_serie[i], knwnstem) for i in range(nb_cpu)]
    pool_results = p.map(map_rem_unknwn, splserdico)
    p.close()
    p.join()

    # merging parts processed by different processes
    parts = pd.concat(pool_results, axis=0)
    return parts


# ======================================================================
# ------------------------ Spell Correction ----------------------------
def get_words_serie(listofserie):
    listwordtemp = []
    for ser in listofserie:
        listwordtemp.extend(ser.map(get_words))

    listword = [word for listwordser in listwordtemp for word in listwordser]
    return listword

def get_words(text):
    return re.findall('[a-z]+', str(text).lower())

def get_w_cnt(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1.0
    return model

def edits1(word):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    replaces = [a + c + b[1:] for a, b in splits for c in alphabet if b]
    inserts = [a + c + b for a, b in splits for c in alphabet]
    return set(deletes + transposes + replaces + inserts)

def known_edits2(word, dicoref):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in dicoref)

def known(words, dicoref):
    return set(w for w in words if w in dicoref)

def correct(args):
    (word, dref) = args
    if re.match("[a-z]+", word) is not None:
        candidates = known([word], dref) or known(edits1(word), dref) or known_edits2(word, dref) or [word]
        if len(candidates) >= 1:
            return max(candidates, key=dref.get)
        else:
            return word
    else:
        return word

def correctsentence(dicoref, sen):
    return " ".join(correct((word, dicoref)) for word in str(sen).split(" "))

def map_spell_corr(args):
    (pdserie, dicoword) = args
    return pdserie.map((lambda x: correctsentence(dicoword, x)))

@Compteur
def para_spell_corr(dicowsc, pdser):
    # Parallelisation
    nb_cpu = mp.cpu_count()
    p = mp.Pool(processes=nb_cpu)
    split_serie = np.array_split(pdser, nb_cpu)
    splserdico = [(split_serie[i], dicowsc) for i in range(nb_cpu)]
    pool_results = p.map(map_spell_corr, splserdico)
    p.close()
    p.join()

    # merging parts processed by different processes
    parts = pd.concat(pool_results, axis=0)
    return parts

def spellcheck_query(text):
    if text in get_dic():
        return get_dic()[text]
    else:
        return text


# ==================== Processing =====================
if __name__ == "__main__":
    listdf = ['query', 'title', 'brand', 'descr']
    sample = None
    # ********************************************************************************************************
    # ********************* Step 1/ load the initial data and make the desired dataframes ********************
    print("Step 1/ load the initial data and make the desired dataframes")
    start = time.clock()
    df_train = pd.read_csv(p.inpth + '\\train.csv', encoding=encod)[:sample]
    df_test = pd.read_csv(p.inpth + '\\test.csv', encoding=encod)[:sample]
    df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    pid = df_all[['product_uid']].drop_duplicates(["product_uid"])['product_uid']
    df_descr = pd.read_csv(p.inpth + '\\product_descriptions.csv')
    df_descr = df_descr[df_descr["product_uid"].isin(pid)]
    df_pattr = pd.read_csv(p.inpth + '\\attributes.csv')
    df_pattr = df_pattr[df_pattr["product_uid"].isin(pid)]

    # Correction of the query using the Google correct typo dict kindly shared on
    # https://www.kaggle.com/steubk/home-depot-product-search-relevance/fixing-typos
    df_all['search_term'] = df_all['search_term'].map(spellcheck_query)

    df_query = df_all[['search_term']].drop_duplicates(["search_term"])
    df_query["qid"] = np.arange(len(df_query["search_term"]))
    df_title = df_all[['product_uid', 'product_title']].drop_duplicates(["product_uid"])
    df_brand = df_pattr[df_pattr.name == "MFG Brand Name"][["product_uid", "value"]].drop_duplicates(
        ["product_uid"]).rename(columns={"value": "product_brand"})

    # assign a 'best' approximation 3-voted relevance for the very few 2 or 4-voted relevance
    # cf: https://www.kaggle.com/briantc/home-depot-product-search-relevance/homedepot-first-dataexploreation-k
    df_all['relevance'][df_all['relevance'].isnull()] = 1
    dict_rel = {1: 1, 1.25: 1.33, 1.33: 1.33, 1.5: 1.67, 1.67: 1.67, 1.75: 1.67, 2: 2, 2.25: 2.33, 2.33: 2.33,
                2.5: 2.67, 2.67: 2.67, 2.75: 2.67, 3: 3}
    df_all['relevance'] = df_all['relevance'].map(lambda x: dict_rel[x])
    df_all = df_all.merge(df_brand, on='product_uid', how='left')
    df_all = df_all.merge(df_descr, on='product_uid', how='left')
    df_all = df_all.merge(df_query, on='search_term', how='left')
    df_all = df_all[["id", "product_uid", "qid", "search_term", "product_title",
                     "product_brand", "product_description", "relevance"]]

    df_all.to_csv(p.featpth + '\\df_all_ini.csv', sep=";", index=False)

    del df_train, df_test, df_pattr, df_all

    dicdf = {'query': df_query, 'title': df_title, 'brand': df_brand, 'descr': df_descr}

    end = time.clock()
    print(round((end - start) / 60.0, 2), " min")

    # ********************************************************************************************************
    # **************** Step 2/ string processing for all the text present in the provided data ***************
    print("Step 2/ string processing for all the text present in the provided data")
    start = time.clock()

    df_query['query'] = parallel_serie_str_proc(df_query['search_term'])
    df_title['title'] = parallel_serie_str_proc(df_title['product_title'])
    df_brand['brand'] = parallel_serie_str_proc(df_brand['product_brand'])
    df_descr['descr'] = parallel_serie_str_proc(df_descr['product_description'])

    end = time.clock()
    print(round((end - start) / 60.0, 2), " min")

    # # ********************************************************************************************************
    # # ******************************** Step 3/ get a reference list of words *********************************
    # # get a clean list of words used in all the Home Depot Dataframe
    # print("Step 3/ get a reference list of words")
    # start = time.clock()
    # dicowordsHD = get_w_cnt(get_words_serie([df_query['query'], df_title['title'],
    #                                             df_brand['brand'], df_descr['descr']]))
    # listwordsHD = list(dicowordsHD.items())
    # listwordsHD.sort(key=operator.itemgetter(1))
    # listwordsHD.reverse()
    # # costly operation to "deconcatenate" recursively words from top (occurences) to down in the dictionnary
    # # ex: {garden: 7000, flower: 5000, gardenflower: 23} -> {garden: 7023, flower: 5023}
    # # (gardenflower become a word to delete regarding to its initial occcurence vs its potential splits occurence)
    # listwordsHDclean = parallel_split_word(listwordsHD)
    # df_wordsHD = pd.DataFrame(listwordsHDclean, columns=['word', 'count_HD'])
    # df_wordsHD.sort_values(by='count_HD', ascending=False, inplace=True)
    #
    # # get a list of words used in the query dataframe
    # dicowordsquery = get_w_cnt(get_words_serie([df_query['query']]))
    # df_wordsquery = pd.DataFrame(list(dicowordsquery.items()), columns=['word', 'count_query'])
    # df_wordsquery.sort_values(by='count_query', ascending=False, inplace=True)
    #
    # # get a list of words used in the title dataframe
    # dicowordstitle = get_w_cnt(get_words_serie([df_title['title']]))
    # df_wordstitle = pd.DataFrame(list(dicowordstitle.items()), columns=['word', 'count_title'])
    # df_wordstitle.sort_values(by='count_title', ascending=False, inplace=True)
    #
    # # get a list of words used in the brand dataframe
    # dicowordsbrand = get_w_cnt(get_words_serie([df_brand['brand']]))
    # df_wordsbrand = pd.DataFrame(list(dicowordsbrand.items()), columns=['word', 'count_brand'])
    # df_wordsbrand.sort_values(by='count_brand', ascending=False, inplace=True)
    #
    # # get a list of words used in a big corpus of english text
    # # that you can find at http://norvig.com/big.txt
    # dicowordseng = get_w_cnt(get_words(open(p.featpth + "\\big.txt", 'r').read()))
    # df_wordseng = pd.DataFrame(list(dicowordseng.items()), columns=['word', 'count_english'])
    # df_wordseng.sort_values(by='count_english', ascending=False, inplace=True)
    #
    # # make the final dataframe of the reference list of word
    # df_AllW = df_wordsHD.merge(df_wordseng, on='word', how='left')
    # df_AllW = df_AllW.merge(df_wordsquery, on='word', how='left')
    # df_AllW = df_AllW.merge(df_wordstitle, on='word', how='left')
    # df_AllW = df_AllW.merge(df_wordsbrand, on='word', how='left')
    #
    # df_AllW['count_english'][df_AllW['count_english'].isnull()] = 0.0
    # df_AllW['count_query'][df_AllW['count_query'].isnull()] = 0.0
    # df_AllW['count_title'][df_AllW['count_title'].isnull()] = 0.0
    # df_AllW['count_brand'][df_AllW['count_brand'].isnull()] = 0.0
    #
    # df_AllW = df_AllW[(df_AllW['count_HD'] >= 30) |
    #                   (df_AllW['count_english'] >= 30) |
    #                   (df_AllW['count_query'] >= 1) |
    #                   (df_AllW['count_title'] >= 5) |
    #                   (df_AllW['count_brand'] >= 1)]
    # df_sw = pd.read_csv(p.featpth + '\\eng_stopwords.csv', sep=";", encoding="ISO-8859-1")
    # stemmer = pstem.PorterStemmer()
    # df_AllW['stem'] = df_AllW['word'].map(stemmer.stem)
    # df_AllW['issw'] = df_AllW['word'].isin(df_sw['stopword'])
    # df_AllW['weight'] = df_AllW['count_HD']**0.35 + df_AllW['count_english']**0.45 + df_AllW['count_query']**0.7 + \
    #                     df_AllW['count_title']**0.6 + df_AllW['count_brand']**0.8
    #
    # df_AllW.to_csv(p.featpth + '\\Listwordscount.csv', sep=";", index=False)
    #
    # end = time.clock()
    # print(round((end - start) / 60.0, 2), " min")

    # ********************************************************************************************************
    # *********************************** Step 4/ process the dataframes *************************************
    print("Step 4/ process the dataframes")
    start = time.clock()
    df_AllW = pd.read_csv(p.featpth + '\\Listwordscount.csv', sep=";")
    stopwords = list(df_AllW[df_AllW['issw']]['word'])
    knownstem = list(df_AllW['stem'])
    df_wordspellc = df_AllW[['word', 'weight']].dropna()
    dwordspellc = df_wordspellc.set_index('word')['weight'].to_dict()

    for df in listdf:
        startdf = time.clock()
        print("+++++++++++++ %s ++++++++++++ " % df)

        dicdf[df][df] = para_elim_sw(stopwords, dicdf[df][df])
        print("1/ para_elim_sw OK after", round((time.clock() - startdf) / 60.0, 2), " min")

        dicdf[df][df] = para_split(df_AllW['word'], dicdf[df][df])
        print("2/ para_split OK after", round((time.clock() - startdf) / 60.0, 2), " min")

        if df != "descr":
            dicdf[df][df] = para_spell_corr(dwordspellc, dicdf[df][df])
            print("3/ para_spell_corr OK after", round((time.clock() - startdf) / 60.0, 2), " min")

        dicdf[df][df] = parallel_serie_stem(dicdf[df][df])
        print("4/ parallel_serie_stem OK after", round((time.clock() - startdf) / 60.0, 2), " min")

        dicdf[df][df] = para_rem_unknwn(knownstem, dicdf[df][df])
        print("5/ para_rem_unknwn OK after", round((time.clock() - startdf) / 60.0, 2), " min")

        with open(p.featpth + "\\df_%s_processed_ok" %df, 'wb') as f:
            pickle.dump(dicdf[df], f, -1)

    # print("para_elim_sw:", para_elim_sw.resultat())
    # print("para_split:", para_split.resultat())
    # print("para_spell_corr:", para_spell_corr.resultat())
    # print("parallel_serie_stem:", parallel_serie_stem.resultat())
    # print("para_rem_unknwn:", para_rem_unknwn.resultat())

    end = time.clock()
    print(round((end - start) / 60.0, 2), " min")

    # ********************************************************************************************************
    # ******************************** Step 5/ write the new base dataframe **********************************
    print("Step 5/ write the new base dataframe")
    start = time.clock()
    # dicdf = {}
    # listdf = ['query', 'title', 'brand', 'descr']
    # for df in listdf:
    #     with open(p.featpth + "\\df_%s_processed_ok" %df, 'rb') as f:
    #         dicdf[df] = pickle.load(f)

    df_all = pd.read_csv(p.featpth + '\\df_all_ini.csv', sep=";", encoding=encod)
    df_all = df_all[["id", "product_uid", "search_term", "relevance"]]
    df_all = df_all.merge(dicdf['query'], on='search_term', how='left')
    df_all = df_all.merge(dicdf['title'], on='product_uid', how='left')
    df_all = df_all.merge(dicdf['brand'], on='product_uid', how='left')
    df_all = df_all.merge(dicdf['descr'], on='product_uid', how='left')

    df_all = df_all[["id", "product_uid", "qid", "query", "title", "brand", "descr", "relevance"]]
    df_all['brand'][df_all['brand'].isnull()] = ""
    df_all['brand'][df_all['brand'] == "unbrand"] = ""

    print(df_all.info())

    df_all.to_csv(p.featpth + '\\df_all_base.csv', sep=";", index=False)
    with open(p.featpth + "\\df_all_base.pkl", 'wb') as f:
        pickle.dump(df_all, f, -1)

    end = time.clock()
    print(round((end - start) / 60.0, 2), " min")


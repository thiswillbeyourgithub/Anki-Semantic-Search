import re
from pprint import pprint
from tqdm import tqdm
from pathlib import Path
import shutil
import argparse

import pandas as pd
import ankipandas as akp

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
import fasttext
import fasttext.util
from nltk.corpus import stopwords

from util import whi, red, yel, text_formatter
tqdm.pandas()


class SemanticSearcher:
    def __init__(self, profiles, user_input, sw_lang, rebuild):
        # misc settings:
        self.keep_OCR = True
        self.compress_arg = None  # {'method': 'gzip', 'compresslevel': 9}
        # set to None to disable compression, which is faster but bigger
        self.cache_fp = Path("anki_vectors_cache.json.gzip")

        # init values:
        self.stops = set()
        for lang in sw_lang:
            [self.stops.add(w) for w in stopwords.words(lang)]

        # load vectorizer
        self.v = Vectorizer('cc.en.300._ignore_backups_.bin', self.stops)

        # start of the program
        yel("Copying anki profiles")
        cached_db = []
        Path.mkdir(Path("cache"), exist_ok=True)
        for p in profiles:
            original_db = akp.find_db(user=p)
            new_name = f"anki_profile_{p}.cache".replace(" ", "_")
            cached_db.append(shutil.copy(original_db, f"./cache/{new_name}"))

        if len(cached_db) != len(profiles):
            raise Exception

        yel("Loading anki profiles...", end=" ")
        col_notes = []
        for i in range(len(profiles)):
            col_notes.append(akp.Collection(cached_db[i]).notes[["nflds", "nmod"]])
            for j in col_notes[i].index:
                col_notes[i].at[j, "pid"] = f"{profiles[i]}_{j}"
            col_notes[i] = col_notes[i].set_index("pid")

        self.col = pd.concat(col_notes, axis=0, join="inner")

        if len(self.col.index) != sum([len(c.index) for c in col_notes]):
            n = len(self.col.index)
            m = sum([len(c.index) for c in col_notes])
            red(f"Error during merging: {n} vs {m}")
            raise Exception
        else:
            yel("Success!")

        if rebuild:
            yel("Removing old cache and build a new one from scratch...")
            Path.unlink(self.cache_fp, missing_ok=True)
            self._update_cache(create=True)
        elif not self.cache_fp.exists():
            self._update_cache(create=True)
        else:
            self._update_cache()

        if user_input is not None:
            self._process_input(user_input)
        else:
            self.wait_for_input()

        while True:
            self.wait_for_input()

    def _process_input(self, user_input):
        "search for notes based on user input, then wait for more inputs"
        if isinstance(user_input, list):
            assert len(user_input) == 1
            user_input = user_input[0]
        cache = self.cache
        col = self.col
        cache["dist"] = 0
        vec = self.v.get_vec(user_input)
        vecs = cache.drop(columns=["dist", "nmod"])

        cache["dist"] = pairwise_distances(vecs, vec, metric="cosine", j_jobs=-1)
        index = cache.index
        good_order = sorted(index,
                            key=lambda row: cache.loc[row, "dist"],
                            reverse=False)

        pd.set_option('display.max_colwidth', None)
        for n, i in enumerate(good_order[0:10]):
            print(f"\n{n+1}) {i}: {str(self._call_tp(col.loc[i, 'nflds']))[0:300]}")
        pd.reset_option('display.max_colwidth')


    def wait_for_input(self):
        red("\nSearch query? (q to quit)")
        ans = input(">")
        if ans in ["q", "quit", "exit"]:
            raise SystemExit()
        else:
            self._process_input(ans)


    def _update_cache(self, create=False):
        """checks modtime of stored values vs loaded collection, update the
        vectors if needed"""
        vec_list = [f"VEC_{x}" for x in range(0, self.v.ft.get_dimension())]
        if create:
            yel("Local cache not found, creating...")
            self.cache = pd.DataFrame(index=self.col.index,
                    columns=["nmod"] + vec_list,
                    data=None)
            self.cache["nmod"] = 0
            to_update = list(self.col.index)
        else:
            yel(f"Importing {self.cache_fp}...", end=" ")
            self.cache = pd.read_json(self.cache_fp, compression=self.compress_arg)
            yel("Success!")

            to_update = []
            for i in tqdm(self.col.index,
                          desc="Comparing collections to cached values",
                          unit=" note"):
                if (i not in self.cache.index) or \
                        (self.col.loc[i, "nmod"] != self.cache.loc[i, "nmod"]):
                    to_update.append(i)

        if not to_update:
            yel("Local cache is up to date!")
            return

        red(f"{len(to_update)} notes will be updated.\n")

        for note in tqdm(to_update, desc="Processing text", unit=" note"):
            self.col.at[note, "nflds"] = self._call_tp(self.col.loc[note, "nflds"])

        self.cache.loc[to_update, vec_list] = np.array([self.v.get_vec(x)
                for x in tqdm(
                    self.col.loc[to_update, "nflds"],
                    desc="Vectorizing text",
                    unit=" note")
                ]).reshape(len(to_update), self.v.ft.get_dimension())
        self.cache.loc[to_update, "nmod"] = self.col.loc[to_update, "nmod"]

        yel(f"Storing vectors to {self.cache_fp}...", end=" ")
        self.cache.to_json(self.cache_fp, compression=self.compress_arg)
        yel("Done!")
        return

    def _call_tp(self, text):
        if isinstance(text, list):
            text = " ".join(text)
        return text_formatter(text, self.keep_OCR)



class Vectorizer:
    def __init__(self, ft_name, stops):
        self.alphanum = re.compile(r"[^ _\w]|\d|_|\b\w\b")
        self.stops = stops

        if not Path(ft_name).exists():
            yel("Downloading fasttext model")
            fasttext.util.download_model("en", if_exists="ignore")
        yel("Importing fasttext model")
        self.ft = fasttext.load_model(ft_name)

        self.memoized_vec = self.memoize(self.ft.get_word_vector)

    def memoize(self, f):
        """
        store previous value to speed up vector retrieval
        (40x speed up)
        """
        memo = {}

        def helper(x):
            if x not in memo:
                memo[x] = f(x)
            return memo[x]
        return helper

    def preprocessor(self, string):
        """
        prepare string of text to be vectorized by fastText
        * makes lowercase
        * replaces all non letters by a space
        * outputs preprocessed comb_text as a list of words
        """
        return re.sub(self.alphanum, " ", string.lower()).split()


    def get_vec(self, string):
        """
        compute vector representation of each word of the note, then pool
        them to represent the complete note text
        """
        return normalize(
                np.sum(
                    [
                        self.memoized_vec(x)
                        for x in self.preprocessor(string)
                        if x not in self.stops
                        ],
                    axis=0).reshape(1, -1),
                norm='l1')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles",
                        nargs="+",
                        metavar="PROFILES",
                        dest='profiles',
                        type=str,
                        required=True,
                        help="Anki profiles to search in")
    parser.add_argument("--query",
                        nargs=1,
                        metavar="query",
                        dest='user_input',
                        type=str,
                        required=False,
                        help="Search query")
    parser.add_argument("--rebuild",
                        dest='rebuild',
                        action="store_true",
                        required=False,
                        default=False,
                        help="deletes old cache and start from scratch")
    parser.add_argument("--sw_lang",
                        nargs="+",
                        metavar="STOPWORDS_LANG",
                        dest='sw_lang',
                        type=str,
                        required=False,
                        default=["french", "english", "spanish"],
                        help="Language of reference, used to ignore stop words")
    args = parser.parse_args().__dict__
    whi(f"Launched Anki SemSearch with arguments :")
    pprint(args)

    SemanticSearcher(**args)

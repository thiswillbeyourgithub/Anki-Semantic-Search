import re
from pprint import pprint
from tqdm import tqdm
from pathlib import Path
import shutil
import argparse
import threading

import pandas as pd
import ankipandas as akp

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
import fasttext
import fasttext.util
from nltk.corpus import stopwords

from util import whi, red, yel, text_formatter
import prompt_toolkit
tqdm.pandas()


class SemanticSearcher:
    def __init__(self, profiles, user_input, sw_lang, rebuild):
        # misc settings:
        self.keep_OCR = True
        self.compress_arg = None  # {'method': 'gzip', 'compresslevel': 9}
        # set to None to disable compression, which is faster but bigger
        self.cache_fp = Path("anki_vectors_cache.json.gzip")
        self.last_input = user_input

        # init values:
        self.stops = set()
        for lang in sw_lang:
            [self.stops.add(w) for w in stopwords.words(lang)]

        # load vectorizer
        self.async_loader = threading.Thread(target=self._model_loader)
        self.async_loader.start()

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
        self.col_notes = []
        self.threads = []
        for i in range(len(profiles)):
            t = threading.Thread(target = self._profile_loader,
                                 args=(cached_db, profiles, i))
            t.start()
            self.threads.append(t)

        [t.join() for t in self.threads]
        self.col = pd.concat(self.col_notes, axis=0, join="inner")

        if len(self.col.index) != sum([len(c.index) for c in self.col_notes]):
            n = len(self.col.index)
            m = sum([len(c.index) for c in self.col_notes])
            red(f"Error during merging: {n} vs {m}")
            raise Exception
        else:
            yel("Success!")

        self.async_loader.join()
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

    def _profile_loader(self, cached_db, profiles, i):
        "loading a profile is slow so use a specific thread"
        col_notes = akp.Collection(cached_db[i]).notes[["nflds", "nmod"]]
        for j in col_notes.index:
            col_notes.at[j, "pid"] = f"{profiles[i]}_{j}"
        col_notes = col_notes.set_index("pid")
        self.col_notes.append(col_notes)
        yel(f"Loaded profile {profiles[i]}")

    def _model_loader(self):
        "loading model is slow so use a specific thread"
        self.v = Vectorizer('cc.en.300._ignore_backups_.bin', self.stops)

    def _process_input(self, user_input):
        "search for notes based on user input, then wait for more inputs"
        if isinstance(user_input, list):
            assert len(user_input) == 1
            user_input = user_input[0]
        self.last_input = user_input
        cache = self.cache
        col = self.col
        cache["dist"] = 0
        vecs = cache.drop(columns=["dist", "nmod"])
        vec = self.v.get_vec(user_input)

        cache["dist"] = pairwise_distances(vecs,
                                           vec,
                                           metric="cosine",
                                           n_jobs=-1)

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
        ans = prompt_toolkit.prompt(">", default=self.last_input)
        if ans in ["q", "quit", "exit"]:
            raise SystemExit()
        else:
            self._process_input(ans)


    def _update_cache(self, create=False):
        """checks modtime of stored values vs loaded collection, update the
        vectors if needed"""
        vec_list = [f"VEC_{x}" for x in range(1, self.v.ft.get_dimension() + 1)]
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

        to_drop = []
        for note in tqdm(to_update, desc="Processing text", unit=" note"):
            self.col.at[note, "nflds"] = self._call_tp(self.col.loc[note, "nflds"]).strip()
            if self.col.loc[note, "nflds"] == "":
                to_drop.append(note)

        if to_drop:
            for drop in to_drop:
                red(f"Removed empty note #{drop}")
                self.col.drop(drop, inplace=True)
                to_update.remove(drop)
            red(f"{len(to_drop)} notes were removed.")

        vecs = [self.v.get_vec(self.col.loc[ind, "nflds"]) for ind in tqdm(to_update, desc="Vectorizing") ]
        for i, v in enumerate(vecs):
            if v.shape != (1, self.v.ft.get_dimension()):
                red(f"Invalid vectors : note {to_update[i]}")
                self.col.at[to_update[i], "nflds"] = "Invalid vector :     " + self.col.loc[to_update[i], "nflds"]
                vecs[i] = self.v.get_vec(self.col.loc[to_update[i], "nflds"])

        ar = np.array(vecs).reshape( (len(to_update), self.v.ft.get_dimension()) )

        already_present = [x for x in to_update if x in self.cache.index]
        to_add = [x for x in to_update if x not in self.cache.index]
        for x in to_add:
            self.cache.loc[x, :] = 0
        for vec_n in range(self.v.ft.get_dimension()):
            self.cache.loc[to_update, vec_list[vec_n]] = ar[:, vec_n]

        yel(f"Updated {len(already_present)} notes, added {len(to_add)} new notes.")

        old_shape = self.cache.values.shape[0]
        self.cache = self.cache.dropna()
        if self.cache.shape[0] != old_shape:
            red(f"Had to drop {old_shape-self.cache.shape[0]} rows that were NA.")

        self.cache.loc[to_update, "nmod"] = self.col.loc[to_update, "nmod"]

        yel(f"Storing vectors to {self.cache_fp}...", end=" ")
        self.cache.to_json(self.cache_fp, compression=self.compress_arg)
        yel("Done!")
        return

    def _call_tp(self, text):
        "call to text processor"
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
        out = normalize(
              np.sum(
                  [
                      self.memoized_vec(x)
                      for x in self.preprocessor(string)
                      if x not in self.stops
                      ],
                  axis=0).reshape(1, -1),
              norm='l2',
              copy=False)
        return out


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

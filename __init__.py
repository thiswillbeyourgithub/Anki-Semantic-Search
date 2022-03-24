import re
import json
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

from util import whi, red, yel
tqdm.pandas()

# todo :
# remove akp warnings
# acronym replacement?
# create repo
# mention in AnnA
# add requirements
# ignore suspended cards


class SemanticSearcher:
    def __init__(self, profiles, user_input, sw_lang):
        self.profiles = profiles
        self.cache_fp = Path("anki_vectors_cache.json.gzip")
        self.keep_OCR = True
        self.stops = set()
        for lang in sw_lang:
            [self.stops.add(w) for w in stopwords.words(lang)]
        self.v = Vectorizer('cc.en.300._ignore_backups_.bin', self.stops)

        yel("Copying anki profiles")
        cached_db = []
        Path.mkdir(Path("cache"), exist_ok=True)
        for p in profiles:
            print(p)
            original_db = akp.find_db(user=p)
            new_name = f"anki_profile_{p}.cache".replace(" ", "_")
            cached_db.append(shutil.copy(original_db, f"./cache/{new_name}"))

        if len(cached_db) != len(profiles):
            raise Exception

        yel("Loading anki profiles")
        col_notes = []
        for i in range(len(profiles)):
            col_notes.append(akp.Collection(cached_db[i]).notes[["nflds", "ntags", "nmod"]])
            for j in col_notes[i].index:
                col_notes[i].at[j, "pid"] = f"{profiles[i]}_{j}"
            col_notes[i] = col_notes[i].set_index("pid")

        yel("Combining profiles...", end=" ")
        self.col = pd.concat(col_notes, axis=0, join="inner")

        if len(self.col.index) != sum([len(c.index) for c in col_notes]):
            n = len(self.col.index)
            m = sum([len(c.index) for c in col_notes])
            red(f"Error during merging: {n} vs {m}")
            raise Exception
        else:
            yel("Success!")

        if not self.cache_fp.exists():
            self._update_cache(create=True)
        else:
            self._update_cache()

        if user_input is None:
            self.wait_for_input()
        else:
            self._process_input(user_input)

    def _process_input(self, user_input):
        "search for notes based on user input, then wait for more inputs"
        vec = self.v.get_vec(user_input)
        self.cache["dist"] = self.cache["vec"].progress_apply(
            lambda x: pairwise_distances(vec, x), metric="cosine")

        self.wait_for_input()


    def wait_for_input(self):
        red("\nInput? (q to quit)\r>")
        ans = input()
        if ans in ["q", "quit", "exit"]:
            raise SystemExit()
        else:
            self._process_input(ans)


    def _update_cache(self, create=False):
        """checks modtime of stored values vs loaded collection, update the
        vectors if needed"""
        if create:
            yel("Local cache not found, creating...")
            self.cache = pd.DataFrame()
        else:
            yel(f"Importing {self.cache_fp}")
            self.cache = pd.read_json(self.cache_fp)

        to_update = []
        for i in tqdm(self.col.index, desc="Comparing collections to cached values", unit=" note"):
            if i not in self.cache.index:
                to_update.append(i)
                red(f"Note {i} is missing from cache")
            elif self.col.loc[i, "nmod"] != self.cache.loc[i, "nmod"]:
                to_update.append(i)
                red(f"Note {i} is outdated")
        
        if not to_update:
            yel("Local cache is up to date!")
            return

        to_update = to_update[0:50]  # testing

        def process(fields):
            return self._text_formatter(" ".join(fields))

        for note in tqdm(to_update, desc="Processing text", unit=" note"):
            self.col.at[note, "nflds"] = process(self.col.loc[note, "nflds"])

        for note in tqdm(to_update, desc="Vectorizing text", unit=" note"):
            self.col.at[note, "vec"] = self.v.get_vec(self.col.loc[note, "nflds"])

        yel(f"Storing vectors to {self.cached_db}...", end=" ")
        self.col[["vec", "nmod"]].to_json(
            self.cache_fp,
            compression={'method': 'gzip', 'compresslevel': 1}
            )
        yel("Done!")

        yel("Reloading cache...")
        self.cache = pd.read_json(self.cache_fp)

    def _text_formatter(self, text):
        """
        copied from my other repository : AnnA
        """
        text = text.replace("&amp;", "&"
                    ).replace("+++", " important "
                    ).replace("&nbsp", " "
                    ).replace("\u001F", " ")

        # remove weird clozes
        text = re.sub(r"}}{{c\d+::", "", text)

        # remove sound recordings
        text = re.sub(r"\[sound:.*?\..*?\]", " ", text)

        # duplicate bold and underlined content, as well as clozes
        text = re.sub(r"\b<u>(.*?)</u>\b", r" \1 \1 ", text,
                      flags=re.M | re.DOTALL)
        text = re.sub(r"\b<b>(.*?)</b>\b", r" \1 \1 ", text,
                      flags=re.M | re.DOTALL)
        text = re.sub(r"{{c\d+::.*?}}", lambda x: f"{x.group(0)} {x.group(0)}",
                      text, flags=re.M | re.DOTALL)

        # if blockquote or li or ul, mention that it's a list item
        # usually indicating a harder card
        if re.match("</?blockquote/?>|</?li/?>|</?ul/?>|", text, flags=re.M):
            text += " list list list list list"

        # remove html spaces
        text = re.sub('\\n|</?div/?>|</?br/?>|</?span/?>|</?li/?>|</?ul/?>', " ", text)
        text = re.sub('</?blockquote(.*?)>', " ", text)

        # OCR
        if self.keep_OCR:
            # keep image title (usually OCR)
            text = re.sub("<img src=.*? title=\"(.*?)\".*?>",
                          lambda string: self._store_OCR(string),
                          text,
                          flags=re.M | re.DOTALL)

        # cloze
        text = re.sub(r"{{c\d+?::|}}", "", text)  # remove cloze brackets
        text = re.sub("::", " ", text)  # cloze hints
        text = re.sub("{{c", "", text)  # missed cloze?

        # misc
        text = re.sub(r'[a-zA-Z0-9-]+\....', " ", text)  # media file name
        text = re.sub("<a href.*?</a>", " ", text)  # html links
        text = re.sub(r'https?://\S*?', " ", text)  # plaintext links
        text = re.sub("</?su[bp]>", "", text) # exponant or indices
        text = re.sub(r"\[\d*\]", "", text)  # wiki style citation

        text = re.sub("<.*?>", "", text)  # remaining html tags
        text = text.replace("&gt", "").replace("&lt", "").replace("<", "").replace(">", "").replace("'", " ")  # misc + french apostrophe


        # misc
        text = " ".join(text.split())  # multiple spaces
        text = re.sub(r"\b[a-zA-Z]'(\w{2,})", r"\1", text)  # misc etc

        return text


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
                norm='l2')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles",
                        nargs=1,
                        metavar="PROFILES",
                        dest='profiles',
                        type=str,
                        required=True,
                        help="Anki profiles to search in")
    parser.add_argument("--query",
                        nargs=1,
                        metavar="query",
                        dest='query',
                        type=str,
                        required=False,
                        help="Search query")
    parser.add_argument("--sw_lang",
                        nargs=1,
                        metavar="STOPWORDS_LANG",
                        dest='sw_lang',
                        type=str,
                        required=False,
                        default='["french", "english", "spanish"]',
                        help="Language of reference, used to ignore stop words")
    args = parser.parse_args().__dict__
    whi(f"Launched ASS with args :\r{args}")
    args["profiles"] = json.loads(args["profiles"][0])
    args["sw_lang"] = json.loads(args["sw_lang"])
    SemanticSearcher(profiles=args["profiles"],
                     user_input=args["query"],
                     sw_lang=args["sw_lang"])

# This project has been abandonned for a while. but I made a way better tool that allows directly asking questions to your anki database: [DocToolsLLM](https://github.com/thiswillbeyourgithub/DocToolsLLM)


# Anki SemSearch: do AI powered semantic search on your anki collection(s)
Use AI (fastText's multilingual word vectors) to search through your anki notes.

# This is still a work in progress!

## FAQ
* What is it and how does it work?
* Why should I use this?
* **This looks awesome! Are you working on other things like that?** Yes and thank you! Another project of mine uses AI to reduce your daily workload on anki : [I call it AnnA (Anki neuronal Appendix)](https://github.com/thiswillbeyourgithub/AnnA_Anki_neuronal_Appendix)

## Usage

## How to
*tested on `python3.9`*
* `pip install -r requirements.txt`
* `python ./__init__.py --profiles ["Main", "Old"] --query="path taken by Nile river"`


## TODO
* investigate using [Faiss](https://github.com/facebookresearch/faiss/wiki/Getting-started) to speed up the search but orders of magnitude
* use joblib caching instead of diy solutions
* prettier results
* shortcut to open in anki browser
* remove warnings from ankipandas and pandas
* replace acronyms
* add requirements file
* add argument to load other fastText models
* add argument to ignore suspended cards or not
* only vectorize words deemed important according to keyword extraction techniques from TFIDF


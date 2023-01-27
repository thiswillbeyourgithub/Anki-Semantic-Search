import re
from tqdm import tqdm


def coloured_log(color_asked):
    """used to print color coded logs"""
    col_red = "\033[91m"
    col_yel = "\033[93m"
    col_rst = "\033[0m"

    if color_asked == "white":
        def printer(string, **args):
            if isinstance(string, list):
                string = ",".join(string)
            tqdm.write(col_rst + string + col_rst, **args)
    elif color_asked == "yellow":
        def printer(string, **args):
            if isinstance(string, list):
                string = ",".join(string)
            tqdm.write(col_yel + string + col_rst, **args)
    elif color_asked == "red":
        def printer(string, **args):
            if isinstance(string, list):
                string = ",".join(string)
            tqdm.write(col_red + string + col_rst, **args)
    return printer


whi = coloured_log("white")
yel = coloured_log("yellow")
red = coloured_log("red")


def text_formatter(text, keep_OCR):
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

    # remove html spaces
    text = re.sub('\\n|</?div/?>|</?br/?>|</?span/?>|</?li/?>|</?ul/?>', " ", text)
    text = re.sub('</?blockquote(.*?)>', " ", text)

    # OCR
    if keep_OCR:
        # keep image title (usually OCR)
        text = re.sub("<img .*?title=\"(.*?)\".*?>",
                      lambda x: f" {x.group(1)} ",
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

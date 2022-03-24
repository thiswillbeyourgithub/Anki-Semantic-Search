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

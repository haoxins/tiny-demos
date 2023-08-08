import re
import os.path as path

# http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt


def read_time_machine():
    fp = path.join(path.dirname(__file__), "../../data/time_machine.txt")
    with open(fp, "r") as f:
        lines = f.readlines()
    return [re.sub("[^A-Za-z]+", " ", line).strip().lower() for line in lines]


lines = read_time_machine()
print("# text lines:", len(lines))
print(lines[0])
print(lines[10])


def tokenize(lines, token="word"):
    if token == "word":
        return [line.split() for line in lines]
    elif token == "char":
        return [list(line) for line in lines]
    else:
        print("ERROR: unknown token type:", token)
        return None


tokens = tokenize(lines)
print("# tokens:", len(tokens))
for i in range(11):
    print(tokens[i])


class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []

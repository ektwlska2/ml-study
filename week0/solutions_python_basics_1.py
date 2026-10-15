"""Reference solutions for python_basics_1.py. Look only after your own version passes (or after a real attempt)."""


def fizzbuzz(n):
    out = []
    for i in range(1, n + 1):
        if i % 15 == 0:          # check "both" first, otherwise "Fizz" would win
            out.append("FizzBuzz")
        elif i % 3 == 0:
            out.append("Fizz")
        elif i % 5 == 0:
            out.append("Buzz")
        else:
            out.append(str(i))
    return out


def reverse_words(s):
    return " ".join(s.split()[::-1])   # split() with no argument drops extra spaces


def is_palindrome(s):
    cleaned = [c.lower() for c in s if c.isalnum()]
    return cleaned == cleaned[::-1]


def count_chars(s):
    counts = {}
    for c in s:
        counts[c] = counts.get(c, 0) + 1
    return counts


def second_largest(nums):
    distinct = sorted(set(nums))       # set removes duplicates
    return distinct[-2] if len(distinct) >= 2 else None


def flatten(nested):
    return [x for sub in nested for x in sub]   # read left to right like nested for-loops


def merge_sum(d1, d2):
    out = dict(d1)                      # copy, so d1 is not modified
    for k, v in d2.items():
        out[k] = out.get(k, 0) + v
    return out


def chunk(lst, k):
    return [lst[i:i + k] for i in range(0, len(lst), k)]


def transpose(m):
    # loop version:
    # rows, cols = len(m), len(m[0])
    # return [[m[r][c] for r in range(rows)] for c in range(cols)]
    return [list(col) for col in zip(*m)]


def moving_average(xs, w):
    if w > len(xs):
        return []
    s = sum(xs[:w])
    out = [s / w]
    for i in range(w, len(xs)):
        s += xs[i] - xs[i - w]          # add the new element, drop the oldest
        out.append(s / w)
    return out

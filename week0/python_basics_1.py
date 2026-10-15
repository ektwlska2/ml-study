"""
Week 0 · Python Basics I — 10 exercises
=======================================

How to use
1. Write each solution on PAPER first (exam practice), then type it in below.
2. Replace each `raise NotImplementedError` with your code.
3. Run:   python3 python_basics_1.py
   It tells you which exercises pass. Fix until all 10 pass.
4. Only then compare with solutions_python_basics_1.py.

Use only the basics from the lesson: no imports.
"""


# 1. FizzBuzz
# Return a list of strings for 1..n: "Fizz" if divisible by 3, "Buzz" if by 5,
# "FizzBuzz" if by both, otherwise the number as a string.
# fizzbuzz(5) -> ["1", "2", "Fizz", "4", "Buzz"]
def fizzbuzz(n):
    raise NotImplementedError


# 2. Reverse the order of words (single spaces between words in the output).
# reverse_words("  hello   big world ") -> "world big hello"
def reverse_words(s):
    raise NotImplementedError


# 3. Palindrome check that ignores case and anything that is not a letter or digit.
# is_palindrome("A man, a plan, a canal: Panama") -> True
# Hint: str.isalnum(), str.lower(), slicing [::-1]
def is_palindrome(s):
    raise NotImplementedError


# 4. Count each character (including spaces) and return a dict.
# count_chars("aab ") -> {"a": 2, "b": 1, " ": 1}
# Hint: dict.get(key, 0)
def count_chars(s):
    raise NotImplementedError


# 5. Second largest DISTINCT value. Return None if it doesn't exist.
# second_largest([3, 1, 3, 2]) -> 2 ;  second_largest([5, 5]) -> None
def second_largest(nums):
    raise NotImplementedError


# 6. Flatten one level of nesting.
# flatten([[1, 2], [3], [], [4, 5]]) -> [1, 2, 3, 4, 5]
# Try to write it as ONE list comprehension.
def flatten(nested):
    raise NotImplementedError


# 7. Merge two dicts; where a key is in both, add the values.
# Do NOT modify d1 or d2.
# merge_sum({"a": 1, "b": 2}, {"b": 10, "c": 5}) -> {"a": 1, "b": 12, "c": 5}
def merge_sum(d1, d2):
    raise NotImplementedError


# 8. Split a list into chunks of size k (the last chunk may be shorter).
# chunk([1, 2, 3, 4, 5], 2) -> [[1, 2], [3, 4], [5]]
# Hint: range(start, stop, step) and slicing
def chunk(lst, k):
    raise NotImplementedError


# 9. Transpose a matrix stored as a list of rows.
# transpose([[1, 2, 3], [4, 5, 6]]) -> [[1, 4], [2, 5], [3, 6]]
# Write it twice in your head: with nested loops, then with zip(*m).
def transpose(m):
    raise NotImplementedError


# 10. Moving average with window w (the ML one).
# Return a list of len(xs) - w + 1 floats: the mean of each window.
# moving_average([1, 2, 3, 4, 5], 3) -> [2.0, 3.0, 4.0]
# Bonus: make it O(n) by keeping a running sum instead of re-summing each window.
def moving_average(xs, w):
    raise NotImplementedError


# ---------------------------------------------------------------- tests
def _tests():
    return {
        1: lambda: fizzbuzz(15)[-1] == "FizzBuzz" and fizzbuzz(5) == ["1", "2", "Fizz", "4", "Buzz"] and fizzbuzz(0) == [],
        2: lambda: reverse_words("  hello   big world ") == "world big hello" and reverse_words("one") == "one",
        3: lambda: is_palindrome("A man, a plan, a canal: Panama") and not is_palindrome("race a car") and is_palindrome(""),
        4: lambda: count_chars("aab ") == {"a": 2, "b": 1, " ": 1} and count_chars("") == {},
        5: lambda: second_largest([3, 1, 3, 2]) == 2 and second_largest([5, 5]) is None and second_largest([]) is None and second_largest([-1, -7]) == -7,
        6: lambda: flatten([[1, 2], [3], [], [4, 5]]) == [1, 2, 3, 4, 5] and flatten([]) == [],
        7: lambda: _t7(),
        8: lambda: chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]] and chunk([], 3) == [] and chunk([1, 2], 5) == [[1, 2]],
        9: lambda: transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]] and transpose([[7]]) == [[7]],
        10: lambda: moving_average([1, 2, 3, 4, 5], 3) == [2.0, 3.0, 4.0] and moving_average([4, 6], 2) == [5.0] and moving_average([1, 2], 3) == [],
    }


def _t7():
    d1, d2 = {"a": 1, "b": 2}, {"b": 10, "c": 5}
    out = merge_sum(d1, d2)
    return out == {"a": 1, "b": 12, "c": 5} and d1 == {"a": 1, "b": 2} and d2 == {"b": 10, "c": 5}


if __name__ == "__main__":
    passed = 0
    for n, test in _tests().items():
        try:
            ok = bool(test())
            msg = "pass" if ok else "WRONG ANSWER"
        except NotImplementedError:
            ok, msg = False, "not done yet"
        except Exception as e:
            ok, msg = False, f"ERROR: {type(e).__name__}: {e}"
        passed += ok
        print(f"Exercise {n:2d}: {msg}")
    print(f"\n{passed}/10 passed")

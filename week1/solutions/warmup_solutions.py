"""Warm-up solutions. Look only after a real attempt."""


def invert(d):
    return {v: k for k, v in d.items()}


def group_by_length(words):
    groups = {}
    for w in words:
        n = len(w)
        if n not in groups:
            groups[n] = []
        groups[n].append(w)
    return groups


def common_keys(d1, d2):
    return sorted(k for k in d1 if k in d2)


def count_pairs(nums, target):
    count = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                count += 1
    return count


def matrix_add(a, b):
    result = []
    for i in range(len(a)):
        row = []
        for j in range(len(a[0])):
            row.append(a[i][j] + b[i][j])
        result.append(row)
    return result


def averages(scores):
    out = {}
    for name, xs in scores.items():
        out[name] = sum(xs) / len(xs) if xs else None
    return out

import random

def recommend_numbers(count=1, exclude_numbers=None, include_numbers=None):
    exclude_numbers = set(exclude_numbers or [])
    include_numbers = set(include_numbers or [])
    results = []
    for _ in range(count):
        pool = [n for n in range(1, 46) if n not in exclude_numbers and n not in include_numbers]
        if len(pool) + len(include_numbers) < 6:
            continue
        result = random.sample(pool, 6 - len(include_numbers)) + list(include_numbers)
        results.append(sorted(result))
    return results

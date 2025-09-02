import collections
import random

import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras import Input
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

from app.db import SessionLocal
from app.models.lotto import LottoDraw

def is_similar(combo, seen, threshold=5):
    combo_set = set(combo)
    for prev in seen:
        if len(combo_set & set(prev)) >= threshold:
            return True
    return False

def prepare_lotto_sequences(win_numbers, seq_len=10):
    encoded = []
    for draw in win_numbers:
        vec = np.zeros(45)
        for n in draw:
            vec[n - 1] = 1
        encoded.append(vec)
    X, y = [], []
    for i in range(len(encoded) - seq_len):
        X.append(encoded[i:i + seq_len])
        y.append(encoded[i + seq_len])
    return np.array(X), np.array(y)


def train_lotto_lstm(win_numbers, epochs=5):
    X, y = prepare_lotto_sequences(win_numbers)
    model = Sequential([
        Input(shape=(X.shape[1], X.shape[2])),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(128, activation='relu'),
        Dense(45, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=epochs, batch_size=16, verbose=0)
    return model


def get_all_win_numbers(start_round=None, end_round=None):
    db = SessionLocal()
    try:
        query = db.query(LottoDraw)
        if start_round and end_round:
            query = query.filter(
                LottoDraw.draw_no >= start_round,
                LottoDraw.draw_no <= end_round
            )
        draws = query.order_by(LottoDraw.draw_no).all()
        if not draws:
            latest = db.query(LottoDraw).order_by(LottoDraw.draw_no.desc()).first()
            if not latest:
                return []
            latest_no = latest.draw_no
            start = max(1, latest_no - 99)
            draws = db.query(LottoDraw).filter(
                LottoDraw.draw_no >= start, LottoDraw.draw_no <= latest_no
            ).order_by(LottoDraw.draw_no).all()
        return [[d.n1, d.n2, d.n3, d.n4, d.n5, d.n6] for d in draws]
    finally:
        db.close()


def get_recent_miss_counts(win_numbers):
    last_seen = {n: -1 for n in range(1, 46)}
    for i, nums in enumerate(reversed(win_numbers)):
        for n in nums:
            if last_seen[n] == -1:
                last_seen[n] = i
    return {n: last_seen[n] if last_seen[n] >= 0 else len(win_numbers) for n in range(1, 46)}


def get_bonus_inclusive_frequency(win_numbers):
    flat_all = [n for row in win_numbers for n in row]
    freq = collections.Counter(flat_all)
    return freq


def get_non_appearance_counts(win_numbers):
    last_seen = {n: -1 for n in range(1, 46)}
    counts = [0] * 46
    for i, draw in enumerate(reversed(win_numbers)):
        for n in range(1, 46):
            if n in draw and last_seen[n] == -1:
                last_seen[n] = i
    for n in range(1, 46):
        counts[n] = last_seen[n] if last_seen[n] != -1 else len(win_numbers)
    return {n: counts[n] for n in range(1, 46)}


def get_stat_weights(win_numbers):
    flat_all = [n for row in win_numbers for n in row]
    freq = collections.Counter(flat_all)
    all_nums = list(range(1, 46))
    counts = [freq.get(n, 0) for n in all_nums]
    mean, std = np.mean(counts), np.std(counts) + 1e-6
    z_scores = [(freq.get(n, 0) - mean) / std for n in all_nums]
    softmax_weights = np.exp(z_scores) / np.sum(np.exp(z_scores))
    non_appear = get_non_appearance_counts(win_numbers)
    max_na = max(non_appear.values())
    norm_non_appear = {n: (non_appear[n] / max_na) for n in all_nums}
    weights = {
        n: round(float(sw) + 0.3 * norm_non_appear[n], 6)
        for n, sw in zip(all_nums, softmax_weights)
    }
    return weights


def get_ml_weights(win_numbers):
    flat_all = [n for row in win_numbers for n in row]
    freq = collections.Counter(flat_all)
    max_f, min_f = max(freq.values()), min(freq.values())
    last_seen = {n: -1 for n in range(1, 46)}
    for i, draw in enumerate(reversed(win_numbers)):
        for n in range(1, 46):
            if n in draw and last_seen[n] == -1:
                last_seen[n] = i
    max_na = len(win_numbers)
    non_appear = {n: (last_seen[n] if last_seen[n] != -1 else max_na) for n in range(1, 46)}
    norm_na = {n: non_appear[n] / max_na for n in range(1, 46)}
    weights = {
        n: round(((max_f - freq.get(n, 0)) / (max_f - min_f + 1e-6)) + 1.2 + 0.6 * norm_na[n], 6)
        for n in range(1, 46)
    }
    return weights


def get_opt_weights(win_numbers):
    flat_all = [n for row in win_numbers for n in row]
    freq = collections.Counter(flat_all)
    last_seen = {n: -1 for n in range(1, 46)}
    for i, draw in enumerate(reversed(win_numbers)):
        for n in range(1, 46):
            if n in draw and last_seen[n] == -1:
                last_seen[n] = i
    max_na = len(win_numbers)
    non_appear = {n: (last_seen[n] if last_seen[n] != -1 else max_na) for n in range(1, 46)}
    norm_na = {n: non_appear[n] / max_na for n in range(1, 46)}
    weights = {
        n: round(1.2 + freq.get(n, 0) / 12.0 + 0.6 * norm_na[n] + (0.6 if freq.get(n, 0) == 0 else 0), 6)
        for n in range(1, 46)
    }
    return weights


def prioritize_by_weights(results, weights):
    def score(nums):
        s = sum(weights.get(n, 1.0) for n in nums)
        if 100 <= sum(nums) <= 180:
            s += 0.5
        return -s, random.random()

    return sorted(results, key=score)


def recommend_by_stat(count=1, exclude=None, include=None, start_round=None, end_round=None):
    exclude, include = set(exclude or []), set(include or [])
    db = SessionLocal()

    try:
        # 회차별 전체 데이터 불러오기
        query = db.query(LottoDraw)
        if start_round and end_round:
            query = query.filter(LottoDraw.draw_no >= start_round, LottoDraw.draw_no <= end_round)
        draws = query.order_by(LottoDraw.draw_no).all()
        if not draws:
            return []

        # 번호 추출
        win_numbers = [[d.n1, d.n2, d.n3, d.n4, d.n5, d.n6] for d in draws]


        # 번호별 등장 횟수 계산 (보너스 포함)
        combined = [n for row in win_numbers for n in row]
        freq = collections.Counter(combined)
        all_nums = list(range(1, 46))
        freq_vector = np.array([freq.get(n, 0) for n in all_nums])

        # softmax + 정규화
        z_scores = (freq_vector - np.mean(freq_vector)) / (np.std(freq_vector) + 1e-6)
        softmax_weights = np.exp(z_scores) / np.sum(np.exp(z_scores))

        # 미출현 횟수 계산
        last_seen = {n: -1 for n in all_nums}
        for i, draw in enumerate(reversed(win_numbers)):
            for n in all_nums:
                if n in draw and last_seen[n] == -1:
                    last_seen[n] = i
        max_na = len(win_numbers)
        non_appear_score = np.array([
            (last_seen[n] if last_seen[n] != -1 else max_na) / max_na
            for n in all_nums
        ])

        # 최종 가중치 (softmax + 미출현 보정)
        weights = {
            n: float(round(softmax_weights[i] + 0.2 * non_appear_score[i], 6))
            for i, n in enumerate(all_nums)
        }

        # 후보군 생성 (상위 빈도 기준)
        top_n = 35
        top_candidates = set([n for n, _ in freq.most_common(top_n)])
        candidates = top_candidates - exclude - include

        # 조합 생성
        results, seen = [], set()
        attempts = 0
        max_attempts = 300

        while len(results) < count * 3 and attempts < max_attempts:
            attempts += 1
            pick = set(include)
            while len(pick) < 6:
                pool = list(candidates - pick)
                if not pool:
                    break
                probs = [weights[n] for n in pool]
                chosen = random.choices(pool, weights=probs, k=1)[0]
                pick.add(chosen)

            combo = tuple(sorted(pick))
            combo_frozen = frozenset(combo)

            if combo_frozen not in seen and not is_similar(combo, seen):
                seen.add(combo_frozen)

                score = sum(weights[n] for n in combo)
                total = sum(combo)
                odds = sum(n % 2 for n in combo)
                score -= abs(odds - 3) * 0.3

                if 100 <= total <= 180 and 2 <= odds <= 4:
                    score += 0.5

                results.append((score, list(combo)))

        results.sort(key=lambda x: (-x[0], random.random()))
        return [combo for _, combo in results[:count]]

    finally:
        db.close()


def recommend_by_ml(count=1, exclude=None, include=None, start_round=None, end_round=None):
    exclude, include = set(exclude or []), set(include or [])
    raw_numbers = get_all_win_numbers(start_round, end_round)
    if not raw_numbers:
        return []

    win_numbers = [row[:6] for row in raw_numbers if
                   isinstance(row, list) and len(row) >= 6 and all(isinstance(n, int) for n in row[:6])]

    # 가중치 계산
    weights = get_ml_weights(win_numbers)
    miss_counts = get_recent_miss_counts(win_numbers)

    # 마르코프 전이행렬 생성
    transitions = {i: collections.Counter() for i in range(1, 46)}
    for prev, next_ in zip(win_numbers, win_numbers[1:]):
        for p in prev:
            transitions[p].update(next_)

    results, seen = [], set()
    while len(results) < count * 3 and len(results) < 100:
        pick = set(include)
        last_draw = random.choice(win_numbers[-1])
        while len(pick) < 6:
            if transitions[last_draw]:
                next_num = random.choices(*zip(*transitions[last_draw].items()))[0]
            else:
                next_num = random.choice([n for n in range(1, 46) if n not in pick and n not in exclude])
            pick.add(next_num)

        combo = tuple(sorted(pick))
        if combo not in seen and not is_similar(combo, seen):
            seen.add(combo)
            total = sum(combo)
            score = sum(weights[n] + miss_counts[n] * 0.01 for n in combo)
            odds = sum(n % 2 for n in combo)
            score -= abs(odds - 3) * 0.3
            if 100 <= total <= 180 and 2 <= odds <= 4:
                score += 0.5
            results.append((score, list(combo)))

    results.sort(key=lambda x: (-x[0], random.random()))
    return [combo for _, combo in results[:count]]


def recommend_by_opt(count=1, exclude=None, include=None, start_round=None, end_round=None):
    exclude, include = set(exclude or []), set(include or [])
    win_numbers_raw = get_all_win_numbers(start_round, end_round)
    win_numbers = [row[:6] for row in win_numbers_raw if
                   isinstance(row, list) and len(row) >= 6 and all(isinstance(n, int) for n in row[:6])]

    weights = get_opt_weights(win_numbers)
    miss_counts = get_recent_miss_counts(win_numbers)
    pool = [n for n in range(1, 46) if n not in exclude | include]

    def fitness(chrom):
        total = sum(chrom)
        score = sum(weights[n] + miss_counts[n] * 0.01 for n in chrom)
        odds = sum(n % 2 for n in chrom)
        score -= abs(odds - 3) * 0.3  # 홀짝 비율 편향 보정

        if 100 <= total <= 180 and 2 <= odds <= 4:
            score += 0.5

        return score

    def valid(chrom):
        return len(set(chrom)) == 6 and all(1 <= n <= 45 for n in chrom)

    population = []
    while len(population) < 100:
        c = random.sample(pool, 6 - len(include)) + list(include)
        if valid(c):
            population.append(c)

    for _ in range(50):
        population.sort(key=fitness, reverse=True)
        next_gen = population[:20]
        while len(next_gen) < 100:
            p1, p2 = random.sample(population[:50], 2)
            cut = random.randint(1, 4)
            child = p1[:cut] + [n for n in p2 if n not in p1[:cut] and n not in include]

            while len(child) < 6 - len(include):
                available = [n for n in pool if n not in child]
                if not available:
                    break
                child.append(random.choice(available))

            if random.random() < 0.1:
                idx = random.randint(0, len(child) - 1)
                mutation_pool = [n for n in pool if n not in child]
                if mutation_pool:
                    child[idx] = random.choice(mutation_pool)

            child += list(include)
            if valid(child):
                next_gen.append(child)

        population = next_gen

    unique = []
    seen = set()
    for chrom in sorted(population, key=fitness, reverse=True):
        key = tuple(sorted(chrom))
        if key not in seen and valid(chrom) and not is_similar(key, seen):
            seen.add(key)
            unique.append(list(key))
        if len(unique) == count:
            break

    while len(unique) < count:
        fallback = random.sample(pool, 6 - len(include)) + list(include)
        fallback = tuple(sorted(fallback))
        if fallback not in seen and not is_similar(fallback, seen):
            seen.add(fallback)
            unique.append(list(fallback))

    return unique

def get_purely_cold_weights(win_numbers):
    non_appearance = get_non_appearance_counts(win_numbers)
    weights = {n: count for n, count in non_appearance.items()}
    return weights

def recommend_by_greedy(count=1, exclude=None, include=None, start_round=None, end_round=None):
    exclude, include = set(exclude or []), set(include or [])

    raw = get_all_win_numbers(start_round, end_round)
    if not raw:
        return []

    win_numbers = [row[:6] for row in raw if len(row) >= 6 and all(isinstance(n, int) for n in row[:6])]

    weights = get_purely_cold_weights(win_numbers)

    sorted_numbers = sorted(
        [n for n in range(1, 46) if n not in exclude],
        key=lambda n: -weights.get(n, 0)
    )

    results = []
    seen = set()
    max_attempts = 1000
    attempts = 0

    while len(results) < count * 3 and attempts < max_attempts:
        attempts += 1
        pick = list(include)
        for n in sorted_numbers:
            if n in pick or n in exclude:
                continue
            pick.append(n)
            if len(pick) == 6:
                break

        if len(pick) != 6:
            continue

        combo = tuple(sorted(pick))
        if combo in seen:
            continue
        seen.add(combo)

        # 가중치 점수 계산
        score = sum(weights.get(n, 0) for n in combo)
        odds = sum(n % 2 for n in combo)
        score -= abs(odds - 3) * 0.3

        # 번호 합이 100~180일 경우 가산점 부여
        total = sum(combo)
        if 100 <= total <= 180 and 2 <= odds <= 4:
            score += 0.5

        results.append((score, list(combo)))

        random.shuffle(sorted_numbers)  # 다양성 확보

    results.sort(key=lambda x: (-x[0], random.random()))
    return [combo for _, combo in results[:count]]



# 모델 정의
class LottoPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 45),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)


def get_dl_data(start_round=None, end_round=None):
    db = SessionLocal()
    try:
        win_numbers = get_all_win_numbers(start_round, end_round)
        if not win_numbers or len(win_numbers) < 2:
            return [], [], None
        sequences = [set(draw) for draw in win_numbers]
        X, y = sequences[:-1], sequences[1:]
        binarizer = MultiLabelBinarizer(classes=list(range(1, 46)))
        X_bin = binarizer.fit_transform(X)
        y_bin = binarizer.transform(y)
        return torch.tensor(X_bin, dtype=torch.float32), torch.tensor(y_bin, dtype=torch.float32), binarizer
    finally:
        db.close()


def recommend_by_dl(count=1, exclude=None, include=None, start_round=None, end_round=None):
    exclude, include = set(exclude or []), set(include or [])

    # 1. 전체 데이터 불러오기
    win_numbers = get_all_win_numbers(start_round, end_round)
    if not win_numbers:
        return []

    # 2. 보너스 제거 + 유효성 검사
    cleaned_numbers = []
    for row in win_numbers:
        if isinstance(row, list) and len(row) >= 6:
            draw = row[:6]
            if all(isinstance(n, int) and 1 <= n <= 45 for n in draw):
                cleaned_numbers.append(draw)

    # 3. 최소 학습 시퀀스 수 확인 (예: seq_len=10 기준 최소 11개 필요)
    if len(cleaned_numbers) < 11:
        return []
# 8 11 14 17 36 39
    # 4. LSTM 학습
    model = train_lotto_lstm(win_numbers, epochs=5)

    recent_seq = win_numbers[-10:]
    encoded = []
    for draw in recent_seq:
        vec = np.zeros(45)
        for n in draw:
            vec[n - 1] = 1
        encoded.append(vec)
    input_seq = np.array(encoded).reshape(1, 10, 45)

    results, seen = [], set()
    max_attempts = 1000
    attempts = 0

    while len(results) < count * 5 and attempts < max_attempts:
        attempts += 1
        pred = model.predict(input_seq, verbose=0)[0]
        top_candidates = np.argsort(pred)[::-1]

        pick = set(include)
        for n in top_candidates:
            if (n + 1) not in pick and (n + 1) not in exclude:
                pick.add(n + 1)
            if len(pick) == 6:
                break

        if len(pick) < 6:
            continue

        combo = tuple(sorted(pick))
        if combo in seen or is_similar(combo, seen):
            continue
        seen.add(combo)

        score = sum(pred[n - 1] for n in combo)
        total = sum(combo)
        odds = sum(n % 2 for n in combo)  # 홀수 개수 계산
        score -= abs(odds - 3) * 0.3

        if 100 <= total <= 180 and 2 <= odds <= 4:
            score += 0.5

        results.append((score, list(combo)))

    # 정렬 및 결과 변환
    results.sort(key=lambda x: (-x[0], random.random()))
    final = [list(map(int, combo)) for _, combo in results[:count]]

    # 부족할 경우 무작위 채움 (fallback)
    while len(final) < count:
        rand = random.sample([n for n in range(1, 46) if n not in exclude], 6 - len(include)) + list(include)
        rand = list(sorted(set(rand)))[:6]
        if rand not in final and not is_similar(rand, seen):
            final.append(rand)

    return final


def recommend_by_hybrid(count=1, exclude=None, include=None, start_round=None, end_round=None):
    strategies = [
        recommend_by_stat,
        recommend_by_ml,
        recommend_by_opt,
        recommend_by_greedy,
        recommend_by_dl,
    ]

    exclude, include = set(exclude or []), set(include or [])
    seen = set()
    scored_results = []

    for strategy in strategies:
        try:
            result = strategy(count=count * 2, exclude=exclude, include=include,
                              start_round=start_round, end_round=end_round)

            for combo in result:
                combo_tuple = tuple(sorted(combo))
                if combo_tuple not in seen and not is_similar(combo_tuple, seen):
                    seen.add(combo_tuple)
                    score = sum(combo)
                    if 100 <= score <= 180:
                        score += 0.5
                    scored_results.append((score, combo_tuple))
        except Exception as e:
            print(f"[HYBRID] {strategy.__name__} 실패: {e}")

    scored_results.sort(key=lambda x: (-x[0], random.random()))
    top_combos = [list(map(int, combo)) for _, combo in scored_results[:count]]

    # 부족할 경우 보완
    while len(top_combos) < count:
        rand = random.sample([n for n in range(1, 46) if n not in exclude], 6 - len(include)) + list(include)
        rand = list(sorted(set(rand)))[:6]
        rand_tuple = tuple(rand)
        if rand_tuple not in seen and not is_similar(rand_tuple, seen):
            seen.add(rand_tuple)
            top_combos.append(rand)

    return top_combos



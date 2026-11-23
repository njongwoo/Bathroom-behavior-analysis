# AI를 활용한 공중화장실 소변기 선택 행동 패턴 분석
******이해하기 쉬운 문장으로 재작성**********
AI+X: Deep Learning — Final Project

## 목차
1. Proposal
2. Datasets
3. Methodology
4. Evaluation & Analysis
5. Related Work
6. Conclusion

## I. Proposal

### 연구 동기
공중화장실 이용자는 소변기의 위치, 주변 이용자의 존재, 혼잡도 등에 따라 서로 다른 선택을 할 수 있다.

본 프로젝트에서는 가상의 화장실 환경을 구성하고, 인공지능을 활용해 소변기 선택 행동 패턴을 분석하고자 한다.

### 연구 목표
- 소변기별 이용 빈도 분석
- 혼잡도에 따른 선택 행동 비교
- AI 학습 기법을 이용한 행동 패턴 분석

## II. Datasets

Data 생성 환경 :가장 기본적인 화장실 배치(일렬, 장애인 소변기 한개)로 데이터 구성 / 가능하면 한양대학교 교내에 있는 실제 배치(ㄷ자 배치, 2열 배치 등 고려)
변수 -> 
        Id(화장실 이용자 식별 번호)
        Occupancy/bathroom status(선택할 때 전체 소변기의 상태)
        Distance(입구-> 소변기까지의 거리)
        Adj occupancy(양 옆 소변기의 점유상태)
        waiting time (예상 대기시간)
        Urinal number(선택한 소변기 번호)
행동 특성 분석
-이용 기록: 한 번 방문했을 때 선택한 소변기를 기록
-행동 특성: 한 이용자의 여러번의 방문기록을 토대로 평균적인 선택 패턴을 계산
==> 최종적으로 생성된 데이터로 K-Means 알고리즘 -> 비슷한 선택 특성의 이용자를 그룹으로 분류 / 분석
## III. Methodology(K-Means 알고리즘 소개/적용방법)

1. K-means 알고리즘 원리:
2. 프로젝트에 적용 방법: 

## IV. Evaluation & Analysis
실험가상 이용자 100명 -> 인당 20번 (소변기 4개 일렬 배치)
"""Generate illustrative synthetic urinal choices for an educational project.

The results describe the assumptions below, not observed human behavior.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

N_USERS = 100
VISITS_PER_USER = 20
N_URINALS = 4
SEED = 2026
OUTPUT_DIR = Path(__file__).resolve().parent
rng = np.random.default_rng(SEED)

# For illustration: 1 is nearest the door, 4 is farthest.
distances = np.arange(1, N_URINALS + 1, dtype=float)
records = []

# Individuals have continuous preferences, NOT assigned cluster labels.
# A high distance_weight makes nearby stalls more attractive.
# A high privacy_weight makes a free urinal beside an occupied one less attractive.
for user_id in range(1, N_USERS + 1):
    distance_weight = float(rng.uniform(0.1, 1.8))
    privacy_weight = float(rng.uniform(0.0, 3.0))
    for visit in range(1, VISITS_PER_USER + 1):
        busy = bool(rng.random() < 0.4)
        occupancy_probability = 0.60 if busy else 0.20
        occupied = rng.random(N_URINALS) < occupancy_probability

        # Model immediate choice among free urinals only; no waiting model.
        # Resample a rare fully occupied scene, so a legal choice exists.
        while occupied.all():
            occupied = rng.random(N_URINALS) < occupancy_probability

        adjacent_occupied = np.array([
            int((j > 0 and occupied[j-1]) or
                (j < N_URINALS-1 and occupied[j+1]))
            for j in range(N_URINALS)
        ])

        scores = -distance_weight * distances - privacy_weight * adjacent_occupied
        scores[occupied] = -np.inf
        # Softmax expresses stochastic choice; it does not learn from real people.
        free = ~occupied
        shifted = scores[free] - max(scores[free])
        probs = np.exp(shifted) / np.exp(shifted).sum()
        selected = int(rng.choice(np.flatnonzero(free), p=probs))
        nearest_free = int(np.flatnonzero(free)[0])

        records.append({
            "user_id": user_id,
            "visit": visit,
            "busy_period": int(busy),
            "occupied_1": int(occupied[0]),
            "occupied_2": int(occupied[1]),
            "occupied_3": int(occupied[2]),
            "occupied_4": int(occupied[3]),
            "occupied_count": int(occupied.sum()),
            "selected_urinal": selected + 1,
            "selected_distance": int(distances[selected]),
            "selected_adjacent_occupied": int(adjacent_occupied[selected]),
            "chose_nearest_available": int(selected == nearest_free),
        })

visits = pd.DataFrame.from_records(records)
summary = (
    visits.groupby("user_id", as_index=False)
    .agg(
        mean_selected_distance=("selected_distance", "mean"),
        adjacent_occupied_choice_rate=("selected_adjacent_occupied", "mean"),
        nearest_available_choice_rate=("chose_nearest_available", "mean"),
        mean_occupied_count=("occupied_count", "mean"),
    )
)

visits.to_csv(OUTPUT_DIR / "synthetic_visits.csv", index=False, encoding="utf-8-sig")
summary.to_csv(OUTPUT_DIR / "user_features.csv", index=False, encoding="utf-8-sig")

counts = visits["selected_urinal"].value_counts().reindex(range(1, N_URINALS+1), fill_value=0)
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.bar(counts.index.astype(str), counts.values)
ax.set(xlabel="Urinal number (1 = nearest entrance)", ylabel="Number of visits",
       title="Synthetic urinal selection frequency")
for i, c in enumerate(counts.values):
    ax.text(i, c+8, f"{c} ({100*c/len(visits):.1f}%)", ha="center", fontsize=9)
ax.set_ylim(0, max(counts)*1.15)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "urinal_frequency.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.scatter(summary["mean_selected_distance"],
           summary["adjacent_occupied_choice_rate"], alpha=0.7, s=28)
ax.set(xlabel="Average chosen distance (1–4)",
       ylabel="Fraction choosing next to occupied urinal",
       title="Synthetic user behavior features (100 users)")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "behavior_features.png", dpi=180)
plt.close(fig)

assert len(visits) == N_USERS*VISITS_PER_USER
assert len(summary) == N_USERS
assert all(visits.apply(lambda r: r[f"occupied_{int(r['selected_urinal'])}"] == 0, axis=1))
print("Generated synthetic visits:", len(visits))
print("Generated user feature rows:", len(summary))
print("Selections per urinal:", counts.to_dict())
print("Files saved in:", OUTPUT_DIR)

결과: https://chatgpt.com/s/w_6b04253b78e0819192f3e3d5cead6f5d
분석: 입구에서 가장 가까운 소변기 1번 선택이 963/2000 비율로 많았음 -> 행동 패턴 분석은 아직 미완료

## V. Related Work

관련 연구 및 참고 문헌을 조사할 예정.

## VI. Conclusion

최종 실험 결과와 한계점을 정리할 예정.

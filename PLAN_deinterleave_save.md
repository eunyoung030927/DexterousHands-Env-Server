# Plan: Bi-DexHands 데이터 저장 방식 변경 (인터리빙 → 환경별 순차)

## 목표
데이터를 저장할 때 인터리빙을 해제하여, 로딩 시 stride slicing 없이 Forge처럼 단순하게 불러올 수 있게 한다.

---

## 현재 상태

### 저장 (`ppo_collect.py`)
- 매 step마다 `(num_envs, obs_dim)` 텐서를 리스트에 append
- `np.vstack` → `(total_steps, obs_dim)` 형태로 flat 저장
- 결과: `[env0_t0, env1_t0, env2_t0, env0_t1, env1_t1, env2_t1, ...]` (인터리빙)

### 로딩 (`load_bidexhands`)
- `raw_data[k][env_i::num_envs]` stride slicing으로 환경별 분리
- trimming (나머지 잘라내기) 필요
- done 기준 에피소드 분할

---

## 변경 계획

### 1단계: 저장 코드 수정 (`ppo_collect.py:156-165`)

**변경 내용**: `np.vstack` 후 reshape + transpose로 환경별 순차 배치

```python
# 기존: (total,) = [e0_t0, e1_t0, e2_t0, e0_t1, e1_t1, e2_t1, ...]
# 변경: (total,) = [e0_t0, e0_t1, ..., e1_t0, e1_t1, ..., e2_t0, e2_t1, ...]

num_envs = self.vec_env.num_envs
num_steps = len(arr) // num_envs
# (num_steps, num_envs, ...) → (num_envs, num_steps, ...) → (num_envs * num_steps, ...)
arr[:valid].reshape(num_steps, num_envs, *shape).swapaxes(0, 1).reshape(-1, *shape)
```

- `is_testing` 블록 (line 156-165)과 `else` 블록 (line 224-233) **둘 다** 수정
- `num_envs.npy` 파일 추가 저장 (로딩 시 환경 수 자동 감지용)

### 2단계: 로딩 코드 수정 (`train_collection.py:load_bidexhands`)

**변경 내용**: Forge 로딩과 동일한 방식으로 단순화

- stride slicing 제거
- trimming 제거
- `num_envs` 파라미터 불필요해짐
- done 기준으로 에피소드 분할만 수행 (Forge와 동일 로직)

```python
def load_bidexhands(path_to_dataset, max_episode_use=-1, image_obs=False):
    keys = ["states", "next_states", "actions", "rewards", "dones"]
    raw_data = {k: np.load(...) for k in keys}

    # Forge와 동일하게 done 기준 에피소드 분할
    done_indices = np.where(raw_data["dones"] == 1)[0]
    prev_idx = 0
    for done_idx in done_indices:
        end_idx = done_idx + 1
        # 에피소드 추출 & append
        ...
```

---

## 수정 대상 파일
| 파일 | 변경 내용 |
|------|----------|
| `bidexhands/algorithms/offrl/ppo_collect/ppo_collect.py` | 저장 시 de-interleave 추가 (2곳) |
| `bidexhands/train_collection.py` (`load_bidexhands`) | stride slicing 제거, Forge식 로딩으로 단순화 |

## 주의사항
- **기존 데이터와 호환 불가**: 이미 저장된 인터리빙 데이터는 새 로딩 코드로 읽을 수 없음. 데이터 재수집 필요.
- `else` 블록(학습 중 중간 저장, line 224-233)도 동일하게 수정해야 함
- `data_size` truncation은 de-interleave 이후에 적용

---

## 질문
1. 기존 인터리빙 데이터와의 하위 호환이 필요한가? (필요하면 로딩 시 포맷 감지 로직 추가)
2. `load_bidexhands`와 `load_forge`를 하나로 통합할 것인가?

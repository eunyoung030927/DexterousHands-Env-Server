# Plan: Bi-DexHands 데이터 저장 방식 변경

## 목표
1. 인터리빙 해제: 저장 시 환경별 순차 배치로 변경하여 로딩을 단순화
2. 에피소드 단위 수집: `data_size`를 에피소드 수 기준으로 수집할 수 있도록 변경

---

## 현재 상태

### 저장 (`ppo_collect.py`)
- 매 step마다 `(num_envs, obs_dim)` 텐서를 리스트에 append
- `np.vstack` → `(total_steps, obs_dim)` 형태로 flat 저장
- 결과: `[env0_t0, env1_t0, env2_t0, env0_t1, env1_t1, env2_t1, ...]` (인터리빙)
- 종료 조건: transition 수 기준 (`nums += num_envs`, `while nums < data_size`)

### 로딩 (`load_bidexhands`)
- `raw_data[k][env_i::num_envs]` stride slicing으로 환경별 분리
- trimming (나머지 잘라내기) 필요
- done 기준 에피소드 분할

---

## 변경 계획

### 1단계: 저장 코드 수정 — 에피소드 단위 수집 (`ppo_collect.py` is_testing 블록)

**변경 내용**: transition 카운트 → 완료된 에피소드 카운트 기반으로 변경. 각 환경별 임시 버퍼에 transition을 쌓다가, done 시 완성된 에피소드만 저장 리스트에 추가.

```python
# 기존: while nums < self.data_size: nums += num_envs (transition 단위)
# 변경: while num_episodes < self.data_size: done 발생 시 num_episodes += 1 (에피소드 단위)

# 환경별 임시 버퍼
ep_buffers = [{'states': [], 'next_states': [], 'actions': [], 'rewards': [], 'dones': []}
              for _ in range(num_envs)]

while num_episodes < self.data_size:
    # step 실행
    # 각 환경의 임시 버퍼에 transition append
    for i in range(num_envs):
        ep_buffers[i]['states'].append(obs_np[i])
        ...
        if dones_np[i] > 0:
            # 완성된 에피소드를 저장 리스트에 추가
            states_off.append(np.array(ep_buffers[i]['states']))
            ...
            num_episodes += 1
            ep_buffers[i] = {초기화}
```

- 완성된 에피소드만 저장되므로 잘린 에피소드 없음
- 저장 결과: 환경별 순차 배치가 자연스럽게 달성됨 (에피소드 단위로 연속 저장)
- `data_size`의 의미가 "수집할 에피소드 수"로 변경됨
- `[0:data_size]` truncation 불필요 (정확히 `data_size`개 에피소드만 수집)

---

## 수정 대상 파일
| 파일 | 변경 내용 |
|------|----------|
| `bidexhands/algorithms/offrl/ppo_collect/ppo_collect.py` | is_testing 블록만 에피소드 단위 수집으로 변경 (else 블록은 미수정) |
| `bidexhands/cfg/ppo_collect/config.yaml` | `data_size` 값을 에피소드 수로 설정 (예: 1000) |

## 주의사항
- `data_size`의 의미가 transition 수 → 에피소드 수로 바뀌므로, config 값도 함께 변경 필요 (1000000 → 1000 등)
- 에피소드 단위 수집 시 환경별 임시 버퍼 메모리 사용량은 미미 (에피소드 하나분)
- 저장 시 `[0:data_size]` truncation 제거 필요 (`ppo_collect.py:161-165`)
  - 현재: `np.save(..., states_off[0:self.data_size])` — transition 인덱스 기준으로 자름
  - 문제: 에피소드 단위 수집에서는 `data_size=1000`이면 1000번째 transition에서 잘려버림
  - 변경: `np.save(..., states_off)` — 완성된 에피소드만 수집했으므로 슬라이싱 없이 전체 저장

## 추가 변경: 성공한 에피소드만 저장

### 배경
- 현재 코드는 done 발생 시 성공/실패 구분 없이 모든 에피소드를 저장
- done 발생 조건: 성공(`goal_dist < 0.03`), 낙하(`object_pos[:, 2] <= 0.2`), 타임아웃(`progress_buf >= max_episode_length`)
- 모방학습에는 성공 데이터만 필요

### 성공 판별 방법
- `vec_env.step()` 반환값 `infos`에 `infos['successes']` 텐서가 포함됨 (`shadow_hand_over.py:575`)
- `infos['successes'][i] == 1`이면 해당 환경의 에피소드가 성공
- done 시점에 이 값을 확인하여 성공한 에피소드만 저장

### 변경 내용 (`ppo_collect.py` is_testing 블록)
- done 발생 시 `infos['successes'][i]` 확인
- 성공(`== 1`)인 경우에만 저장 리스트에 추가하고 `num_episodes += 1`
- 실패인 경우 임시 버퍼만 초기화하고 저장하지 않음

```python
if dones_np[i] > 0:
    if num_episodes < self.data_size and infos['successes'][i] > 0:
        # 성공한 에피소드만 저장
        states_off.append(...)
        num_episodes += 1
    ep_buffers[i] = {초기화}
    cur_reward_sum[i] = 0
```

### 주의사항
- 성공률이 낮으면 수집 시간이 크게 늘어날 수 있음
- 수집 진행 상황 출력 추가 권장 (현재 성공 에피소드 수 / 목표)

## 결정 사항
- `--test`로만 데이터 수집 (else 블록 미수정)
- `data_size`에 에피소드 수를 넣으면 해당 수만큼 수집
- 성공한 에피소드만 저장
- 로딩 코드는 클라이언트쪽 사항이므로 이 계획에서 제외

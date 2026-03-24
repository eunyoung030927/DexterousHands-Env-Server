# Bi-DexHands 서버 통합 계획

## 현황

현재 3개의 서버 파일이 존재하며, 각각 독립적으로 환경을 ZMQ로 서빙한다.

| 파일 | 태스크 | 프레임워크 | Agent 모드 | 포트 |
|------|--------|-----------|-----------|------|
| `server_over_env.py` | ShadowHandOver | IsaacGym (Bi-DexHands) | SingleAgent | 5555 |
| `server_bottlecap_env.py` | ShadowHandBottleCap | IsaacGym (Bi-DexHands) | MultiAgent | 5555 |
| `forge_server_env.py` | Isaac-Forge-PegInsert | Isaac Lab (gymnasium) | SingleAgent | 5556 |

### 핵심 차이점

1. **환경 생성 방식이 완전히 다름**
   - Over/BottleCap: `parse_task()` → `VecTaskPython` / `MultiVecTaskPython` (IsaacGym 직접 사용)
   - Forge: `gymnasium.make()` → Isaac Lab 환경 (Omniverse 기반)

2. **step() 반환값 형식이 다름**
   - Over (SingleAgent): `(obs, reward, done, info)` — 4-tuple
   - BottleCap (MultiAgent): `(obs, state, reward, done, info, None)` — 6-tuple
   - Forge: `(obs, reward, terminated, truncated, info)` — Gymnasium 5-tuple

3. **클라이언트로 보내는 응답 형식이 다름**
   - Over: `(obs, rew, terminated, truncated, info, mass_str)` — 6-tuple
   - BottleCap: `(obs, rew, terminated, truncated, info)` — 5-tuple (mass in info)
   - Forge: `(obs, reward, terminated, truncated, infos)` — 5-tuple (배열 기반)

---

## 통합 목표

`server_over_env.py`와 `server_bottlecap_env.py`를 하나의 `server_env.py`로 통합한다.
(`forge_server_env.py`는 Isaac Lab 기반이라 프레임워크 자체가 다르므로 통합 대상에서 제외)

### 통합 후 모습
```
python server_env.py --task ShadowHandOver        # Over 환경 실행
python server_env.py --task ShadowHandBottleCap   # BottleCap 환경 실행
python server_env.py --task ShadowHandCatchUnderarm  # 다른 태스크도 가능
```

---

## 통합 설계

### 1단계: CLI 인터페이스 통일

```
python server_env.py \
  --task ShadowHandBottleCap \
  --port 5555 \
  --num_envs 1 \
  --episode_length 125 \
  --headless
```

- `--task`: 태스크 이름 (필수, 기본값 없음 또는 ShadowHandOver)
- `--port`: ZMQ 포트 (기본 5555)
- `--num_envs`: 환경 수 (기본 1)
- `--episode_length`: 에피소드 길이 (기본값은 태스크별 설정 따름)
- `--headless`: 렌더링 비활성화 (기본 활성)

### 2단계: 태스크별 설정 관리

태스크 이름에 따라 자동으로 적절한 설정을 결정하는 레지스트리 딕셔너리:

```python
TASK_CONFIGS = {
    "ShadowHandOver": {
        "task_type": "Python",        # SingleAgent
        "cfg_train": "bidexhands/cfg/mappo/config.yaml",
        "cfg_env": "bidexhands/cfg/ShadowHandOver.yaml",
        "episode_length": 75,
        "num_agents": 1,
        "obs_per_agent": None,        # 단일 obs 텐서 그대로 사용
        "action_dim_per_agent": None,  # 단일 action 텐서 그대로 사용
    },
    "ShadowHandBottleCap": {
        "task_type": "MultiAgent",
        "cfg_train": "cfg/ppo/config.yaml",
        "cfg_env": "cfg/ShadowHandBottleCap_server.yaml",
        "episode_length": 125,
        "num_agents": 2,
        "obs_per_agent": 221,
        "action_dim_per_agent": 26,
        "asymmetric_observations": True,
    },
    # 추후 확장 시 여기에 추가
}
```

### 3단계: 환경 어댑터 패턴

SingleAgent와 MultiAgent의 step/reset 차이를 흡수하는 어댑터 레이어:

```python
class EnvAdapter:
    """IsaacGym 환경의 SingleAgent/MultiAgent 차이를 추상화"""

    def __init__(self, env, task, task_config, device):
        self.env = env
        self.task = task
        self.is_multi_agent = (task_config["task_type"] == "MultiAgent")
        self.num_agents = task_config["num_agents"]
        self.action_dim_per_agent = task_config.get("action_dim_per_agent")
        self.device = device

    def reset(self) -> np.ndarray:
        """통일된 1D numpy 배열 반환"""
        ...

    def step(self, action_flat: np.ndarray) -> tuple:
        """
        입력: 1D action (모든 agent 합쳐진)
        출력: (obs_flat, reward, terminated, truncated, info)
        - obs_flat: 1D numpy (모든 agent obs 합쳐진)
        - reward: float (agent 평균)
        - terminated: bool
        - truncated: bool
        - info: dict
        """
        ...
```

**SingleAgent 경로** (Over 등):
- reset: `env.reset()` → `(num_envs, obs_dim)` → `[0]` → 1D
- step: `action.unsqueeze(0)` → `env.step()` → `(obs, rew, done, info)` 4-tuple 처리

**MultiAgent 경로** (BottleCap 등):
- reset: `env.reset()` → `(obs, state, None)` → `obs.flatten()` → 1D
- step: `action_flat` → `[agent0_action, agent1_action]` 리스트로 분리 → `env.step()` → 6-tuple 처리
- reward: `mean()`, done: `any()`

### 4단계: 클라이언트 응답 형식 통일

**forge_server_env.py의 형식을 표준으로 채택:**

```python
# reset 응답
("ok", obs)           # obs: np.ndarray, 1D

# step 응답
("ok", (obs, reward, terminated, truncated, info))
# obs: np.ndarray 1D
# reward: float
# terminated: bool
# truncated: bool
# info: dict (mass 등 부가정보 포함)
```

- Over 서버의 기존 6-tuple `(obs, rew, term, trunc, info, mass_str)` → mass를 info 안에 넣어 5-tuple로 통일
- BottleCap 서버는 이미 이 형식이므로 그대로

### 5단계: mass 조회 공통화

두 서버 모두 동일한 mass 조회 로직 사용 → 헬퍼 함수로 추출:

```python
def get_object_mass(task) -> str:
    try:
        env_ptr = task.envs[0]
        handle = task.gym.find_actor_handle(env_ptr, "object")
        if handle != isaacgym.gymapi.INVALID_HANDLE:
            props = task.gym.get_actor_rigid_body_properties(env_ptr, handle)
            if props:
                return f"{props[0].mass:.4f} kg"
    except:
        pass
    return "N/A"
```

### 6단계: 작업 디렉토리 처리 통일

- BottleCap: `os.chdir(bidexhands/)` 후 상대경로 사용
- Over: `os.chdir(프로젝트 루트)` 후 `bidexhands/` 접두사 사용

→ **통일**: 항상 프로젝트 루트(`DexterousHands/`)에서 실행, 설정 파일 경로는 태스크 레지스트리에서 관리

---

## 파일 구조 (통합 후)

```
DexterousHands/
├── server_env.py              # [NEW] 통합 서버 (Over + BottleCap + 기타 Bi-DexHands 태스크)
├── forge_server_env.py        # [유지] Isaac Lab Forge 전용 서버
├── server_over_env.py         # [보존] 기존 파일 (백업, 추후 삭제 가능)
├── server_bottlecap_env.py    # [보존] 기존 파일 (백업, 추후 삭제 가능)
```

---

## 통합 서버 전체 흐름

```
1. CLI 파싱 (--task, --port, --num_envs, --episode_length, --headless)
2. TASK_CONFIGS에서 태스크 설정 로드
3. 작업 디렉토리 및 sys.path 설정
4. bidexhands 유틸리티로 환경 생성 (task_type에 따라 SingleAgent / MultiAgent)
5. EnvAdapter로 환경 래핑
6. ZMQ 서버 루프:
   - reset → adapter.reset() → 1D obs 반환
   - step  → adapter.step(action) → (obs, rew, term, trunc, info) 반환
   - info  → obs_dim, act_dim, task 정보 반환
   - close → 종료
```

---

## 클라이언트 영향

- **Over 클라이언트**: step 응답이 6-tuple → 5-tuple로 변경됨. `mass_str`을 `info["mass"]`에서 읽도록 수정 필요
- **BottleCap 클라이언트**: 변경 없음 (이미 5-tuple 사용 중)
- 새로운 `info` 커맨드 추가로 클라이언트가 obs_dim/act_dim을 서버에서 동적으로 조회 가능

---

## 구현 순서

1. `server_env.py` 파일 생성, TASK_CONFIGS 정의
2. EnvAdapter 클래스 구현 (SingleAgent/MultiAgent 분기)
3. ZMQ 서버 루프 구현 (forge_server_env.py의 구조 참고)
4. Over 환경으로 테스트
5. BottleCap 환경으로 테스트
6. 클라이언트 코드 응답 형식 맞춤 수정 (Over 클라이언트)

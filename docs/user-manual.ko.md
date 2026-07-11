# CameraE2E Integrated 사용자 매뉴얼

## 1. 사용 목적

CameraE2E는 카메라 시스템 연구원을 위한 로컬 의사결정 Workbench입니다. 요구사항
정의, 카메라 시뮬레이션, 민감도 분석, 파라미터 최적화, 후보 검증, 가상 RAW 생성,
근거 보고서 작성을 지원합니다.

현재 릴리스의 주 용도는 상대 비교와 재현 가능한 연구입니다. 측정 calibration이
없는 analytic, 공개자료 기반, geometric 또는 solver 결과를 제품 sign-off로
해석하면 안 됩니다.

## 2. 설치와 실행

Git, Python 3.12, Node.js 20 이상이 필요합니다.

```bash
git clone https://github.com/SeongcheolJeong/CameraE2E-Integrated.git
cd CameraE2E-Integrated
./tools/workbench.sh bootstrap
```

Workbench는 `http://127.0.0.1:5175`, API 문서는
`http://127.0.0.1:8010/docs`에서 확인합니다. 이후에는 다음 명령을 사용합니다.
Python 3.12가 `PATH`에 없다면 먼저
`CAMERAE2E_PYTHON=/path/to/python3.12`를 지정합니다.

```bash
./tools/workbench.sh start
./tools/workbench.sh status
./tools/workbench.sh stop
```

수동 설치가 필요한 경우:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[workbench,yolo,dev]"
cd camerae2e-workbench
npm ci
```

## 3. 외부 데이터가 없을 때

1. 기본 ADAS 또는 general 프로젝트를 선택합니다.
2. 요구사항과 baseline camera를 확인합니다.
3. analytic fidelity로 baseline simulation을 실행합니다.
4. geometry, clipping, signal, optics, fidelity gate를 확인합니다.
5. 준비된 raw-quality 목적에 대해서만 sensitivity/optimization을 실행합니다.
6. 검증된 후보에서 소규모 RAW dataset과 report를 생성합니다.

YOLO 모델과 label이 없으면 perception optimization은 비활성화됩니다. 이미지 품질
proxy를 detector mAP처럼 표시하지 않기 위한 의도된 동작입니다.

## 4. 주요 워크플로

### Configure

먼저 카메라 mission을 정의합니다. HFOV, pixel pitch, sensor geometry, CFA/QE,
exposure, OCL/CRA, lens PSF, ISP, fidelity가 baseline의 핵심입니다.

| 항목 | 주요 영향 |
|---|---|
| HFOV | 장면 범위와 물체의 pixel 점유율 |
| Pixel pitch | sampling, 수광 면적, noise/full-well trade-off |
| CFA/QE | 파장별 sampling과 demosaic 특성 |
| OCL/CRA | field에 따른 pixel stack angular response |
| PSF radius | 선택된 모델에서의 광학 blur 범위 |
| Exposure | signal, clipping, motion risk |
| CCM | 색 변환; 자유 최적화에는 color evidence가 필요 |
| Fidelity | analytic, LUT, solver, calibrated 근거 수준 |

### Run Simulation

현재 설정의 카메라 한 개를 평가합니다. 각 stage 결과, metric, requirement gate,
fidelity, seed, lineage를 저장하며 다른 설정을 자동 탐색하지는 않습니다.

### Sensitivity와 Optimization

Sensitivity는 선언된 축의 영향도를 조사합니다. Optimization은 objective와 hard
constraint를 이용해 여러 후보를 평가합니다. 최종 evidence budget까지 평가된
후보만 winner가 될 수 있습니다. `indistinguishable`은 차이가 통계적·실용적으로
충분하지 않다는 의미입니다.

ADAS 모드는 실제 YOLO와 KITTI label을 사용해 source-model preflight, geometry
검증, rankability gate, successive-halving, perturbation robustness, bootstrap
비교를 실행합니다.

### RAW Dataset Factory

선택된 후보를 각 scene에 다시 실행하여 다음 구조를 생성합니다.

```text
manifest.json
metadata.jsonl
raw/*.npz
rgb/*.png
labels/*.json
optional raw_tiff/*.tiff
optional stages/*.npz
```

manifest에는 config, scene, seed, source hash, commit, fidelity와 validation이
기록됩니다. label은 camera output과 같은 geometry transform을 적용하며, 알 수
없는 물체의 label을 자동 생성하지 않습니다.

### Report

요구사항, baseline, search space, gate, 후보 순위, 불확실성, fidelity 한계,
artifact와 dataset을 정리합니다. warning과 `not_evaluable`도 결과의 일부로
해석해야 합니다.

## 5. KITTI와 YOLO 연결

backend를 시작하기 전에 경로를 지정합니다.

```bash
export CAMERAE2E_KITTI_ROOT=/data/kitti-yolo
export CAMERAE2E_YOLO_MODEL=/models/kitti-yolo.pt
./tools/workbench.sh restart
```

```text
kitti-yolo/
  images/train/
  images/val/
  labels/train/
  labels/val/
```

이미지와 label의 stem이 일치해야 하며 모델, label, class mapping이 같은 task를
가리켜야 합니다. 일반 COCO YOLO도 실험에는 사용할 수 있지만 KITTI-trained ADAS
benchmark로 보고하면 안 됩니다.

## 6. 재현성과 검증

프로젝트는 `project.toml`, `project.db`, content-addressed artifact, run, export,
report로 구성됩니다. 결과 공유 시 project, benchmark manifest, model/source hash,
seed, config revision, Git commit을 함께 보존합니다.

```bash
./tools/workbench.sh check
.venv/bin/camerae2e doctor /path/to/project
.venv/bin/python -m pytest
cd camerae2e-workbench && npm run build
```

오류 해결은 [Troubleshooting](troubleshooting.md), 물리적 효용성과 한계는
[Research Boundaries](research-boundaries.md)를 참고하십시오.

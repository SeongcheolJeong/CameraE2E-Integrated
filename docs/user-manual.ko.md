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

### Lens/Sensor Component Explorer

`Design Space`의 `Camera Module` 영역에서 `Find & Compare`를 누릅니다.

1. Lens는 회사, 특허, focal length, F-number, FOV, geometric PSF 유무로 검색합니다.
2. Sensor는 제조사, pixel pitch, resolution, CFA, DTI, simulation readiness로
   검색합니다. 기본값은 `Simulation-ready only`입니다.
3. Lens와 sensor를 하나씩 선택하면 현재 study 요구사항에 대한 compatibility를
   즉시 계산합니다.
4. 2~4개 조합을 compare tray에 넣고 hard gate와 Pareto 표시를 확인합니다.
5. `Compare on Scene`을 실행하면 모든 조합을 같은 scene과 seed로 다시 계산합니다.
6. 호환 조합을 baseline으로 적용하면 module descriptor artifact가 생기고 study
   revision이 증가합니다.

Gate에는 frame sensor/CFA 모델 지원, image circle, RayOptics field domain, HFOV,
거리별 물체 pixel 수, diffraction sampling, pixel bandwidth, rolling-shutter 가정이
포함됩니다. `not_evaluable`은 pass가 아닙니다. Pareto 표시는 현재 analytic support
metric에서 지배되지 않는 후보라는 뜻이며 자동 winner 선언이 아닙니다.

Native sensor 해상도는 geometry와 hardware gate에 사용합니다. 로컬 실행 시간과
메모리를 제한하기 위해 실제 image simulation은 CFA 정렬된 최대 640 x 360 readout을
사용하고 이 차이를 metadata에 기록합니다. 원본 DB는 read-only이고, project에는
선택한 source ID, module descriptor, compatibility, provenance와 결과 artifact만
저장됩니다.
Simulation sampling pitch를 키워 전체 sensor 물리 폭을 유지하고 fill factor를 줄여
native photodiode 면적을 유지합니다. 따라서 이 모델은 binning이 아니라 sparse
sampling proxy입니다.

현재의 중요한 경계:

- Event/NIR/SWIR sensor는 검색과 검토는 가능하지만 전용 acquisition physics가 없어
  frame-RAW simulation에서는 차단됩니다.
- CFA가 없으면 임의 Bayer로 치환하지 않습니다. Bayer/RGGB Bayer/Quad Bayer/
  Tetracell Bayer로 확인된 record만 직접 configure할 수 있습니다.
- DB 이름으로 센서별 measured QE, noise, full-well을 추정하지 않습니다. 기존
  baseline QE profile을 유지하고 그 사실을 metadata에 기록합니다.
- RayOptics PSF는 geometric ray histogram이며 diffraction 또는 measured MTF가
  아닙니다.
- MP 값에서 유도한 geometry는 `resolution_mp_16_9_proxy`로 표시합니다.
- Sparse simulation은 FOV와 sample당 수광 면적은 유지하지만 native spatial
  resolution과 native CFA phase 관계까지 보존하지는 않습니다.

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
optional raw_uint16/*.npy
```

manifest에는 config, scene, seed, source hash, commit, fidelity와 validation이
기록됩니다. label은 camera output과 같은 geometry transform을 적용하며, 알 수
없는 물체의 label을 자동 생성하지 않습니다.

Workbench의 Dataset Factory에서는 다음 순서로 생성합니다.

1. `Adapter`, dataset root와 split을 선택하고 `Inspect & Estimate`를 실행합니다.
2. baseline, best, top 또는 Pareto camera profile을 선택합니다.
3. 기본 `source_bounded`는 입력 RGB가 가진 공간 정보 범위 안에서 RAW를 만듭니다.
4. `target_readout_proxy`는 목표 readout 크기를 사용할 수 있지만 원본보다 큰 경우
   `upsampled_scene_proxy=true`가 기록되며 새로운 공간 정보가 생긴 것으로 해석하지
   않습니다.
5. exposure bracket과 noise 반복 수를 정한 뒤 export합니다. 같은 source frame의
   모든 camera/exposure/noise variant는 같은 split을 유지합니다.

KITTI calibration의 `P2`가 있으면 source intrinsics에서 target HFOV와 principal
point로 2D pinhole warp를 적용하고 bbox도 같은 변환을 사용합니다. depth가 없으므로
parallax, disocclusion, 시점 이동은 생성하지 않습니다. `P2`가 없으면 중심 정렬
geometry proxy를 사용하고 manifest에 calibration 누락이 남습니다.

RAW NPZ에는 `raw`, `sensor_digital`, `black_level`, `white_level`, `bit_depth`,
`cfa_pattern`이 포함됩니다. 이 파일은 표준 DNG가 아니며 단위는
`simulator_sensor_response`입니다.

### Report

요구사항, baseline, module 비교, search space, gate, 후보 순위, 불확실성,
fidelity 한계, artifact와 dataset을 정리합니다. warning과 `not_evaluable`도
결과의 일부로 해석해야 합니다.

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
  calib/train/       # optional KITTI P2 files
  calib/val/         # optional
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

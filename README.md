# 🌾 MYSTIC 노지 작물 해충 AI 진단 앱 (YOLOv8-Large v5 + Qwen2-VL-2B LoRA)

> **한동대학교 AI·컴퓨터공학부 캡스톤 디자인 — MYSTIC 팀 (여민기, 신태현, 문경빈)**  
> **지도교수**: 최희열 교수님 | **참여기업체**: (주)스톰힐 이노베이션 (Stormhill Innovation)

노지 작물 미소·유사 해충의 정밀 진단을 위한 **2-Stage 계층적 멀티모달 AI 파이프라인(`1단계 YOLOv8-Large v5` ➔ `2단계 Qwen2-VL-2B LoRA Rank 16`)**을 탑재한 크로스플랫폼 **Flutter 애플리케이션 & FastAPI 백엔드** 프로젝트입니다.

---

## 📱 주요 화면 및 기능 구성 (3탭 네비게이션 바)

1. **🏠 홈 (`HomeScreen`)**
   * 2-Stage 계층적 멀티모달 파이프라인 상태 대시보드
   * 대표 노지 해충 5종(`큰28점박이무당벌레`, `썩덩나무노린재 1령 약충 군집`, `열대거세미나방 유사종 자가교정`, `비단노린재`, `검거세미밤나방`) 원클릭 파이프라인 시연
   * 최근 진단 히스토리 자동 기록
2. **📷 해충 카메라 (`PestCameraScreen`)**
   * 카메라 직접 촬영(`ImageSource.camera`) / 앨범 사진 선택(`ImageSource.gallery`) / 샘플 프리셋 선택
   * **[1단계] YOLOv8-Large v5 객체 탐지**: 이미지 위 실시간 빨간색 시각적 바운딩 박스(`VisualBBoxPainter`) 오버레이 및 탐지 개체 수·확신도 표시
   * **[2단계] Qwen2-VL-2B LoRA (Rank 16) 정밀 진단 및 방제 처방**:
     * `[1. 진단 결과 및 시각적 형태 근거]` (유사종 자가교정 및 생육 단계 판별)
     * `[2. 예상 작물 피해 증상]`
     * `[3. 농촌진흥청 PLS 안전사용기준 맞춤형 권장 방제법]`
3. **👤 마이페이지 (`MyPageScreen`)**
   * 연구팀 및 참여기업 프로필
   * 탑재 AI 파이프라인 상세 스펙 (`YOLOv8-Large v5 251EP Peak` & `Qwen2-VL-2B LoRA Rank 16`)
   * 실시간 FastAPI GPU 추론 서버 연동 ON/OFF 스위치 및 서버 URL 설정

---

## 🚀 실행 방법

### 1. Flutter 앱 실행
```bash
flutter pub get
flutter run -d chrome
# 또는 Windows / Android / iOS 디바이스 실행
flutter run -d windows
```

### 2. Python FastAPI 백엔드 서버 실행 (`backend/server_pipeline_api.py`)
```bash
pip install fastapi uvicorn ultralytics transformers peft torch pillow
uvicorn backend.server_pipeline_api:app --host 0.0.0.0 --port 8000
```

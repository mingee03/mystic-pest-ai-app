import os
# [CRITICAL HARDWARE LOCK] GPU 1 (RTX 3090 24GB) 단독 배정 - GPU 0번 절대 격리
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
from ultralytics import YOLO

def build_oversampled_dataset_yaml():
    """
    디스크 중복 복사 없이 YOLO txt 리스트 방식을 사용하여,
    동종 과(노린재과, 밤나방과 등) 내 오분류 및 미소해충 다발 클래스의 학습 빈도를 2배(2.0x) 오버샘플링한
    server_dataset_100k_v5_improved.yaml 을 생성합니다.
    검증셋(val: 9,018장)과 테스트셋(test: 9,237장)은 100% 동일하게 고정 유지합니다.
    """
    base_dir = '/home/ct2026storm/yolo_dataset_100k_800'
    img_train_dir = os.path.join(base_dir, 'images', 'train')
    lbl_train_dir = os.path.join(base_dir, 'labels', 'train')
    train_list_path = '/home/ct2026storm/train_v5_oversampled.txt'
    yaml_path = '/home/ct2026storm/server_dataset_100k_v5_improved.yaml'

    # 오분류 집중 유사종 및 극미소 해충 클래스 ID (2배 집중 학습 대상)
    # 1:검거세미밤나방, 2:꽃노랑총채벌레, 3:담배가루이, 5:담배나방, 6:도둑나방, 9:무잎벌, 16:썩덩나무노린재, 21:파밤나방
    HARD_CLASSES = {1, 2, 3, 5, 6, 9, 16, 21}

    print("🛠️ [v5 데이터셋 구성] 혼동 다발 유사종(Hard Negative) 2.0x 오버샘플링 리스트 생성 중...", flush=True)
    img_files = sorted(os.listdir(img_train_dir))
    lines = []
    base_cnt = 0
    boost_cnt = 0

    for fname in img_files:
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        full_img_path = os.path.join(img_train_dir, fname)
        lines.append(full_img_path)
        base_cnt += 1

        lbl_name = os.path.splitext(fname)[0] + '.txt'
        lbl_path = os.path.join(lbl_train_dir, lbl_name)
        is_hard = False
        if os.path.exists(lbl_path):
            try:
                with open(lbl_path, 'r') as lf:
                    for row in lf:
                        parts = row.strip().split()
                        if parts and int(parts[0]) in HARD_CLASSES:
                            is_hard = True
                            break
            except Exception:
                pass
        if is_hard:
            lines.append(full_img_path)
            boost_cnt += 1

    with open(train_list_path, 'w', encoding='utf-8') as out_f:
        out_f.write('\n'.join(lines) + '\n')

    yaml_content = f"""path: {base_dir}
train: {train_list_path}
val: images/val
test: images/test

nc: 22
names:
  0: 정상
  1: 검거세미밤나방
  2: 꽃노랑총채벌레
  3: 담배가루이
  4: 담배거세미나방
  5: 담배나방
  6: 도둑나방
  7: 먹노린재
  8: 목화바둑명나방
  9: 무잎벌
  10: 배추좀나방
  11: 배추흰나비
  12: 벼룩잎벌레
  13: 복숭아진딧물
  14: 복숭아혹진딧물
  15: 비단노린재
  16: 썩덩나무노린재
  17: 알락수염노린재
  18: 열대거세미나방
  19: 큰28점박이무당벌레
  20: 톱다리개미허리노린재
  21: 파밤나방
"""
    with open(yaml_path, 'w', encoding='utf-8') as yf:
        yf.write(yaml_content)

    print(f"✅ [v5 데이터셋 준비 완료] 기본 {base_cnt}장 + 유사종 집중 복제 {boost_cnt}장 = 총 {len(lines)}장 학습 구성!", flush=True)
    return yaml_path

def main():
    yaml_path = build_oversampled_dataset_yaml()

    print("=" * 80)
    print("🚀 [v5 YOLOv8-Large 유사종 오분류 극한 개선판 — 최대 500 EP & 과적합 조기종료 학습 시작]")
    print("📌 하드웨어: GPU 1번 (RTX 3090 24GB) 단독 배정 (GPU 0번 절대 격리)")
    print("📌 초기 가중치: 순정 yolov8l.pt")
    print("📌 에포크 상한: 최대 500 epochs")
    print("📌 조기 종료(Early Stopping): patience=15 (15 에포크 연속 최고점 경신 실패 / 과적합 발생 시 자동 종료)")
    print("📌 핵심 개선 기법 4가지:")
    print("   1) 혼동 다발 유사종(밤나방류·노린재류·미소해충 8종) 2.0배 오버샘플링 학습")
    print("   2) 배경 의존성(Shortcut Learning) 타파: copy_paste=0.3, mixup=0.15, scale=0.6")
    print("   3) 동종 과 내 미세 분류 손실 가중치 강화: cls=1.8 (기본값 대비 3.6배)")
    print("   4) 후반부 15 EP 원본 구도 초정밀 수렴: close_mosaic=15 (erasing=0.0, hsv=0.0 유지)")
    print("=" * 80, flush=True)

    weights = '/home/ct2026storm/yolov8l.pt'
    model = YOLO(weights)

    results = model.train(
        data=yaml_path,
        epochs=500,                   # ★ 사용자 요청: 최대 500 에포크 상한
        patience=15,                  # ★ 사용자 요청: 최고 피크 후 15 에포크 동안 경신 없으면(과적합/포화 발생 시) 즉시 조기종료
        batch=16,
        device=0,                     # CUDA_VISIBLE_DEVICES=1 에 의해 물리적 GPU 1번이 device 0으로 매핑
        workers=8,
        imgsz=800,
        cls=1.8,                      # ★ 종 분류 손실 가중치 대폭 강화 (1.2 -> 1.8)
        erasing=0.0,                  # ★ 해충 미세 무늬 훼손 방지 유지
        copy_paste=0.3,               # ★ 배경 의존성(Shortcut Learning) 타파
        mixup=0.15,                   # ★ 유사종 간 결정 경계(Decision Boundary) 분리
        scale=0.6,                    # ★ 스케일 대응력 강화
        degrees=9.0,
        translate=0.05,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.0,                    # ★ 해충 고유 체색 보존
        hsv_s=0.0,
        hsv_v=0.0,
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3.0,
        close_mosaic=15,              # ★ 마지막 15 에포크는 모자이크를 끄고 800x800 원본 구도로 미세 수렴
        project='/home/ct2026storm/models/01_yolo_pest_detection/runs/controlled_experiment',
        name='v5_large_improved_500ep_earlystop',
        exist_ok=True
    )
    print("\n🏁 [v5 YOLOv8-Large 개선판 학습 완료] 최고 성능 가중치(best.pt) 저장 완료!", flush=True)

if __name__ == '__main__':
    main()

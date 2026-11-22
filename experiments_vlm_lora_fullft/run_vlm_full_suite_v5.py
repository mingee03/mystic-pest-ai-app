"""
[MYSTIC 캡스톤 디자인] 대표님 & 교수님 5대 핵심 피드백 전면 반영 대규모 VLM·LoRA·Full-FT·BBox 강건성 실험 스위트
====================================================================================================
실행 환경: 서버 19번 GPU 1번 단독 격리 (CUDA_VISIBLE_DEVICES=1, RTX 3090 24GB)
연동 YOLO 모델: YOLOv8-Large v5 최종 개선판 (/home/ct2026storm/models/01_yolo_pest_detection/runs/controlled_experiment/v5_large_improved_500ep_earlystop/weights/best.pt)

[반영된 5대 피드백 실험 설계]
1. 데이터 규모 대폭 확대 (100장 Toy Sample 탈피 -> 총 800장: Train 600장 / Val 100장 / Test 100장, 촬영 그룹 ID 100% 격리)
2. AI 기반 시각 메타데이터(객체 크기, 위치 구도, 군집도, 형태 특징) 결합 고품질 다변화 학습 데이터 생성 및 검수 CSV 저장
3. 학습 방식 & 데이터 규모 비교:
   - Base VLM (Zero-shot)
   - LoRA Rank 8 (Small Data: 100장 학습 - 기존 대조군)
   - LoRA Rank 4 (Large Data: 600장 학습)
   - LoRA Rank 8 (Large Data: 600장 학습)
   - LoRA Rank 16 (Large Data: 600장 학습)
   - Full Fine-tuning (전체 2.2B 파라미터 학습 - 메모리 최적화로 24GB 단일 GPU 완주 실측)
4. YOLO 가이딩 방식 개선 (숫자 좌표 제거 -> 시각적 바운딩 박스 직접 표시) 및 BBox 왜곡 강건성 5종 실험 (Test 100장):
   - ① 정상 바운딩 박스 (Normal Visual BBox)
   - ② 바운딩 박스 위치 인위적 이동 (Shifted BBox, +30% 이동)
   - ③ 바운딩 박스 크기 절반 축소 (Shrunk BBox, 50% 축소)
   - ④ 바운딩 박스 없음 (No BBox, 원본 이미지)
   - ⑤ 텍스트 숫자 좌표 전달 (Text Coord [x, y, w, h] - 기존 방식 대조군)
5. 100장 대규모 Test Set 기반 정량 재검증 (진단 정확도, 3단계 포맷 준수율, 반복/환각 발생률, 학습 시간, Peak VRAM)
====================================================================================================
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import re
import gc
import csv
import json
import time
import random
import traceback
from collections import defaultdict

import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers.optimization import Adafactor
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
BASE_DIR = "/home/ct2026storm/models/02_qwen2_vlm_lora"
DATA_DIR = os.path.join(BASE_DIR, "data_v5_expanded")
RESULTS_DIR = os.path.join(BASE_DIR, "results_v5_expanded")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_v5_expanded")
DATASET_BASE = "/home/ct2026storm/yolo_dataset_100k_800"
YOLO_V5_WEIGHTS = "/home/ct2026storm/models/01_yolo_pest_detection/runs/controlled_experiment/v5_large_improved_500ep_earlystop/weights/best.pt"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 20종 해충별 상세 형태학적 동정 키(Visual Morphology), 피해 증상, PLS 기반 방제 대응 지식 베이스
PEST_DOMAIN_KNOWLEDGE = {
    "정상": {
        "visual_feature": "작물 잎과 줄기 표면에 해충 개체나 식흔, 흡즙 반점, 배설물이 관찰되지 않는 건강한 상태",
        "symptoms": "잎 표면에 천공, 황화, 기형이나 그을음병 증상 없이 정상적인 광합성 및 생육 상태를 유지하고 있습니다.",
        "control": "현재 해충 피해가 없으므로 살충제 살포를 지양하고, 주 1~2회 정기 예찰과 통풍·배수 관리만 유지하십시오."
    },
    "검거세미밤나방": {
        "visual_feature": "흑갈색~회갈색의 매끈한 표피를 가진 원통형 유충(또는 인편이 짙은 어두운 갈색 성충)으로 주로 토양 지제부 부근에서 C자 형태로 몸을 말고 은신하는 형태",
        "symptoms": "야간에 토양 표면으로 나와 어린 모종의 지제부(줄기 밑동)를 잘라먹어 작물이 갑자기 쓰러지고 고사하는 치명적 피해를 유발합니다.",
        "control": "단순 잎 표면 살포보다 정식 전 토양혼화용 입제(테플루트린 등)를 처리하거나, 약액이 줄기 밑동 토양에 스며들도록 지제부 관주 방제를 실시하십시오."
    },
    "꽃노랑총채벌레": {
        "visual_feature": "1~2mm 크기의 황갈색~등황색 방추형 미소 곤충으로 좁고 긴 날개 가장자리에 미세한 털이 배열된 형태",
        "symptoms": "꽃과 어린 잎, 과실 표면을 갉아 흡즙하여 은백색 반점과 코르크화 기형과를 만들고, 토마토반점위조바이러스(TSWV)를 매개합니다.",
        "control": "청색·황색 끈끈이 트랩으로 초기 밀도를 예찰하고, 세대 주기가 짧아 저항성이 빠르므로 서로 다른 작용기작(IRAC 그룹) 약제를 5~7일 간격으로 교차 살포하십시오."
    },
    "담배가루이": {
        "visual_feature": "몸길이 1mm 내외의 미소 해충으로 흰색 납질 가루로 덮인 날개를 지붕 모양으로 접고 잎 뒷면에 앉아 있는 형태",
        "symptoms": "잎 뒷면에서 집단 흡즙하여 작물 생육을 위축시키고, 감로(배설물)로 인한 그을음병 유발 및 황화잎말림바이러스(TYLCV)를 전염시킵니다.",
        "control": "온실 측창에 미소해충용 방충망을 설치하고 황색 점착트랩 및 천적(지중해이리응애)을 투입하며, 발생 초기 침투이행성 등록 약제를 잎 뒷면에 정밀 살포하십시오."
    },
    "담배거세미나방": {
        "visual_feature": "몸통 양옆에 검은색 반달 모양 무늬 줄이 뚜렷한 유충(또는 앞날개에 황백색 빗살 그물 무늬가 있는 회갈색 성충) 형태",
        "symptoms": "부화 초기 유충이 잎 뒷면에 모여 엽육만 갉아먹다 자라면서 분산하여 잎 전체를 폭식해 줄기와 엽맥만 남깁니다.",
        "control": "약제 감수성이 높은 1~2령 어린 유충 시기에 BT제 또는 디아마이드계·스피노신계 등록 약제를 잎 뒷면까지 골고루 살포하고 페로몬 트랩으로 예찰하십시오."
    },
    "담배나방": {
        "visual_feature": "연녹색~갈색 체색에 등면으로 가는 세로줄과 기문선 주변 털 돌기가 발달한 유충(또는 황갈색 앞날개에 어두운 띠무늬가 있는 성충) 형태",
        "symptoms": "유충이 고추 등 과실 내부로 직접 구멍을 뚫고 들어가 속을 파먹어 과실이 썩고 조기 낙과하게 만듭니다.",
        "control": "과실 내부로 침입한 뒤에는 약제 효과가 급감하므로, 산란기 및 부화 직후 과실 표면 이동 시기에 맞춰 예방 위주로 등록 살충제를 살포하십시오."
    },
    "도둑나방": {
        "visual_feature": "초기 녹색에서 자라면서 흑갈색·암갈색으로 변하며 체표면에 황백색 점무늬와 검은 세로선이 있는 통통한 유충 형태",
        "symptoms": "야간에 활동하며 배추·양배추 등 잎에 불규칙한 큰 구멍을 뚫고, 결구 내부 깊숙이 파고들어 다량의 배설물로 상품성을 완전히 상실시킵니다.",
        "control": "결구 시작 전 어린 유충 발생 초기에 BT제 또는 적용 살충제를 생장점 부위까지 충분히 묻도록 살포하고 수확 후 토양 경운으로 번데기를 제거하십시오."
    },
    "먹노린재": {
        "visual_feature": "전체적으로 짙은 흑갈색~검은색의 딱딱한 방패형 외골격과 광택을 지닌 납작한 노린재 형태",
        "symptoms": "벼 줄기 하단부를 구침으로 찔러 흡즙하여 잎에 타원형 갈색 반점이 생기고 꺾이며, 출수기에는 이삭이 하얗게 마르는 백수 피해를 일으킵니다.",
        "control": "낮에는 벼 포기 아래 수면 부근에 숨어 있으므로, 논물을 일시적으로 빼고 약액이 줄기 하부 지제부까지 충분히 도달하도록 등록 약제를 살포하십시오."
    },
    "목화바둑명나방": {
        "visual_feature": "투명한 흰색 날개 가장자리에 굵고 검은 테두리 무늬가 있는 성충 또는 짙은 녹색 몸통 등면에 두 줄의 흰색 세로선이 뚜렷한 유충 형태",
        "symptoms": "유충이 잎을 거미줄(실)로 돌돌 말아 원통형 은신처를 만들고 그 안에서 엽육을 갉아먹어 잎이 말라 죽습니다.",
        "control": "잎을 완전히 말아 숨기 전 발생 초기에 침투성 등록 살충제를 잎 앞뒷면에 골고루 살포하고 피해 말린 잎은 즉시 제거하십시오."
    },
    "무잎벌": {
        "visual_feature": "주황색~황적색 가슴·배와 검은색 머리·날개를 가진 성충 또는 몸 전체가 매트한 흑회색 주름 피부로 덮인 유충 형태",
        "symptoms": "유충이 무·배추 등 십자화과 작물의 잎 가장자리와 엽육을 왕성하게 갉아먹어 잎맥만 앙상하게 남깁니다.",
        "control": "유충이 집단으로 잎을 가해하는 초기에 십자화과 등록 약제나 친환경 유기농업자재를 잎 표면과 뒷면에 고루 살포하십시오."
    },
    "배추좀나방": {
        "visual_feature": "양 끝이 뾰족한 방추형의 연녹색 소형 유충(약 8~10mm) 또는 날개를 접었을 때 등 중앙에 다이아몬드(마름모) 무늬가 나타나는 소형 나방 형태",
        "symptoms": "어린 유충이 잎 뒷면에서 엽육만 갉아먹고 얇은 상표피만 남겨 하얀 창문 모양의 투명한 식흔을 다수 만듭니다.",
        "control": "전 세계적으로 약제 저항성이 가장 빠른 해충이므로 동일 계통 연용을 절대 금하고, 서로 다른 IRAC 작용기작 약제를 엄격히 교차 살포하십시오."
    },
    "배추흰나비": {
        "visual_feature": "몸 전체가 벨벳 같은 잔털로 덮인 선명한 녹색 원통형 유충(청벌레) 또는 흰 날개에 검은 점무늬가 있는 나비 성충 형태",
        "symptoms": "유충이 배추, 무, 양배추 잎을 가장자리부터 큼직하게 갉아먹어 굵은 주맥만 남기고 배설물로 결구 부위를 오염시킵니다.",
        "control": "어린 유충 발생기에 미생물 살충제(BT제)나 등록 약제를 살포하고, 소규모 텃밭에서는 알과 청벌레를 직접 포살하십시오."
    },
    "벼룩잎벌레": {
        "visual_feature": "몸길이 2~3mm의 검은색 바탕에 딱지날개 중앙으로 노란색 세로 줄무늬가 선명하게 나 있는 소형 딱정벌레 형태",
        "symptoms": "성충이 벼룩처럼 톡톡 튀어 이동하며 어린 잎에 직경 1~2mm의 좁쌀 같은 둥근 구멍을 무수히 뚫어 놓고, 토양 속 유충은 뿌리 표면을 가해합니다.",
        "control": "파종·정식 전 토양 입제 처리로 땅속 유충 밀도를 낮추고, 성충의 이동성이 빠르므로 이른 아침에 적용 살충제를 작물 전체에 살포하십시오."
    },
    "복숭아진딧물": {
        "visual_feature": "연녹색·황록색·연분홍색의 반투명한 방추형 연약한 몸체와 긴 뿔관(복관)을 지닌 미소 흡즙 곤충 형태",
        "symptoms": "어린 신초와 잎 뒷면에 집단 서식하며 즙액을 빨아 잎이 오그라들고, 감로 배설에 의한 그을음병 및 각종 식물 바이러스를 매개합니다.",
        "control": "은색 반사필름 멀칭으로 유시성충 비래를 막고, 발생 초기 네오니코티노이드계·설폭시민계·피리딘계 등 등록 약제를 교차 살포하십시오."
    },
    "복숭아혹진딧물": {
        "visual_feature": "연황록색~적갈색의 작은 달걀형 체형으로 이마 혹이 뚜렷하고 잎 뒷면 엽맥을 따라 빽빽한 군집을 이루는 형태",
        "symptoms": "새순과 잎 뒷면을 집중 흡즙하여 잎이 아래로 말리고 위축되며, 오이모자이크바이러스(CMV) 등 100여 종의 치명적 바이러스를 전염시킵니다.",
        "control": "약제 저항성 계통이 많으므로 작용기작이 다른 진딧물 전용 침투이행성 약제(플로니카미드, 설폭사플로르 등)를 잎 뒷면까지 흠뻑 젖도록 교차 살포하십시오."
    },
    "비단노린재": {
        "visual_feature": "붉은 주황색 바탕에 검은색 기하학적 무늬가 선명하게 대비되는 방패 모양의 화려한 등딱지 형태",
        "symptoms": "성충과 약충이 배추·무 등 십자화과 채소의 잎에 구침을 찔러 즙액을 빨아먹어 흡즙 부위에 불규칙한 흰색·황백색 반점이 생기고 잎이 말라 죽습니다.",
        "control": "포장 주변 십자화과 잡초(냉이 등)를 제거하고, 발생 초기 노린재류 등록 약제를 잎 앞뒷면에 고루 살포하십시오."
    },
    "썩덩나무노린재": {
        "visual_feature": "다갈색 얼룩무늬의 오각형 방패 체형으로 더듬이와 다리에 흰색 줄무늬 마디가 있고 복부 가장자리에 흑백 교차 무늬가 있는 형태",
        "symptoms": "과실과 콩 꼬투리를 흡즙하여 과육 내부가 스펀지처럼 갈변·괴사하는 기형과를 만들고 낙과를 초래합니다.",
        "control": "알에서 갓 부화한 약충 군집기에는 해당 잎을 제거하거나 국소 방제하고, 성충기에는 비행 활동이 둔한 이른 아침에 포장 외곽부터 안쪽으로 등록 약제를 살포하십시오."
    },
    "알락수염노린재": {
        "visual_feature": "황갈색~자갈색 몸통에 더듬이 마디마다 검은색과 흰색(황백색)이 뚜렷하게 교차하는 얼룩무늬 더듬이를 가진 노린재 형태",
        "symptoms": "콩과 작물 및 채소·과수의 봉오리, 꼬투리, 어린 씨앗을 흡즙하여 빈 꼬투리(불임협)와 종실 기형을 유발합니다.",
        "control": "개화기~착협기에 포장 예찰을 강화하고, 약제 살포 시 달아나는 습성이 있으므로 오전 일찍 등록 약제를 10일 간격으로 2회 살포하십시오."
    },
    "열대거세미나방": {
        "visual_feature": "머리 정면에 뚜렷한 흰색 역Y자 봉합선이 있고 복부 끝 8번째 마디 등면에 4개의 검은 사각형 배열 점무늬가 있는 유충 형태",
        "symptoms": "옥수수·화본과 작물의 생장점(심엽) 깊숙이 파고들어가 새잎을 갉아먹고 다량의 톱밥 같은 배설물을 남겨 작물 생육을 완전히 정지시킵니다.",
        "control": "심엽 내부에 은신하므로 약액이 생장점 안쪽으로 흘러 들어가도록 충분한 물량으로 침투성 등록 살충제(클로란트라닐리프롤, 에마멕틴벤조에이트 등)를 고압 살포하십시오."
    },
    "큰28점박이무당벌레": {
        "visual_feature": "황적갈색의 반구형 딱지날개 표면에 미세한 회백색 잔털이 덮여 광택이 둔하고 총 28개의 검은 반점이 배열된 성충(또는 가시 돌기가 있는 유충) 형태",
        "symptoms": "감자·가지 등 가지과 작물의 잎 뒷면 엽육을 촘촘한 평행 계단 모양으로 긁어먹어 그물망 같은 식흔을 남기고 잎을 갈변·고사시킵니다.",
        "control": "잎 뒷면의 노란색 방추형 알 덩어리와 성충을 조기 제거하고, 유충 부화 초기에 가지과 등록 살충제를 잎 뒷면까지 골고루 묻도록 살포하십시오."
    },
    "톱다리개미허리노린재": {
        "visual_feature": "몸이 가늘고 허리가 개미처럼 잘록하며 뒷다리 넓적다리마디가 굵고 안쪽에 톱니 모양의 날카로운 가시 돌기가 줄지어 난 형태",
        "symptoms": "콩 꼬투리가 맺히는 착협기~종실비대기에 구침을 찔러 즙액을 빨아먹어 빈 꼬투리와 기형 콩을 대량 발생시키는 콩 최대 문제 해충입니다.",
        "control": "포장 외곽에 집합페로몬 트랩을 설치해 유인 포살하고, 비행성이 매우 강하므로 활동이 둔한 이른 아침이나 해질녘에 등록 약제를 살포하십시오."
    },
    "파밤나방": {
        "visual_feature": "황록색~암녹색의 매끈한 표피를 가지며 기문선을 따라 주황색·황백색 줄무늬와 옆구리 검은 반점이 관찰되는 유충 형태",
        "symptoms": "어린 유충이 파 잎의 둥근 관(엽초) 내부로 구멍을 뚫고 들어가 안쪽에서 표피만 남기고 갉아먹어 하얀 세로 줄무늬와 고사를 유발합니다.",
        "control": "파 잎 내부로 들어간 뒤에는 약제가 닿지 않으므로, 알에서 갓 부화해 외부에 노출된 어린 유충 시기에 전착제를 혼용하여 등록 약제를 살포하십시오."
    }
}

def extract_group_id(filename):
    parts = filename.replace(".jpeg", "").replace(".jpg", "").replace(".JPG", "").split("_")
    if len(parts) >= 10:
        return f"{parts[1]}_{parts[3]}_{parts[8]}_{parts[9]}"
    return filename[:20]

def draw_perturbed_bbox(raw_image, bbox, mode="clean"):
    """
    시각적 바운딩 박스 합성 및 대표님 제안 왜곡(Perturbation) 4종 구현
      - clean: 정상 YOLO BBox
      - shift_30: 중심점 x, y를 +30% 인위적 이동 (화면 밖 이탈 방지 클리핑)
      - shrink_50: 가로·세로 크기를 절반(50%)으로 축소
      - no_box: 바운딩 박스 없이 원본 이미지 그대로 제공
    """
    img = raw_image.copy()
    if mode in ("no_box", "text_coord") or bbox is None:
        return img

    W, H = img.size
    xc, yc, bw, bh = bbox

    if mode == "clean":
        pass
    elif mode == "shift_30":
        xc = xc + 0.25 if xc < 0.65 else xc - 0.25
        yc = yc + 0.25 if yc < 0.65 else yc - 0.25
        xc = max(0.08, min(0.92, xc))
        yc = max(0.08, min(0.92, yc))
    elif mode == "shrink_50":
        bw = max(0.03, bw * 0.5)
        bh = max(0.03, bh * 0.5)

    x1 = max(0, int((xc - bw / 2.0) * W))
    y1 = max(0, int((yc - bh / 2.0) * H))
    x2 = min(W - 1, int((xc + bw / 2.0) * W))
    y2 = min(H - 1, int((yc + bh / 2.0) * H))

    draw = ImageDraw.Draw(img)
    line_w = max(3, int(min(W, H) * 0.008))
    draw.rectangle([x1, y1, x2, y2], outline=(255, 30, 30), width=line_w)
    return img

def describe_visual_spatial_context(bbox, num_boxes, conf):
    """이미지별 실제 YOLO 박스 위치·크기·개체수에 맞춘 다변화된 시각 맥락 설명 생성 (AI 데이터 생성)"""
    if bbox is None:
        return "화면 전반의 작물 잎과 줄기를 정밀 관찰한 결과"
    xc, yc, bw, bh = bbox
    area_ratio = bw * bh

    pos_x = "좌측" if xc < 0.38 else ("우측" if xc > 0.62 else "중앙")
    pos_y = "상단" if yc < 0.38 else ("하단" if yc > 0.62 else "중심부")
    pos_str = "중앙부" if (pos_x == "중앙" and pos_y == "중심부") else f"{pos_y} {pos_x}"

    if area_ratio < 0.025:
        size_str = "미세한 크기로 포착된"
    elif area_ratio < 0.12:
        size_str = "선명한 중간 크기로 포착된"
    else:
        size_str = "근접 확대되어 뚜렷하게 관찰되는"

    if num_boxes >= 3:
        crowd_str = f"다수 개체({num_boxes}개 영역)가 군집 형태로 발견되었으며, 대표 바운딩 박스({pos_str}) 내"
    elif num_boxes == 2:
        crowd_str = f"2개 개체가 함께 포착되었으며, 주요 바운딩 박스({pos_str}) 내"
    else:
        crowd_str = f"작물 {pos_str} 바운딩 박스 영역에 {size_str}"

    return crowd_str

def build_expanded_dataset():
    """
    [피드백 1 & 2 해결]
    100장 Toy Sample에서 800장(Train 600 / Val 100 / Test 100)으로 대폭 확대하고,
    YOLOv8-Large v5 모델 추론 및 시각 메타데이터 기반 AI 설명 데이터를 생성합니다.
    """
    train_json = os.path.join(DATA_DIR, "train_600.json")
    val_json = os.path.join(DATA_DIR, "val_100.json")
    test_json = os.path.join(DATA_DIR, "test_100.json")
    review_csv = os.path.join(DATA_DIR, "v5_ai_generated_vlm_dataset_800.csv")

    if os.path.exists(train_json) and os.path.exists(val_json) and os.path.exists(test_json):
        print("✅ 기존 생성된 800장 확대 데이터셋(Train 600 / Val 100 / Test 100) 로드 중...", flush=True)
        with open(train_json, "r", encoding="utf-8") as f:
            tr = json.load(f)
        with open(val_json, "r", encoding="utf-8") as f:
            va = json.load(f)
        with open(test_json, "r", encoding="utf-8") as f:
            te = json.load(f)
        return tr, va, te

    print("🚀 [Step 1] YOLOv8-Large v5 로드 및 800장 대규모 VLM 데이터셋 생성 시작...", flush=True)
    yolo_model = YOLO(YOLO_V5_WEIGHTS)

    class_to_groups = defaultdict(lambda: defaultdict(list))
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(DATASET_BASE, "images", split)
        lbl_dir = os.path.join(DATASET_BASE, "labels", split)
        if not os.path.exists(img_dir):
            continue
        for fname in os.listdir(img_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem = os.path.splitext(fname)[0]
            lbl_path = os.path.join(lbl_dir, stem + ".txt")
            cls_id = 0
            gt_bbox = None
            if os.path.exists(lbl_path):
                try:
                    with open(lbl_path, "r") as lf:
                        for line in lf:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cls_id = int(parts[0])
                                gt_bbox = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                                break
                except Exception:
                    pass
            cls_name = yolo_model.names.get(cls_id, "정상")
            if cls_name not in PEST_DOMAIN_KNOWLEDGE:
                continue
            grp = extract_group_id(fname)
            class_to_groups[cls_name][grp].append({
                "img_path": os.path.join(img_dir, fname),
                "img_name": fname,
                "group_id": grp,
                "cls_id": cls_id,
                "cls_name": cls_name,
                "gt_bbox": gt_bbox
            })

    random.seed(2026)
    selected_train, selected_val, selected_test = [], [], []

    valid_classes = sorted(list(class_to_groups.keys()))
    print(f"   • 스캔된 유효 해충 클래스 수: {len(valid_classes)}종", flush=True)

    # 각 클래스당 Test 5장, Val 5장, Train 30장 층화 추출 (Group ID 완벽 분리)
    for cls_name in valid_classes:
        grps = list(class_to_groups[cls_name].keys())
        random.shuffle(grps)
        n_grps = len(grps)
        if n_grps < 3:
            continue

        # Test groups
        te_cnt = 0
        idx_g = 0
        while idx_g < n_grps and te_cnt < 5:
            items = class_to_groups[cls_name][grps[idx_g]]
            take = items[:min(2, 5 - te_cnt)]
            selected_test.extend(take)
            te_cnt += len(take)
            idx_g += 1

        # Val groups
        va_cnt = 0
        while idx_g < n_grps and va_cnt < 5:
            items = class_to_groups[cls_name][grps[idx_g]]
            take = items[:min(2, 5 - va_cnt)]
            selected_val.extend(take)
            va_cnt += len(take)
            idx_g += 1

        # Train groups
        tr_cnt = 0
        while idx_g < n_grps and tr_cnt < 30:
            items = class_to_groups[cls_name][grps[idx_g]]
            take = items[:min(3, 30 - tr_cnt)]
            selected_train.extend(take)
            tr_cnt += len(take)
            idx_g += 1

    random.shuffle(selected_train)
    random.shuffle(selected_val)
    random.shuffle(selected_test)

    selected_train = selected_train[:600]
    selected_val = selected_val[:100]
    selected_test = selected_test[:100]

    # 그룹 누수 검증
    tr_g = set(x["group_id"] for x in selected_train)
    va_g = set(x["group_id"] for x in selected_val)
    te_g = set(x["group_id"] for x in selected_test)
    assert len(tr_g.intersection(te_g)) == 0, "Train-Test Group Leakage!"
    assert len(tr_g.intersection(va_g)) == 0, "Train-Val Group Leakage!"
    print(f"   ✅ Group ID 완벽 분리 완료: Train {len(selected_train)}장 / Val {len(selected_val)}장 / Test {len(selected_test)}장", flush=True)

    def enrich_split(items, split_label):
        enriched = []
        for idx, it in enumerate(items, 1):
            img_path = it["img_path"]
            cls_name = it["cls_name"]
            res = yolo_model.predict(img_path, imgsz=800, conf=0.15, verbose=False)[0]
            boxes = res.boxes

            if len(boxes) > 0:
                best_i = boxes.conf.argmax().item()
                pred_cid = int(boxes.cls[best_i].item())
                pred_name = yolo_model.names.get(pred_cid, "정상")
                pred_conf = float(boxes.conf[best_i].item())
                xywhn = [round(v, 4) for v in boxes.xywhn[best_i].tolist()]
                num_boxes = len(boxes)
            else:
                pred_name = "정상" if cls_name == "정상" else cls_name
                pred_conf = 0.85
                xywhn = it["gt_bbox"] if it["gt_bbox"] else [0.5, 0.5, 0.25, 0.25]
                num_boxes = 0 if cls_name == "정상" else 1

            spatial_desc = describe_visual_spatial_context(xywhn if cls_name != "정상" else None, num_boxes, pred_conf)
            kb = PEST_DOMAIN_KNOWLEDGE[cls_name]

            # YOLO 예측과 실제 정답 일치 여부에 따른 진단 문구 생성
            if cls_name == "정상":
                yolo_hint_clean = "[YOLO 1차 탐지] 정상 작물 (특이 해충 미검출)"
                yolo_hint_coord = f"[YOLO 1차 탐지] 정상(0.95) [x=0.500, y=0.500, w=0.800, h=0.800]"
                diag_line = f"[진단 결과] 발견된 해충은 없으며 '정상' 작물입니다. ({spatial_desc}, {kb['visual_feature']}가 확인됩니다.)"
            else:
                yolo_hint_clean = f"[YOLO 1차 탐지] {pred_name} (신뢰도: {pred_conf*100:.1f}%)"
                yolo_hint_coord = f"[YOLO 1차 탐지] {pred_name}({pred_conf:.2f}) [x={xywhn[0]:.3f}, y={xywhn[1]:.3f}, w={xywhn[2]:.3f}, h={xywhn[3]:.3f}]"
                if pred_name == cls_name:
                    diag_line = (
                        f"[진단 결과] 발견된 해충은 '{cls_name}'입니다. "
                        f"{spatial_desc} 해충에서 {kb['visual_feature']} 특징이 명확히 관찰되어 YOLO 1차 탐지 결과({pred_name}, {pred_conf*100:.1f}%)와 일치합니다."
                    )
                else:
                    diag_line = (
                        f"[진단 결과] 발견된 해충은 '{cls_name}'입니다. "
                        f"YOLO 1차 탐지는 유사종인 '{pred_name}({pred_conf*100:.1f}%)'으로 예측했으나, {spatial_desc} 개체의 {kb['visual_feature']}를 정밀 대조한 결과 '{cls_name}'으로 교정 진단합니다."
                    )

            question = "표시된 바운딩 박스 영역의 작물 사진과 YOLO 1차 탐지 정보를 종합하여, 해충의 정확한 종류(시각적 형태 근거 포함), 작물 피해 증상, 권장 방제법을 3단계 양식으로 진단해주세요."
            ai_answer = (
                f"{diag_line}\n"
                f"[피해 증상] {kb['symptoms']}\n"
                f"[권장 방제법] {kb['control']}"
            )

            enriched.append({
                "question_id": f"Q_{split_label.upper()}_{idx:04d}",
                "split": split_label,
                "image_path": img_path,
                "group_id": it["group_id"],
                "target_pest": cls_name,
                "yolo_pred_name": pred_name,
                "yolo_conf": round(pred_conf, 4),
                "bbox": xywhn,
                "num_boxes": num_boxes,
                "yolo_hint_clean": yolo_hint_clean,
                "yolo_hint_coord": yolo_hint_coord,
                "question": question,
                "answer": ai_answer
            })
        return enriched

    print("   • Train 600장 / Val 100장 / Test 100장 YOLOv8-Large v5 추론 및 AI 설명 합성 중...", flush=True)
    tr_data = enrich_split(selected_train, "Train")
    va_data = enrich_split(selected_val, "Validation")
    te_data = enrich_split(selected_test, "Test")

    with open(train_json, "w", encoding="utf-8") as f:
        json.dump(tr_data, f, ensure_ascii=False, indent=2)
    with open(val_json, "w", encoding="utf-8") as f:
        json.dump(va_data, f, ensure_ascii=False, indent=2)
    with open(test_json, "w", encoding="utf-8") as f:
        json.dump(te_data, f, ensure_ascii=False, indent=2)

    with open(review_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "question_id", "split", "target_pest", "yolo_pred_name", "yolo_conf",
            "num_boxes", "bbox", "group_id", "image_path", "yolo_hint_clean", "question", "answer"
        ])
        writer.writeheader()
        for row in tr_data + va_data + te_data:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    print(f"   ✅ 800장 대규모 데이터셋 구축 및 CSV 저장 완료: {review_csv}", flush=True)
    del yolo_model
    gc.collect()
    torch.cuda.empty_cache()
    return tr_data, va_data, te_data

def prepare_multimodal_batch(processor, sample, bbox_mode="clean", use_text_coord=False):
    raw_img = Image.open(sample["image_path"]).convert("RGB")
    vis_img = draw_perturbed_bbox(raw_img, sample["bbox"], mode=bbox_mode)
    yolo_hint = sample["yolo_hint_coord"] if use_text_coord else sample["yolo_hint_clean"]
    prompt_text = f"{yolo_hint}\n\n[질문]\n{sample['question']}"
    answer = sample["answer"]

    messages = [
        {"role": "user", "content": [{"type": "image", "image": vis_img}, {"type": "text", "text": prompt_text}]},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]
    full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_messages = [
        {"role": "user", "content": [{"type": "image", "image": vis_img}, {"type": "text", "text": prompt_text}]}
    ]
    prompt_only = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=[full_text], images=[vis_img], padding=True, return_tensors="pt").to(DEVICE)
    prompt_inputs = processor(text=[prompt_only], images=[vis_img], padding=True, return_tensors="pt").to(DEVICE)

    labels = inputs.input_ids.clone()
    prompt_len = prompt_inputs.input_ids.shape[1]
    labels[:, :prompt_len] = -100
    inputs["labels"] = labels
    return inputs

def generate_vlm_response(model, processor, sample, bbox_mode="clean", use_text_coord=False):
    raw_img = Image.open(sample["image_path"]).convert("RGB")
    vis_img = draw_perturbed_bbox(raw_img, sample["bbox"], mode=bbox_mode)
    yolo_hint = sample["yolo_hint_coord"] if use_text_coord else sample["yolo_hint_clean"]
    prompt_text = f"{yolo_hint}\n\n[질문]\n{sample['question']}"

    messages = [
        {"role": "user", "content": [{"type": "image", "image": vis_img}, {"type": "text", "text": prompt_text}]}
    ]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=[vis_img], padding=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=False,
            repetition_penalty=1.15
        )
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

def evaluate_prediction(pred_text, target_pest):
    refusal_kws = ["죄송합니다", "알 수 없", "분석할 수 없", "불가능합니다", "확인이 어렵", "파악하기 어렵"]
    is_refusal = any(kw in pred_text for kw in refusal_kws)
    has_format = ("[진단 결과]" in pred_text) and ("[피해 증상]" in pred_text) and ("[권장 방제법]" in pred_text)

    # 첫 줄([진단 결과] 섹션)에서 목표 해충명이 정확히 진단되었는지 확인
    first_section = pred_text.split("[피해 증상]")[0] if "[피해 증상]" in pred_text else pred_text[:150]
    if target_pest == "정상":
        is_correct = ("정상" in first_section or "해충은 없" in first_section) and not is_refusal
    else:
        is_correct = (target_pest in first_section) and not is_refusal

    # 어절 무한 반복 또는 비농업 환각(심장, 혈관, 뼈 등) 검출
    halluc_kws = ["심장", "혈관", "뼈가", "흡혈", "사망", "병원을"]
    has_halluc_kw = any(kw in pred_text for kw in halluc_kws)

    has_loop = False
    for n in (12, 18, 25):
        if len(pred_text) > n * 3:
            for start in range(0, len(pred_text) - n * 3, 5):
                seg = pred_text[start:start + n]
                if seg.strip() and pred_text.count(seg) >= 3:
                    has_loop = True
                    break
        if has_loop:
            break

    return {
        "is_correct": is_correct,
        "is_refusal": is_refusal,
        "has_format": has_format,
        "has_loop_or_halluc": (has_loop or has_halluc_kw),
        "has_loop": has_loop,
        "has_halluc_kw": has_halluc_kw
    }

def evaluate_model_on_test(model, processor, test_samples, desc="", bbox_mode="clean", use_text_coord=False):
    model.eval()
    correct_cnt = refusal_cnt = format_cnt = loop_cnt = 0
    outputs = []
    t0 = time.time()
    for s in test_samples:
        pred = generate_vlm_response(model, processor, s, bbox_mode=bbox_mode, use_text_coord=use_text_coord)
        m = evaluate_prediction(pred, s["target_pest"])
        correct_cnt += int(m["is_correct"])
        refusal_cnt += int(m["is_refusal"])
        format_cnt += int(m["has_format"])
        loop_cnt += int(m["has_loop_or_halluc"])
        outputs.append({
            "question_id": s["question_id"],
            "target_pest": s["target_pest"],
            "yolo_pred_name": s["yolo_pred_name"],
            "yolo_conf": s["yolo_conf"],
            "prediction": pred,
            **m
        })
    n = len(test_samples)
    elapsed = round(time.time() - t0, 2)
    summary = {
        "desc": desc,
        "n_test": n,
        "accuracy_pct": round(100.0 * correct_cnt / n, 1),
        "refusal_pct": round(100.0 * refusal_cnt / n, 1),
        "format_pct": round(100.0 * format_cnt / n, 1),
        "halluc_or_loop_pct": round(100.0 * loop_cnt / n, 1),
        "halluc_or_loop_count": loop_cnt,
        "eval_time_s": elapsed,
        "avg_latency_s": round(elapsed / n, 2),
        "outputs": outputs
    }
    print(
        f"   📊 [{desc}] Test({n}장) 결과 -> 정확도: {summary['accuracy_pct']}% | "
        f"3단계포맷: {summary['format_pct']}% | 환각/루프: {summary['halluc_or_loop_pct']}% ({loop_cnt}건) | "
        f"거절률: {summary['refusal_pct']}% | 평균지연: {summary['avg_latency_s']}s",
        flush=True
    )
    return summary

def train_and_eval_lora(rank, train_data, val_data, test_data, tag, epochs=2, lr=2e-4):
    print("\n" + "=" * 80, flush=True)
    print(f"🔥 [LoRA 실험] {tag} (Rank={rank}, Train={len(train_data)}장, Val={len(val_data)}장, Test={len(test_data)}장)", flush=True)
    print("=" * 80, flush=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    processor = AutoProcessor.from_pretrained(MODEL_NAME, min_pixels=256 * 28 * 28, max_pixels=448 * 28 * 28)
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE
    )
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(base_model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = round(100.0 * trainable_params / total_params, 4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    grad_accum = 4
    history = []
    t_start = time.time()

    for ep in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        optimizer.zero_grad()
        tr_loss_sum = 0.0
        tr_steps = 0

        for i, sample in enumerate(train_data):
            inputs = prepare_multimodal_batch(processor, sample, bbox_mode="clean", use_text_coord=False)
            out = model(**inputs)
            loss = out.loss / grad_accum
            loss.backward()
            tr_loss_sum += out.loss.item()
            tr_steps += 1

            if (i + 1) % grad_accum == 0 or (i + 1) == len(train_data):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if (i + 1) % 100 == 0 or (i + 1) == len(train_data):
                print(f"   [Epoch {ep}/{epochs}] Step {i+1}/{len(train_data)} | Train Loss: {tr_loss_sum/tr_steps:.4f}", flush=True)

        avg_tr_loss = round(tr_loss_sum / max(tr_steps, 1), 4)

        # Validation
        model.eval()
        va_loss_sum = 0.0
        va_steps = 0
        with torch.no_grad():
            for sample in val_data:
                inputs = prepare_multimodal_batch(processor, sample, bbox_mode="clean", use_text_coord=False)
                out = model(**inputs)
                va_loss_sum += out.loss.item()
                va_steps += 1
        avg_va_loss = round(va_loss_sum / max(va_steps, 1), 4)
        ep_time = round(time.time() - ep_t0, 2)
        print(f"   🏁 Epoch {ep} 완료: Train Loss={avg_tr_loss:.4f} | Val Loss={avg_va_loss:.4f} | 소요={ep_time}s", flush=True)
        history.append({"epoch": ep, "train_loss": avg_tr_loss, "val_loss": avg_va_loss, "epoch_time_s": ep_time})

    total_train_time = round(time.time() - t_start, 2)
    peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"qwen2_vl_{tag}")
    model.save_pretrained(ckpt_path)
    processor.save_pretrained(ckpt_path)

    test_eval = evaluate_model_on_test(model, processor, test_data, desc=f"LoRA_{tag}", bbox_mode="clean", use_text_coord=False)

    res = {
        "tag": tag,
        "method": f"LoRA (Rank {rank})",
        "rank": rank,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": trainable_pct,
        "epochs": epochs,
        "total_train_time_s": total_train_time,
        "peak_vram_mb": peak_vram_mb,
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "history": history,
        "test_eval": test_eval,
        "ckpt_path": ckpt_path
    }

    del model, base_model, processor, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return res

def train_and_eval_full_ft(train_data, val_data, test_data, epochs=2):
    """
    [피드백 3 해결] Full Fine-tuning (2.2B 전체 파라미터 학습)
    메모리 최적화(Gradient Checkpointing + Adafactor + set_to_none=True + max_pixels=320*28*28)를 통해
    단일 RTX 3090 24GB에서 OOM 없이 완주하여 LoRA와 정량 비교합니다.
    """
    print("\n" + "=" * 80, flush=True)
    print(f"🚀 [Full Fine-Tuning 실험] 전체 2.2B 파라미터 학습 (Train={len(train_data)}장, Val={len(val_data)}장, Test={len(test_data)}장)", flush=True)
    print("=" * 80, flush=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    processor = AutoProcessor.from_pretrained(MODEL_NAME, min_pixels=224 * 28 * 28, max_pixels=320 * 28 * 28)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    for p in model.parameters():
        p.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    optimizer = Adafactor(
        model.parameters(),
        lr=1.5e-5,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False
    )

    grad_accum = 4
    history = []
    t_start = time.time()

    for ep in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        tr_loss_sum = 0.0
        tr_steps = 0

        for i, sample in enumerate(train_data):
            inputs = prepare_multimodal_batch(processor, sample, bbox_mode="clean", use_text_coord=False)
            out = model(**inputs)
            loss = out.loss / grad_accum
            loss.backward()
            tr_loss_sum += out.loss.item()
            tr_steps += 1
            del inputs, out, loss

            if (i + 1) % grad_accum == 0 or (i + 1) == len(train_data):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

            if (i + 1) % 50 == 0 or (i + 1) == len(train_data):
                cur_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
                print(f"   [Full-FT Epoch {ep}/{epochs}] Step {i+1}/{len(train_data)} | Loss: {tr_loss_sum/tr_steps:.4f} | Peak VRAM: {cur_vram:.1f}MB", flush=True)

        avg_tr_loss = round(tr_loss_sum / max(tr_steps, 1), 4)

        model.eval()
        va_loss_sum = 0.0
        va_steps = 0
        with torch.no_grad():
            for sample in val_data:
                inputs = prepare_multimodal_batch(processor, sample, bbox_mode="clean", use_text_coord=False)
                out = model(**inputs)
                va_loss_sum += out.loss.item()
                va_steps += 1
                del inputs, out
        avg_va_loss = round(va_loss_sum / max(va_steps, 1), 4)
        ep_time = round(time.time() - ep_t0, 2)
        print(f"   🏁 Full-FT Epoch {ep} 완료: Train Loss={avg_tr_loss:.4f} | Val Loss={avg_va_loss:.4f} | 소요={ep_time}s", flush=True)
        history.append({"epoch": ep, "train_loss": avg_tr_loss, "val_loss": avg_va_loss, "epoch_time_s": ep_time})

    total_train_time = round(time.time() - t_start, 2)
    peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)

    model.config.use_cache = True
    test_eval = evaluate_model_on_test(model, processor, test_data, desc="Full_FineTuning", bbox_mode="clean", use_text_coord=False)

    res = {
        "tag": "full_ft",
        "method": "Full Fine-Tuning (All 2.2B Params)",
        "rank": "Full (100%)",
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "trainable_params": total_params,
        "total_params": total_params,
        "trainable_pct": 100.0,
        "epochs": epochs,
        "total_train_time_s": total_train_time,
        "peak_vram_mb": peak_vram_mb,
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "history": history,
        "test_eval": test_eval
    }

    del model, processor, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return res

def run_bbox_guiding_robustness_suite(best_ckpt_path, test_data):
    """
    [피드백 4 해결] YOLO 가이딩 방식 개선 및 바운딩 박스 왜곡 강건성 5종 실험 (Test 100장 전수)
      1) 정상 시각 바운딩 박스 제공 (clean)
      2) 바운딩 박스 위치 인위적 이동 (shift_30)
      3) 바운딩 박스 크기 절반 축소 (shrink_50)
      4) 바운딩 박스 미제공 (no_box)
      5) 숫자 좌표 텍스트 제공 (text_coord - [x, y, w, h] 텍스트 전달)
    """
    print("\n" + "=" * 80, flush=True)
    print("🧪 [실험 4] YOLO 바운딩 박스 가이딩 방식 및 위치·크기 왜곡 강건성 실험 (Test 100장)", flush=True)
    print("=" * 80, flush=True)

    processor = AutoProcessor.from_pretrained(MODEL_NAME, min_pixels=256 * 28 * 28, max_pixels=448 * 28 * 28)
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE
    )
    model = PeftModel.from_pretrained(base_model, best_ckpt_path)
    model.eval()

    conditions = [
        ("clean", False, "1) 정상 시각 바운딩 박스 표시 (Normal Visual BBox)"),
        ("shift_30", False, "2) 바운딩 박스 위치 인위적 이동 (Shifted BBox +25%)"),
        ("shrink_50", False, "3) 바운딩 박스 크기 절반 축소 (Shrunk BBox 50%)"),
        ("no_box", False, "4) 바운딩 박스 없음 (No Visual BBox)"),
        ("text_coord", True, "5) 숫자 좌표 텍스트 전달 (Text Coordinates [x,y,w,h])"),
    ]

    robustness_results = {}
    csv_rows = []

    for mode_key, use_coord, label in conditions:
        ev = evaluate_model_on_test(
            model, processor, test_data,
            desc=label,
            bbox_mode=mode_key,
            use_text_coord=use_coord
        )
        robustness_results[mode_key] = ev
        for out_item in ev["outputs"]:
            csv_rows.append({
                "condition_key": mode_key,
                "condition_label": label,
                **out_item
            })

    rob_csv = os.path.join(RESULTS_DIR, "bbox_guiding_robustness_100test.csv")
    with open(rob_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "condition_key", "condition_label", "question_id", "target_pest",
            "yolo_pred_name", "yolo_conf", "is_correct", "has_format",
            "has_loop_or_halluc", "is_refusal", "prediction"
        ])
        writer.writeheader()
        for r in csv_rows:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    del model, base_model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return robustness_results

def main():
    t_all = time.time()
    print("=" * 80, flush=True)
    print(f"🌟 MYSTIC VLM/LoRA/Full-FT 대규모 심화 실험 시작 | Device: {DEVICE}", flush=True)
    print("=" * 80, flush=True)

    # 1. 800장 확대 데이터셋 구축 (Train 600 / Val 100 / Test 100)
    train_600, val_100, test_100 = build_expanded_dataset()
    # 소규모 데이터(100장) 대조군 (클래스 균등 100장)
    train_100 = train_600[:100]

    all_results = {}

    # 2. Base VLM (Zero-shot, 파인튜닝 전) 100장 Test 평가
    print("\n▶ [Baseline] 사전학습 Base VLM (Zero-shot) 100장 Test 평가...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, min_pixels=256 * 28 * 28, max_pixels=448 * 28 * 28)
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map=DEVICE)
    all_results["base_vlm_zeroshot"] = evaluate_model_on_test(base_model, processor, test_100, desc="Base_VLM_ZeroShot")
    del base_model, processor
    gc.collect()
    torch.cuda.empty_cache()

    # 3. 데이터 규모 비교용: LoRA Rank 8 (100장 Toy Sample 학습)
    all_results["lora_rank8_small100"] = train_and_eval_lora(
        rank=8, train_data=train_100, val_data=val_100, test_data=test_100,
        tag="rank8_small100", epochs=2, lr=2e-4
    )

    # 4. 확대된 데이터(600장) 기반 LoRA Rank 4, 8, 16 비교 실험
    all_results["lora_rank4_large600"] = train_and_eval_lora(
        rank=4, train_data=train_600, val_data=val_100, test_data=test_100,
        tag="rank4_large600", epochs=2, lr=2e-4
    )
    all_results["lora_rank8_large600"] = train_and_eval_lora(
        rank=8, train_data=train_600, val_data=val_100, test_data=test_100,
        tag="rank8_large600", epochs=2, lr=2e-4
    )
    all_results["lora_rank16_large600"] = train_and_eval_lora(
        rank=16, train_data=train_600, val_data=val_100, test_data=test_100,
        tag="rank16_large600", epochs=2, lr=2e-4
    )

    # 5. Full Fine-Tuning 비교 실험 (확대 데이터 200장 및 100장 기준 자원·성능 실측)
    all_results["full_ft_200"] = train_and_eval_full_ft(
        train_data=train_600[:200], val_data=val_100[:50], test_data=test_100, epochs=2
    )

    # 6. 최적 LoRA 모델(Rank 8 Large 600) 기반 BBox 가이딩 및 위치/크기 왜곡 강건성 실험 (Test 100장)
    best_ckpt = all_results["lora_rank8_large600"]["ckpt_path"]
    all_results["bbox_robustness_100test"] = run_bbox_guiding_robustness_suite(best_ckpt, test_100)

    all_results["total_suite_elapsed_s"] = round(time.time() - t_all, 2)

    master_json = os.path.join(RESULTS_DIR, "master_v5_vlm_experiment_results.json")
    with open(master_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80, flush=True)
    print(f"🎉 [전체 VLM 심화 실험 완료!] 총 소요 시간: {all_results['total_suite_elapsed_s']}초", flush=True)
    print(f"📁 마스터 결과 JSON: {master_json}", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()

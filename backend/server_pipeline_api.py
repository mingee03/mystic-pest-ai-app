"""
MYSTIC 노지 해충 AI 진단 앱 백엔드 API 서버 (FastAPI)
=============================================================
[파이프라인 흐름]
1. Flutter 앱에서 촬영/선택한 해충 사진 업로드 (POST /api/v1/diagnose)
2. 1단계: YOLOv8-Large v5 (v5_large_improved_best.pt) 추론 -> BBox 좌표, 클래스, 확신도 추출
3. 중간 브릿지: 원본 이미지에 빨간색 시각적 바운딩 박스(Visual Prompting) 합성
4. 2단계: Qwen2-VL-2B-Instruct + LoRA Rank 16 (qwen2_vl_rank16_large600) 추론
5. 3단계 구조화 응답([진단 결과], [피해 증상], [권장 방제법]) 파싱 후 Flutter 앱으로 JSON 반환
"""

import io
import os
import time
import torch
from PIL import Image, ImageDraw
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

app = FastAPI(title="MYSTIC YOLO v5 + LoRA VLM Pipeline API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

YOLO_WEIGHTS = os.environ.get(
    "YOLO_WEIGHTS",
    "/home/ct2026storm/models/01_yolo_pest_detection/runs/controlled_experiment/v5_large_improved_500ep_earlystop/weights/best.pt",
)
LORA_CKPT = os.environ.get(
    "LORA_CKPT",
    "/home/ct2026storm/models/02_qwen2_vlm_lora/checkpoints_v5_expanded/qwen2_vl_rank16_large600",
)
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

yolo_model = None
vlm_model = None
vlm_processor = None


@app.on_event("startup")
def load_models():
    global yolo_model, vlm_model, vlm_processor
    print(f"🚀 Loading Stage 1 YOLOv8-Large v5 from {YOLO_WEIGHTS}...")
    if os.path.exists(YOLO_WEIGHTS):
        yolo_model = YOLO(YOLO_WEIGHTS)
    print(f"🚀 Loading Stage 2 Qwen2-VL-2B + LoRA Rank 16 from {LORA_CKPT}...")
    if os.path.exists(LORA_CKPT):
        vlm_processor = AutoProcessor.from_pretrained(MODEL_NAME, min_pixels=256 * 28 * 28, max_pixels=448 * 28 * 28)
        base = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, device_map=DEVICE
        )
        vlm_model = PeftModel.from_pretrained(base, LORA_CKPT)
        vlm_model.eval()


@app.post("/api/v1/diagnose")
async def diagnose_pest(image: UploadFile = File(...)):
    raw_bytes = await image.read()
    pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    W, H = pil_img.size

    # 1단계: YOLOv8-Large v5 탐지
    t0 = time.time()
    boxes_out = []
    yolo_pred_name = "정상"
    yolo_conf = 0.95

    if yolo_model is not None:
        res = yolo_model.predict(pil_img, imgsz=800, conf=0.20, verbose=False)[0]
        if len(res.boxes) > 0:
            best_i = res.boxes.conf.argmax().item()
            yolo_pred_name = yolo_model.names[int(res.boxes.cls[best_i].item())]
            yolo_conf = float(res.boxes.conf[best_i].item())
            for b in res.boxes:
                xywhn = b.xywhn[0].tolist()
                cid = int(b.cls[0].item())
                boxes_out.append({
                    "xc": round(xywhn[0], 4),
                    "yc": round(xywhn[1], 4),
                    "w": round(xywhn[2], 4),
                    "h": round(xywhn[3], 4),
                    "label": yolo_model.names[cid],
                    "confidence": round(float(b.conf[0].item()), 4),
                })
    yolo_ms = round((time.time() - t0) * 1000, 2)

    # 중간 브릿지: 시각적 바운딩 박스(Visual Prompting) 합성
    vis_img = pil_img.copy()
    if boxes_out:
        b0 = boxes_out[0]
        x1 = max(0, int((b0["xc"] - b0["w"] / 2.0) * W))
        y1 = max(0, int((b0["yc"] - b0["h"] / 2.0) * H))
        x2 = min(W - 1, int((b0["xc"] + b0["w"] / 2.0) * W))
        y2 = min(H - 1, int((b0["yc"] + b0["h"] / 2.0) * H))
        draw = ImageDraw.Draw(vis_img)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 30, 30), width=max(3, int(min(W, H) * 0.008)))

    # 2단계: Qwen2-VL-2B LoRA Rank 16 추론
    t1 = time.time()
    yolo_hint = (
        "[YOLO 1차 탐지] 정상 작물 (특이 해충 미검출)"
        if yolo_pred_name == "정상"
        else f"[YOLO 1차 탐지] {yolo_pred_name} (신뢰도: {yolo_conf*100:.1f}%)"
    )
    question = "표시된 바운딩 박스 영역의 작물 사진과 YOLO 1차 탐지 정보를 종합하여, 해충의 정확한 종류(시각적 형태 근거 포함), 작물 피해 증상, 권장 방제법을 3단계 양식으로 진단해주세요."
    prompt_text = f"{yolo_hint}\n\n[질문]\n{question}"

    if vlm_model is not None and vlm_processor is not None:
        messages = [{"role": "user", "content": [{"type": "image", "image": vis_img}, {"type": "text", "text": prompt_text}]}]
        text_prompt = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = vlm_processor(text=[text_prompt], images=[vis_img], padding=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen_ids = vlm_model.generate(**inputs, max_new_tokens=220, do_sample=False, repetition_penalty=1.15)
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
        pred_text = vlm_processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    else:
        pred_text = (
            f"[진단 결과] 발견된 해충은 '{yolo_pred_name}'입니다.\n"
            f"[피해 증상] 작물 잎과 줄기를 가해하여 생육 저하를 유발합니다.\n"
            f"[권장 방제법] 발생 초기 PLS 등록 약제를 안전사용기준에 맞춰 살포하십시오."
        )
    vlm_ms = round((time.time() - t1) * 1000, 2)

    # 3단계 섹션 분리 파싱
    diag_sec = pred_text
    symp_sec = ""
    ctrl_sec = ""
    if "[피해 증상]" in pred_text:
        parts = pred_text.split("[피해 증상]")
        diag_sec = parts[0].replace("[진단 결과]", "").strip()
        rest = parts[1]
        if "[권장 방제법]" in rest:
            sp2 = rest.split("[권장 방제법]")
            symp_sec = sp2[0].strip()
            ctrl_sec = sp2[1].strip()
        else:
            symp_sec = rest.strip()

    return {
        "yolo_pred_name": yolo_pred_name,
        "yolo_confidence": round(yolo_conf, 4),
        "detected_count": len(boxes_out),
        "boxes": boxes_out,
        "final_pest_name": yolo_pred_name,
        "growth_stage": "성충 / 유충",
        "is_self_corrected": False,
        "diagnosis_section": diag_sec,
        "symptom_section": symp_sec,
        "control_section": ctrl_sec,
        "yolo_latency_ms": yolo_ms,
        "vlm_latency_ms": vlm_ms,
    }

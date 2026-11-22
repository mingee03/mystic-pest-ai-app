class PestBoundingBox {
  final double xc; // normalized center X (0.0 ~ 1.0)
  final double yc; // normalized center Y (0.0 ~ 1.0)
  final double w;  // normalized width (0.0 ~ 1.0)
  final double h;  // normalized height (0.0 ~ 1.0)
  final String label;
  final double confidence; // 0.0 ~ 1.0

  const PestBoundingBox({
    required this.xc,
    required this.yc,
    required this.w,
    required this.h,
    required this.label,
    required this.confidence,
  });

  factory PestBoundingBox.fromJson(Map<String, dynamic> json) {
    return PestBoundingBox(
      xc: (json['xc'] as num).toDouble(),
      yc: (json['yc'] as num).toDouble(),
      w: (json['w'] as num).toDouble(),
      h: (json['h'] as num).toDouble(),
      label: json['label'] as String,
      confidence: (json['confidence'] as num).toDouble(),
    );
  }
}

class PestDiagnosisResult {
  final String id;
  final DateTime timestamp;
  final String imagePath;
  final bool isAsset;
  final String yoloPredName;
  final double yoloConfidence;
  final int detectedCount;
  final List<PestBoundingBox> boxes;
  final String finalPestName;
  final String growthStage;
  final bool isSelfCorrected;
  final String diagnosisSection; // [1. 진단 결과 및 시각적 형태 근거]
  final String symptomSection;   // [2. 작물 피해 증상 분석]
  final String controlSection;   // [3. PLS 맞춤형 권장 방제법]
  final double yoloLatencyMs;
  final double vlmLatencyMs;

  const PestDiagnosisResult({
    required this.id,
    required this.timestamp,
    required this.imagePath,
    this.isAsset = false,
    required this.yoloPredName,
    required this.yoloConfidence,
    required this.detectedCount,
    required this.boxes,
    required this.finalPestName,
    required this.growthStage,
    required this.isSelfCorrected,
    required this.diagnosisSection,
    required this.symptomSection,
    required this.controlSection,
    required this.yoloLatencyMs,
    required this.vlmLatencyMs,
  });

  factory PestDiagnosisResult.fromJson(
    Map<String, dynamic> json, {
    required String imagePath,
    bool isAsset = false,
  }) {
    final boxList = (json['boxes'] as List<dynamic>? ?? [])
        .map((b) => PestBoundingBox.fromJson(b as Map<String, dynamic>))
        .toList();
    return PestDiagnosisResult(
      id: json['id'] as String? ?? DateTime.now().millisecondsSinceEpoch.toString(),
      timestamp: DateTime.now(),
      imagePath: imagePath,
      isAsset: isAsset,
      yoloPredName: json['yolo_pred_name'] as String,
      yoloConfidence: (json['yolo_confidence'] as num).toDouble(),
      detectedCount: json['detected_count'] as int? ?? boxList.length,
      boxes: boxList,
      finalPestName: json['final_pest_name'] as String,
      growthStage: json['growth_stage'] as String? ?? '성충/유충',
      isSelfCorrected: json['is_self_corrected'] as bool? ?? false,
      diagnosisSection: json['diagnosis_section'] as String,
      symptomSection: json['symptom_section'] as String,
      controlSection: json['control_section'] as String,
      yoloLatencyMs: (json['yolo_latency_ms'] as num?)?.toDouble() ?? 5.6,
      vlmLatencyMs: (json['vlm_latency_ms'] as num?)?.toDouble() ?? 1850.0,
    );
  }
}

class SamplePestPreset {
  final String title;
  final String subtitle;
  final String cropName;
  final String sampleImagePath;
  final PestDiagnosisResult presetResult;

  const SamplePestPreset({
    required this.title,
    required this.subtitle,
    required this.cropName,
    required this.sampleImagePath,
    required this.presetResult,
  });
}

import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'models/pest_diagnosis.dart';
import 'services/pipeline_api_service.dart';

void main() {
  runApp(const MysticPestAiApp());
}

class MysticPestAiApp extends StatelessWidget {
  const MysticPestAiApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'MYSTIC 노지 해충 AI 진단',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF1B5E20),
          primary: const Color(0xFF1B5E20),
          secondary: const Color(0xFF2E7D32),
          surface: const Color(0xFFF6F9F6),
        ),
        scaffoldBackgroundColor: const Color(0xFFF4F7F4),
        cardTheme: CardThemeData(
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
            side: BorderSide(color: Colors.green.shade100),
          ),
          color: Colors.white,
        ),
      ),
      home: const MainNavigationScaffold(),
    );
  }
}

class MainNavigationScaffold extends StatefulWidget {
  const MainNavigationScaffold({super.key});

  @override
  State<MainNavigationScaffold> createState() => _MainNavigationScaffoldState();
}

class _MainNavigationScaffoldState extends State<MainNavigationScaffold> {
  int _currentIndex = 0;
  final PipelineApiService _apiService = PipelineApiService();
  final List<PestDiagnosisResult> _history = [];
  SamplePestPreset? _pendingPreset;

  void _navigateToCameraWithPreset(SamplePestPreset? preset) {
    setState(() {
      _pendingPreset = preset;
      _currentIndex = 1; // 해충 카메라 탭으로 이동
    });
  }

  void _addDiagnosisToHistory(PestDiagnosisResult result) {
    setState(() {
      _history.insert(0, result);
    });
  }

  @override
  Widget build(BuildContext context) {
    final screens = [
      HomeScreen(
        history: _history,
        onOpenCameraTab: () => _navigateToCameraWithPreset(null),
        onSelectPreset: (preset) => _navigateToCameraWithPreset(preset),
      ),
      PestCameraScreen(
        apiService: _apiService,
        initialPreset: _pendingPreset,
        onPresetConsumed: () {
          _pendingPreset = null;
        },
        onDiagnosisCompleted: _addDiagnosisToHistory,
      ),
      MyPageScreen(
        apiService: _apiService,
        historyCount: _history.length,
      ),
    ];

    return Scaffold(
      body: IndexedStack(
        index: _currentIndex,
        children: screens,
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _currentIndex,
        onDestinationSelected: (idx) {
          setState(() {
            _currentIndex = idx;
          });
        },
        backgroundColor: Colors.white,
        indicatorColor: const Color(0xFFC8E6C9),
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.home_outlined),
            selectedIcon: Icon(Icons.home_rounded, color: Color(0xFF1B5E20)),
            label: '홈',
          ),
          NavigationDestination(
            icon: Icon(Icons.camera_alt_outlined),
            selectedIcon: Icon(Icons.camera_alt_rounded, color: Color(0xFF1B5E20)),
            label: '해충 카메라',
          ),
          NavigationDestination(
            icon: Icon(Icons.person_outline_rounded),
            selectedIcon: Icon(Icons.person_rounded, color: Color(0xFF1B5E20)),
            label: '마이페이지',
          ),
        ],
      ),
    );
  }
}

// ============================================================================
// [탭 1] 홈 화면 (HomeScreen)
// ============================================================================
class HomeScreen extends StatelessWidget {
  final List<PestDiagnosisResult> history;
  final VoidCallback onOpenCameraTab;
  final void Function(SamplePestPreset preset) onSelectPreset;

  const HomeScreen({
    super.key,
    required this.history,
    required this.onOpenCameraTab,
    required this.onSelectPreset,
  });

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: CustomScrollView(
        slivers: [
          SliverAppBar(
            floating: true,
            backgroundColor: const Color(0xFF1B5E20),
            foregroundColor: Colors.white,
            title: const Row(
              children: [
                Icon(Icons.eco_rounded, color: Color(0xFFA5D6A7)),
                SizedBox(width: 8),
                Text(
                  'MYSTIC 노지 해충 AI 진단',
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 19),
                ),
              ],
            ),
            actions: [
              Padding(
                padding: const EdgeInsets.only(right: 12),
                child: Chip(
                  label: const Text(
                    'YOLO v5 + LoRA R16',
                    style: TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: Color(0xFF1B5E20)),
                  ),
                  backgroundColor: const Color(0xFFE8F5E9),
                  visualDensity: VisualDensity.compact,
                ),
              ),
            ],
          ),
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // 히어로 파이프라인 배너 카드
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      gradient: const LinearGradient(
                        colors: [Color(0xFF1B5E20), Color(0xFF388E3C)],
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                      ),
                      borderRadius: BorderRadius.circular(20),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.green.withValues(alpha: 0.25),
                          blurRadius: 12,
                          offset: const Offset(0, 6),
                        ),
                      ],
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                          decoration: BoxDecoration(
                            color: Colors.white.withValues(alpha: 0.18),
                            borderRadius: BorderRadius.circular(20),
                          ),
                          child: const Text(
                            '2-Stage 계층적 멀티모달 파이프라인 가동 중',
                            style: TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.w600),
                          ),
                        ),
                        const SizedBox(height: 12),
                        const Text(
                          '해충 촬영 한 번으로\n정밀 판별부터 PLS 방제 처방까지',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 21,
                            fontWeight: FontWeight.bold,
                            height: 1.3,
                          ),
                        ),
                        const SizedBox(height: 10),
                        const Text(
                          '1단계 YOLOv8-Large v5 (야생 탐지율 96%)\n2단계 Qwen2-VL-2B LoRA Rank 16 (진단 정확도 100%)',
                          style: TextStyle(color: Color(0xFFE8F5E9), fontSize: 13, height: 1.4),
                        ),
                        const SizedBox(height: 16),
                        SizedBox(
                          width: double.infinity,
                          child: ElevatedButton.icon(
                            onPressed: onOpenCameraTab,
                            icon: const Icon(Icons.camera_alt_rounded),
                            label: const Text(
                              '해충 카메라로 바로 진단하기',
                              style: TextStyle(fontWeight: FontWeight.bold, fontSize: 15),
                            ),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.white,
                              foregroundColor: const Color(0xFF1B5E20),
                              padding: const EdgeInsets.symmetric(vertical: 14),
                              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 24),

                  // 파이프라인 3단계 요약 카드
                  const Text(
                    '작동 파이프라인 구조',
                    style: TextStyle(fontSize: 17, fontWeight: FontWeight.bold, color: Color(0xFF1B5E20)),
                  ),
                  const SizedBox(height: 10),
                  Row(
                    children: [
                      _buildPipelineStepChip(
                        icon: Icons.center_focus_strong_rounded,
                        step: 'STEP 1',
                        title: 'YOLOv8L v5',
                        desc: '해충 위치 탐지\n& 시각 BBox 생성',
                      ),
                      const Icon(Icons.arrow_forward_ios_rounded, size: 14, color: Colors.grey),
                      _buildPipelineStepChip(
                        icon: Icons.psychology_rounded,
                        step: 'STEP 2',
                        title: 'LoRA VLM',
                        desc: '형태 특징 검증\n& 유사종 자가교정',
                      ),
                      const Icon(Icons.arrow_forward_ios_rounded, size: 14, color: Colors.grey),
                      _buildPipelineStepChip(
                        icon: Icons.medical_services_rounded,
                        step: 'STEP 3',
                        title: 'PLS 방제법',
                        desc: '피해 증상 분석\n& 맞춤 약제 처방',
                      ),
                    ],
                  ),
                  const SizedBox(height: 24),

                  // 대표 노지 해충 즉시 시연 (원클릭 파이프라인 테스트)
                  const Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        '노지 해충 샘플 원클릭 파이프라인 시연',
                        style: TextStyle(fontSize: 17, fontWeight: FontWeight.bold, color: Color(0xFF1B5E20)),
                      ),
                      Text(
                        '터치 시 즉시 판별',
                        style: TextStyle(fontSize: 12, color: Colors.grey),
                      ),
                    ],
                  ),
                  const SizedBox(height: 10),
                  ...PipelineApiService.samplePresets.map(
                    (preset) => Card(
                      margin: const EdgeInsets.only(bottom: 10),
                      child: ListTile(
                        contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
                        leading: ClipRRect(
                          borderRadius: BorderRadius.circular(10),
                          child: Image.asset(
                            preset.sampleImagePath,
                            width: 56,
                            height: 56,
                            fit: BoxFit.cover,
                          ),
                        ),
                        title: Text(
                          preset.title,
                          style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14),
                        ),
                        subtitle: Text(
                          '${preset.cropName} • ${preset.subtitle}',
                          style: const TextStyle(fontSize: 12),
                        ),
                        trailing: const Icon(Icons.play_circle_fill_rounded, color: Color(0xFF2E7D32), size: 30),
                        onTap: () => onSelectPreset(preset),
                      ),
                    ),
                  ),

                  const SizedBox(height: 16),
                  const Text(
                    '최근 진단 히스토리',
                    style: TextStyle(fontSize: 17, fontWeight: FontWeight.bold, color: Color(0xFF1B5E20)),
                  ),
                  const SizedBox(height: 10),
                  if (history.isEmpty)
                    Card(
                      child: Padding(
                        padding: const EdgeInsets.all(20),
                        child: Center(
                          child: Text(
                            '아직 진단 기록이 없습니다.\n[해충 카메라] 탭에서 사진을 찍거나 샘플을 선택해 보세요!',
                            textAlign: TextAlign.center,
                            style: TextStyle(color: Colors.grey.shade600, fontSize: 13),
                          ),
                        ),
                      ),
                    )
                  else
                    ...history.take(5).map(
                      (item) => Card(
                        margin: const EdgeInsets.only(bottom: 8),
                        child: ListTile(
                          leading: const CircleAvatar(
                            backgroundColor: Color(0xFFE8F5E9),
                            child: Icon(Icons.bug_report_rounded, color: Color(0xFF1B5E20)),
                          ),
                          title: Text(
                            '${item.finalPestName} (${item.growthStage})',
                            style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14),
                          ),
                          subtitle: Text(
                            'YOLO 1차: ${item.yoloPredName} (${(item.yoloConfidence * 100).toStringAsFixed(1)}%)',
                            style: const TextStyle(fontSize: 12),
                          ),
                          trailing: item.isSelfCorrected
                              ? Container(
                                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                                  decoration: BoxDecoration(
                                    color: Colors.orange.shade100,
                                    borderRadius: BorderRadius.circular(8),
                                  ),
                                  child: Text(
                                    'VLM 교정',
                                    style: TextStyle(
                                      fontSize: 11,
                                      fontWeight: FontWeight.bold,
                                      color: Colors.orange.shade900,
                                    ),
                                  ),
                                )
                              : const Icon(Icons.check_circle_rounded, color: Colors.green),
                        ),
                      ),
                    ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPipelineStepChip({
    required IconData icon,
    required String step,
    required String title,
    required String desc,
  }) {
    return Expanded(
      child: Card(
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 8),
          child: Column(
            children: [
              Icon(icon, color: const Color(0xFF2E7D32), size: 24),
              const SizedBox(height: 4),
              Text(
                step,
                style: const TextStyle(fontSize: 10, fontWeight: FontWeight.bold, color: Colors.grey),
              ),
              Text(
                title,
                style: const TextStyle(fontSize: 13, fontWeight: FontWeight.bold, color: Color(0xFF1B5E20)),
              ),
              const SizedBox(height: 4),
              Text(
                desc,
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 11, color: Colors.grey.shade700, height: 1.25),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ============================================================================
// [탭 2] 해충 카메라 & YOLO -> LoRA VLM 통합 판별 화면 (PestCameraScreen)
// ============================================================================
class PestCameraScreen extends StatefulWidget {
  final PipelineApiService apiService;
  final SamplePestPreset? initialPreset;
  final VoidCallback onPresetConsumed;
  final void Function(PestDiagnosisResult result) onDiagnosisCompleted;

  const PestCameraScreen({
    super.key,
    required this.apiService,
    required this.initialPreset,
    required this.onPresetConsumed,
    required this.onDiagnosisCompleted,
  });

  @override
  State<PestCameraScreen> createState() => _PestCameraScreenState();
}

class _PestCameraScreenState extends State<PestCameraScreen> {
  final ImagePicker _picker = ImagePicker();
  String? _selectedImagePath;
  bool _isAssetImage = true;
  bool _isRunningPipeline = false;
  String _pipelineStatusMessage = '';
  bool _showVisualBBox = true;
  PestDiagnosisResult? _result;

  @override
  void initState() {
    super.initState();
    // 초기 기본 샘플 로드
    final defaultPreset = widget.initialPreset ?? PipelineApiService.samplePresets.first;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _runWithPreset(defaultPreset);
    });
  }

  @override
  void didUpdateWidget(covariant PestCameraScreen oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.initialPreset != null && widget.initialPreset != oldWidget.initialPreset) {
      final preset = widget.initialPreset!;
      WidgetsBinding.instance.addPostFrameCallback((_) {
        widget.onPresetConsumed();
        _runWithPreset(preset);
      });
    }
  }

  Future<void> _runWithPreset(SamplePestPreset preset) async {
    setState(() {
      _selectedImagePath = preset.sampleImagePath;
      _isAssetImage = true;
      _isRunningPipeline = true;
      _result = null;
    });

    final res = await widget.apiService.runFullPipeline(
      imagePath: preset.sampleImagePath,
      isAsset: true,
      matchedPreset: preset,
      onProgress: (msg) {
        if (mounted) {
          setState(() {
            _pipelineStatusMessage = msg;
          });
        }
      },
    );

    if (mounted) {
      setState(() {
        _isRunningPipeline = false;
        _result = res;
      });
      widget.onDiagnosisCompleted(res);
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? picked = await _picker.pickImage(source: source, imageQuality: 92);
      if (picked == null) return;

      setState(() {
        _selectedImagePath = picked.path;
        _isAssetImage = false;
        _isRunningPipeline = true;
        _result = null;
      });

      final res = await widget.apiService.runFullPipeline(
        imagePath: picked.path,
        isAsset: false,
        onProgress: (msg) {
          if (mounted) {
            setState(() {
              _pipelineStatusMessage = msg;
            });
          }
        },
      );

      if (mounted) {
        setState(() {
          _isRunningPipeline = false;
          _result = res;
        });
        widget.onDiagnosisCompleted(res);
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('카메라/갤러리 호출 안내: 데스크톱 환경에서는 갤러리 파일 선택 또는 하단 샘플 칩을 이용해 주세요. ($e)')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Scaffold(
        appBar: AppBar(
          backgroundColor: const Color(0xFF1B5E20),
          foregroundColor: Colors.white,
          title: const Text(
            '해충 카메라 & AI 방제 처방',
            style: TextStyle(fontWeight: FontWeight.bold, fontSize: 18),
          ),
          actions: [
            if (_result != null)
              Row(
                children: [
                  const Text('시각 BBox', style: TextStyle(fontSize: 12)),
                  Switch(
                    value: _showVisualBBox,
                    activeThumbColor: const Color(0xFFA5D6A7),
                    onChanged: (val) => setState(() => _showVisualBBox = val),
                  ),
                ],
              ),
          ],
        ),
        body: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // 1. 촬영 및 사진 선택 버튼 바
              Row(
                children: [
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: _isRunningPipeline ? null : () => _pickImage(ImageSource.camera),
                      icon: const Icon(Icons.camera_alt_rounded),
                      label: const Text('해충 직접 촬영'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFF1B5E20),
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(vertical: 13),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                      ),
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: _isRunningPipeline ? null : () => _pickImage(ImageSource.gallery),
                      icon: const Icon(Icons.photo_library_rounded),
                      label: const Text('앨범에서 선택'),
                      style: OutlinedButton.styleFrom(
                        foregroundColor: const Color(0xFF1B5E20),
                        side: const BorderSide(color: Color(0xFF1B5E20), width: 1.5),
                        padding: const EdgeInsets.symmetric(vertical: 13),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),

              // 시연용 빠른 샘플 테스트 칩
              SizedBox(
                height: 38,
                child: ListView.separated(
                  scrollDirection: Axis.horizontal,
                  itemCount: PipelineApiService.samplePresets.length,
                  separatorBuilder: (_, __) => const SizedBox(width: 8),
                  itemBuilder: (context, idx) {
                    final preset = PipelineApiService.samplePresets[idx];
                    final isSelected = _selectedImagePath == preset.sampleImagePath;
                    return ActionChip(
                      avatar: Icon(
                        Icons.bug_report_rounded,
                        size: 16,
                        color: isSelected ? Colors.white : const Color(0xFF1B5E20),
                      ),
                      label: Text(
                        preset.presetResult.finalPestName,
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                          color: isSelected ? Colors.white : const Color(0xFF1B5E20),
                        ),
                      ),
                      backgroundColor: isSelected ? const Color(0xFF2E7D32) : const Color(0xFFE8F5E9),
                      onPressed: _isRunningPipeline ? null : () => _runWithPreset(preset),
                    );
                  },
                ),
              ),
              const SizedBox(height: 14),

              // 2. 이미지 + 시각적 바운딩 박스(Visual Prompting) 뷰어
              Container(
                width: double.infinity,
                height: 270,
                decoration: BoxDecoration(
                  color: Colors.black87,
                  borderRadius: BorderRadius.circular(18),
                  border: Border.all(color: Colors.green.shade200, width: 2),
                ),
                clipBehavior: Clip.antiAlias,
                child: _selectedImagePath == null
                    ? const Center(
                        child: Text(
                          '해충 사진을 촬영하거나 선택해 주세요',
                          style: TextStyle(color: Colors.white70),
                        ),
                      )
                    : Stack(
                        fit: StackFit.expand,
                        children: [
                          _isAssetImage
                              ? Image.asset(_selectedImagePath!, fit: BoxFit.contain)
                              : Image.file(File(_selectedImagePath!), fit: BoxFit.contain),
                          if (_showVisualBBox && _result != null && !_isRunningPipeline)
                            CustomPaint(
                              painter: VisualBBoxPainter(boxes: _result!.boxes),
                            ),
                          if (_isRunningPipeline)
                            Container(
                              color: Colors.black54,
                              padding: const EdgeInsets.all(24),
                              child: Column(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  const CircularProgressIndicator(color: Color(0xFFA5D6A7)),
                                  const SizedBox(height: 16),
                                  Text(
                                    _pipelineStatusMessage,
                                    textAlign: TextAlign.center,
                                    style: const TextStyle(
                                      color: Colors.white,
                                      fontWeight: FontWeight.bold,
                                      fontSize: 14,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                        ],
                      ),
              ),
              const SizedBox(height: 16),

              // 3. 파이프라인 추론 결과 표시 영역
              if (_result != null && !_isRunningPipeline) ...[
                // Stage 1: YOLOv8-Large v5 탐지 메타데이터 배지 카드
                Card(
                  color: const Color(0xFFE8F5E9),
                  child: Padding(
                    padding: const EdgeInsets.all(14),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Container(
                              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                              decoration: BoxDecoration(
                                color: const Color(0xFF1B5E20),
                                borderRadius: BorderRadius.circular(6),
                              ),
                              child: const Text(
                                '1단계 YOLOv8-Large v5',
                                style: TextStyle(color: Colors.white, fontSize: 11, fontWeight: FontWeight.bold),
                              ),
                            ),
                            const SizedBox(width: 8),
                            Text(
                              '탐지 개체: ${_result!.detectedCount}개 • 추론 ${_result!.yoloLatencyMs.toStringAsFixed(1)}ms',
                              style: TextStyle(fontSize: 12, color: Colors.green.shade900, fontWeight: FontWeight.w600),
                            ),
                            const Spacer(),
                            if (_result!.isSelfCorrected)
                              Container(
                                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                                decoration: BoxDecoration(
                                  color: Colors.deepOrange,
                                  borderRadius: BorderRadius.circular(6),
                                ),
                                child: const Text(
                                  '2단계 VLM 유사종 교정 발동',
                                  style: TextStyle(color: Colors.white, fontSize: 11, fontWeight: FontWeight.bold),
                                ),
                              ),
                          ],
                        ),
                        const SizedBox(height: 10),
                        Row(
                          children: [
                            Expanded(
                              child: Text(
                                '1차 YOLO 예측: ${_result!.yoloPredName} (${(_result!.yoloConfidence * 100).toStringAsFixed(1)}%)',
                                style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600),
                              ),
                            ),
                            const Icon(Icons.arrow_forward_rounded, size: 16, color: Color(0xFF1B5E20)),
                            const SizedBox(width: 6),
                            Expanded(
                              child: Text(
                                '최종 확정: ${_result!.finalPestName}',
                                style: const TextStyle(
                                  fontSize: 14,
                                  fontWeight: FontWeight.bold,
                                  color: Color(0xFF1B5E20),
                                ),
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 12),

                // Stage 2: Qwen2-VL-2B LoRA Rank 16 3단계 진단 및 방제 처방 카드
                _buildSectionCard(
                  badge: '2단계 Qwen2-VL-2B LoRA (Rank 16)',
                  icon: Icons.verified_rounded,
                  iconColor: const Color(0xFF1B5E20),
                  title: '1. 진단 결과 및 시각적 형태 근거 (${_result!.growthStage})',
                  content: _result!.diagnosisSection,
                ),
                const SizedBox(height: 10),
                _buildSectionCard(
                  badge: '작물 병해충 피해 분석',
                  icon: Icons.warning_amber_rounded,
                  iconColor: Colors.orange.shade800,
                  title: '2. 예상 작물 피해 증상',
                  content: _result!.symptomSection,
                ),
                const SizedBox(height: 10),
                _buildSectionCard(
                  badge: '농촌진흥청 PLS 안전사용기준 준수',
                  icon: Icons.health_and_safety_rounded,
                  iconColor: Colors.teal.shade700,
                  title: '3. 맞춤형 권장 방제법 및 약제 처방',
                  content: _result!.controlSection,
                  highlight: true,
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSectionCard({
    required String badge,
    required IconData icon,
    required Color iconColor,
    required String title,
    required String content,
    bool highlight = false,
  }) {
    return Card(
      color: highlight ? const Color(0xFFF1F8E9) : Colors.white,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(icon, color: iconColor, size: 20),
                const SizedBox(width: 6),
                Text(
                  badge,
                  style: TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: iconColor),
                ),
              ],
            ),
            const SizedBox(height: 6),
            Text(
              title,
              style: const TextStyle(fontSize: 15, fontWeight: FontWeight.bold, color: Colors.black87),
            ),
            const Divider(height: 18),
            Text(
              content,
              style: const TextStyle(fontSize: 13.5, height: 1.55, color: Colors.black87),
            ),
          ],
        ),
      ),
    );
  }
}

/// 시각적 바운딩 박스(Visual Prompting)를 이미지 위에 렌더링하는 CustomPainter
class VisualBBoxPainter extends CustomPainter {
  final List<PestBoundingBox> boxes;

  const VisualBBoxPainter({required this.boxes});

  @override
  void paint(Canvas canvas, Size size) {
    final boxPaint = Paint()
      ..color = const Color(0xFFFF1E1E)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;

    final fillPaint = Paint()
      ..color = const Color(0x22FF1E1E)
      ..style = PaintingStyle.fill;

    for (final b in boxes) {
      final left = (b.xc - b.w / 2.0).clamp(0.0, 1.0) * size.width;
      final top = (b.yc - b.h / 2.0).clamp(0.0, 1.0) * size.height;
      final right = (b.xc + b.w / 2.0).clamp(0.0, 1.0) * size.width;
      final bottom = (b.yc + b.h / 2.0).clamp(0.0, 1.0) * size.height;

      final rect = Rect.fromLTRB(left, top, right, bottom);
      canvas.drawRect(rect, fillPaint);
      canvas.drawRect(rect, boxPaint);

      final labelText = '${b.label} ${(b.confidence * 100).toStringAsFixed(1)}%';
      final tp = TextPainter(
        text: TextSpan(
          text: labelText,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 11,
            fontWeight: FontWeight.bold,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();

      final tagTop = (top - tp.height - 6) < 0 ? top : (top - tp.height - 6);
      final tagRect = Rect.fromLTWH(left, tagTop, tp.width + 10, tp.height + 6);
      canvas.drawRRect(
        RRect.fromRectAndRadius(tagRect, const Radius.circular(4)),
        Paint()..color = const Color(0xE6D50000),
      );
      tp.paint(canvas, Offset(left + 5, tagTop + 3));
    }
  }

  @override
  bool shouldRepaint(covariant VisualBBoxPainter oldDelegate) => oldDelegate.boxes != boxes;
}

// ============================================================================
// [탭 3] 마이페이지 화면 (MyPageScreen)
// ============================================================================
class MyPageScreen extends StatefulWidget {
  final PipelineApiService apiService;
  final int historyCount;

  const MyPageScreen({
    super.key,
    required this.apiService,
    required this.historyCount,
  });

  @override
  State<MyPageScreen> createState() => _MyPageScreenState();
}

class _MyPageScreenState extends State<MyPageScreen> {
  late TextEditingController _urlController;

  @override
  void initState() {
    super.initState();
    _urlController = TextEditingController(text: widget.apiService.serverUrl);
  }

  @override
  void dispose() {
    _urlController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Scaffold(
        appBar: AppBar(
          backgroundColor: const Color(0xFF1B5E20),
          foregroundColor: Colors.white,
          title: const Text(
            '마이페이지 & 파이프라인 설정',
            style: TextStyle(fontWeight: FontWeight.bold, fontSize: 18),
          ),
        ),
        body: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            // 농가 및 프로젝트 프로필 카드
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Row(
                  children: [
                    const CircleAvatar(
                      radius: 28,
                      backgroundColor: Color(0xFF1B5E20),
                      child: Icon(Icons.agriculture_rounded, color: Colors.white, size: 30),
                    ),
                    const SizedBox(width: 14),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            'MYSTIC 스마트 영농 연구팀',
                            style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            '참여기업: (주)스톰힐 이노베이션\n지도교수: 최희열 교수님 | 누적 진단: ${widget.historyCount}건',
                            style: TextStyle(fontSize: 12, color: Colors.grey.shade700, height: 1.4),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),

            // 탑재된 2-Stage AI 파이프라인 스펙
            const Text(
              '탑재된 AI 파이프라인 모델 정보',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Color(0xFF1B5E20)),
            ),
            const SizedBox(height: 8),
            const Card(
              child: Column(
                children: [
                  ListTile(
                    leading: Icon(Icons.radar_rounded, color: Color(0xFF1B5E20)),
                    title: Text('1단계 객체 탐지: YOLOv8-Large v5', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 14)),
                    subtitle: Text('800×800 Letterbox • 유사종 2.0x 오버샘플링 • 야생 탐지율 96.0% (251 EP Peak)'),
                  ),
                  Divider(height: 1),
                  ListTile(
                    leading: Icon(Icons.auto_awesome_rounded, color: Color(0xFF1B5E20)),
                    title: Text('2단계 정밀 진단: Qwen2-VL-2B LoRA (Rank 16)', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 14)),
                    subtitle: Text('800장 확장 데이터셋 파인튜닝 • 시각적 BBox 가이딩 • Test 100장 정확도 100.0%'),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),

            // FastAPI 백엔드 서버 연동 설정
            const Text(
              '추론 백엔드 서버(FastAPI) 연동 설정',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Color(0xFF1B5E20)),
            ),
            const SizedBox(height: 8),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    SwitchListTile(
                      contentPadding: EdgeInsets.zero,
                      title: const Text(
                        '실시간 FastAPI GPU 서버 연동 모드',
                        style: TextStyle(fontWeight: FontWeight.bold, fontSize: 14),
                      ),
                      subtitle: const Text(
                        'OFF 시 앱 내장 파이프라인 시연 엔진으로 즉시 동작하며, ON 시 아래 FastAPI 서버로 사진을 전송합니다.',
                        style: TextStyle(fontSize: 12),
                      ),
                      value: widget.apiService.useLiveServer,
                      activeThumbColor: const Color(0xFF1B5E20),
                      onChanged: (val) {
                        setState(() {
                          widget.apiService.useLiveServer = val;
                        });
                      },
                    ),
                    const SizedBox(height: 10),
                    TextField(
                      controller: _urlController,
                      decoration: InputDecoration(
                        labelText: 'FastAPI 서버 주소 (예: http://127.0.0.1:8000)',
                        border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
                        isDense: true,
                      ),
                      onChanged: (val) {
                        widget.apiService.serverUrl = val.trim();
                      },
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

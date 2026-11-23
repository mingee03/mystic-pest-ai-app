import 'package:flutter/foundation.dart';
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
          seedColor: _green,
          primary: _green,
          secondary: const Color(0xFF2E7D32),
          surface: const Color(0xFFF6F9F6),
        ),
        scaffoldBackgroundColor: _page,
        cardTheme: CardThemeData(
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
            side: BorderSide(color: const Color(0xFFD9E5DA)),
          ),
          color: Colors.white,
        ),
      ),
      home: const MainNavigationScaffold(),
    );
  }
}

const _green = Color(0xFF184B37);
const _muted = Color(0xFF496358);
const _page = Color(0xFFF4F7F2);
const _desktopWidth = 900.0;

class MainNavigationScaffold extends StatefulWidget {
  const MainNavigationScaffold({super.key});

  @override
  State<MainNavigationScaffold> createState() => _MainNavigationScaffoldState();
}

class _MainNavigationScaffoldState extends State<MainNavigationScaffold> {
  int _index = 0;
  final _api = PipelineApiService();
  final _camera = GlobalKey<_PestCameraScreenState>();
  final List<PestDiagnosisResult> _history = [];

  static const _labels = ['홈', '해충 카메라', '마이페이지'];
  static const _icons = [
    Icons.home_outlined,
    Icons.camera_alt_outlined,
    Icons.person_outline,
  ];

  void _select(int index) => setState(() => _index = index);

  void _start({ImageSource? source, SamplePestPreset? sample}) {
    _select(1);
    if (sample != null) {
      _camera.currentState?._runWithPreset(sample);
    } else if (source != null) {
      _camera.currentState?._pickImage(source);
    }
  }

  @override
  Widget build(BuildContext context) {
    final desktop = MediaQuery.sizeOf(context).width >= _desktopWidth;
    final screens = [
      HomeScreen(
        history: _history,
        onPick: (source) => _start(source: source),
        onSelectPreset: (sample) => _start(sample: sample),
      ),
      PestCameraScreen(
        key: _camera,
        apiService: _api,
        onDiagnosisCompleted:
            (result) => setState(() => _history.insert(0, result)),
      ),
      MyPageScreen(apiService: _api, historyCount: _history.length),
    ];

    return Scaffold(
      appBar: AppBar(
        toolbarHeight: 64,
        backgroundColor: Colors.white,
        surfaceTintColor: Colors.transparent,
        titleSpacing: 0,
        title: _Content(
          child: Row(
            children: [
              const Icon(Icons.eco_rounded, color: _green),
              const SizedBox(width: 8),
              const Flexible(
                child: Text(
                  'MYSTIC  |  노지 해충 AI',
                  style: TextStyle(
                    color: _green,
                    fontSize: 17,
                    fontWeight: FontWeight.w800,
                  ),
                ),
              ),
              if (desktop) ...[
                const Spacer(),
                for (var i = 0; i < _labels.length; i++)
                  Padding(
                    padding: const EdgeInsets.only(left: 8),
                    child: Semantics(
                      selected: _index == i,
                      child: TextButton.icon(
                        onPressed: () => _select(i),
                        style: TextButton.styleFrom(
                          backgroundColor:
                              _index == i ? const Color(0xFFE4EFE5) : null,
                        ),
                        icon: Icon(_icons[i], size: 18),
                        label: Text(_labels[i]),
                      ),
                    ),
                  ),
              ],
            ],
          ),
        ),
      ),
      body: SafeArea(
        top: false,
        bottom: desktop,
        child: IndexedStack(index: _index, children: screens),
      ),
      bottomNavigationBar:
          desktop
              ? null
              : NavigationBar(
                selectedIndex: _index,
                onDestinationSelected: _select,
                backgroundColor: Colors.white,
                indicatorColor: const Color(0xFFE4EFE5),
                destinations: [
                  for (var i = 0; i < _labels.length; i++)
                    NavigationDestination(
                      icon: Icon(_icons[i]),
                      label: _labels[i],
                    ),
                ],
              ),
    );
  }
}

class _Content extends StatelessWidget {
  const _Content({required this.child});
  final Widget child;

  @override
  Widget build(BuildContext context) {
    final gutter = MediaQuery.sizeOf(context).width >= 600 ? 32.0 : 16.0;
    return Align(
      alignment: Alignment.topCenter,
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 1200),
        child: Padding(
          padding: EdgeInsets.symmetric(horizontal: gutter),
          child: SizedBox(width: double.infinity, child: child),
        ),
      ),
    );
  }
}

class _PhotoActions extends StatelessWidget {
  const _PhotoActions({required this.onPick, this.onDark = false});

  final ValueChanged<ImageSource>? onPick;
  final bool onDark;

  @override
  Widget build(BuildContext context) {
    final pick = onPick;
    return Wrap(
      spacing: 10,
      runSpacing: 8,
      children: [
        FilledButton.icon(
          onPressed: pick == null ? null : () => pick(ImageSource.camera),
          style: FilledButton.styleFrom(
            minimumSize: const Size(0, 48),
            backgroundColor: onDark ? const Color(0xFFE2F3D1) : _green,
            foregroundColor: onDark ? _green : Colors.white,
          ),
          icon: const Icon(Icons.camera_alt_outlined),
          label: const Text('사진 촬영'),
        ),
        OutlinedButton.icon(
          onPressed: pick == null ? null : () => pick(ImageSource.gallery),
          style: OutlinedButton.styleFrom(
            minimumSize: const Size(0, 48),
            foregroundColor: onDark ? Colors.white : _green,
            side: const BorderSide(color: Color(0xFF8EA79A)),
          ),
          icon: const Icon(Icons.upload_file_outlined),
          label: const Text('이미지 업로드'),
        ),
      ],
    );
  }
}

class HomeScreen extends StatelessWidget {
  const HomeScreen({
    super.key,
    required this.history,
    required this.onPick,
    required this.onSelectPreset,
  });

  final List<PestDiagnosisResult> history;
  final ValueChanged<ImageSource> onPick;
  final ValueChanged<SamplePestPreset> onSelectPreset;

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      key: const PageStorageKey('home'),
      padding: const EdgeInsets.symmetric(vertical: 20),
      child: _Content(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Container(
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(24),
                gradient: const LinearGradient(
                  colors: [Color(0xFF133D2D), Color(0xFF24634A)],
                ),
              ),
              padding: const EdgeInsets.all(24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    '연구용 프로토타입',
                    style: TextStyle(
                      color: Color(0xFFCEF0BB),
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    '사진으로 노지 해충을 확인하세요',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize:
                          MediaQuery.sizeOf(context).width >= 600 ? 34 : 25,
                      height: 1.2,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                  const SizedBox(height: 8),
                  const Text(
                    '해충 후보와 판별 근거, 방제 참고 정보를 살펴보세요.',
                    style: TextStyle(color: Color(0xFFE3EFE6), height: 1.45),
                  ),
                  const SizedBox(height: 16),
                  _PhotoActions(onPick: onPick, onDark: true),
                ],
              ),
            ),
            const SizedBox(height: 24),
            const _SectionHeading('AI 진단 과정'),
            const SizedBox(height: 10),
            const _ResponsiveCards(
              minWidth: 250,
              maxColumns: 3,
              children: [
                _PipelineStep(
                  Icons.center_focus_strong,
                  '1. 해충 탐지',
                  'YOLOv8-Large v5가 해충 위치를 찾습니다.',
                ),
                _PipelineStep(
                  Icons.visibility_outlined,
                  '2. 형태 검증',
                  'Qwen2-VL LoRA가 형태와 유사종을 비교합니다.',
                ),
                _PipelineStep(
                  Icons.eco_outlined,
                  '3. 방제 정보',
                  '예상 피해와 PLS 방제 참고 정보를 정리합니다.',
                ),
              ],
            ),
            const SizedBox(height: 24),
            const _SectionHeading('샘플 사진으로 체험'),
            const SizedBox(height: 4),
            const Text(
              '사진을 선택하면 내장 시연 결과를 볼 수 있습니다.',
              style: TextStyle(color: _muted),
            ),
            const SizedBox(height: 12),
            _ResponsiveCards(
              minWidth: 148,
              maxColumns: 3,
              children: [
                for (final sample in PipelineApiService.samplePresets)
                  _SampleCard(
                    sample: sample,
                    onTap: () => onSelectPreset(sample),
                  ),
              ],
            ),
            const SizedBox(height: 24),
            const _SectionHeading('최근 진단'),
            const SizedBox(height: 10),
            if (history.isEmpty)
              const Card(
                child: ListTile(
                  leading: Icon(Icons.history, color: _green),
                  title: Text('아직 진단 기록이 없습니다.'),
                ),
              )
            else
              for (final item in history.take(5))
                Card(
                  child: ListTile(
                    leading: const Icon(
                      Icons.bug_report_outlined,
                      color: _green,
                    ),
                    title: Text(item.finalPestName),
                    subtitle: Text(
                      '${item.growthStage} · YOLO 후보 ${item.yoloPredName}',
                    ),
                  ),
                ),
          ],
        ),
      ),
    );
  }
}

class _SectionHeading extends StatelessWidget {
  const _SectionHeading(this.title);
  final String title;

  @override
  Widget build(BuildContext context) => Text(
    title,
    style: Theme.of(context).textTheme.titleLarge?.copyWith(
      color: _green,
      fontWeight: FontWeight.w800,
    ),
  );
}

class _ResponsiveCards extends StatelessWidget {
  const _ResponsiveCards({
    required this.children,
    required this.minWidth,
    required this.maxColumns,
  });
  final List<Widget> children;
  final double minWidth;
  final int maxColumns;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        const gap = 12.0;
        final columns = ((constraints.maxWidth + gap) / (minWidth + gap))
            .floor()
            .clamp(1, maxColumns);
        final width = (constraints.maxWidth - (columns - 1) * gap) / columns;
        return Wrap(
          spacing: gap,
          runSpacing: gap,
          children: [
            for (final child in children) SizedBox(width: width, child: child),
          ],
        );
      },
    );
  }
}

class _PipelineStep extends StatelessWidget {
  const _PipelineStep(this.icon, this.title, this.description);
  final IconData icon;
  final String title;
  final String description;

  @override
  Widget build(BuildContext context) => Card(
    margin: EdgeInsets.zero,
    child: Padding(
      padding: const EdgeInsets.all(14),
      child: Row(
        children: [
          Icon(icon, color: _green),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    fontWeight: FontWeight.w800,
                    color: _green,
                  ),
                ),
                const SizedBox(height: 3),
                Text(
                  description,
                  style: const TextStyle(color: _muted, height: 1.35),
                ),
              ],
            ),
          ),
        ],
      ),
    ),
  );
}

class _SampleCard extends StatelessWidget {
  const _SampleCard({required this.sample, required this.onTap});
  final SamplePestPreset sample;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: EdgeInsets.zero,
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: onTap,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            AspectRatio(
              aspectRatio: 1.65,
              child: Ink.image(
                image: AssetImage(sample.sampleImagePath),
                fit: BoxFit.cover,
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(12),
              child: Row(
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          sample.presetResult.finalPestName,
                          style: const TextStyle(
                            color: _green,
                            fontWeight: FontWeight.w800,
                          ),
                        ),
                        const SizedBox(height: 3),
                        Text(
                          sample.cropName,
                          style: const TextStyle(color: _muted),
                        ),
                      ],
                    ),
                  ),
                  const Icon(
                    Icons.play_circle_outline,
                    size: 21,
                    color: _green,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class PestCameraScreen extends StatefulWidget {
  const PestCameraScreen({
    super.key,
    required this.apiService,
    required this.onDiagnosisCompleted,
  });

  final PipelineApiService apiService;
  final ValueChanged<PestDiagnosisResult> onDiagnosisCompleted;

  @override
  State<PestCameraScreen> createState() => _PestCameraScreenState();
}

class _PestCameraScreenState extends State<PestCameraScreen> {
  final ImagePicker _picker = ImagePicker();
  String? _selectedImagePath;
  Uint8List? _imageBytes;
  bool _picking = false;
  bool _running = false;
  bool _showBoxes = true;
  bool _fromServer = false;
  String _status = '';
  PestDiagnosisResult? _result;

  bool get _busy => _picking || _running;

  Future<void> _runWithPreset(SamplePestPreset preset) async {
    if (_busy) return;
    await _run(imagePath: preset.sampleImagePath, preset: preset);
  }

  Future<void> _pickImage(ImageSource source) async {
    if (_busy) return;
    setState(() => _picking = true);
    try {
      final picked = await _picker.pickImage(source: source, imageQuality: 92);
      if (picked == null || !mounted) return;
      final bytes = await picked.readAsBytes();
      if (!mounted) return;
      await _run(imagePath: picked.path, fileName: picked.name, bytes: bytes);
    } catch (_) {
      if (mounted) {
        _showMessage(
          source == ImageSource.camera
              ? '카메라를 열 수 없습니다. 권한을 확인하거나 이미지를 업로드해 주세요.'
              : '이미지를 열 수 없습니다. 다른 사진을 선택해 주세요.',
        );
      }
    } finally {
      if (mounted) setState(() => _picking = false);
    }
  }

  void _showMessage(String message) {
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(SnackBar(content: Text(message)));
  }

  Future<void> _run({
    required String imagePath,
    String? fileName,
    Uint8List? bytes,
    SamplePestPreset? preset,
  }) async {
    setState(() {
      _selectedImagePath = imagePath;
      _imageBytes = bytes;
      _running = true;
      _status = '진단을 준비하는 중';
      _result = null;
    });

    var fromServer = false;
    try {
      final result = await widget.apiService.runFullPipeline(
        imagePath: imagePath,
        isAsset: preset != null,
        matchedPreset: preset,
        uploadBytes: kIsWeb ? bytes : null,
        uploadFileName: fileName,
        onServerResult: () => fromServer = true,
        onProgress: (message) {
          if (mounted) setState(() => _status = message);
        },
      );
      if (!mounted) return;
      setState(() {
        _result = result;
        _fromServer = fromServer;
      });
      widget.onDiagnosisCompleted(result);
    } catch (_) {
      if (mounted) _showMessage('진단을 완료하지 못했습니다. 다시 시도해 주세요.');
    } finally {
      if (mounted) setState(() => _running = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final result = _running ? null : _result;
    return SingleChildScrollView(
      key: const PageStorageKey('diagnosis'),
      padding: const EdgeInsets.symmetric(vertical: 20),
      child: _Content(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const _SectionHeading('해충 사진 진단'),
            const SizedBox(height: 10),
            _PhotoActions(onPick: _busy ? null : _pickImage),
            const SizedBox(height: 12),
            Wrap(
              spacing: 8,
              runSpacing: 6,
              children: [
                for (final preset in PipelineApiService.samplePresets)
                  ChoiceChip(
                    label: Text(preset.presetResult.finalPestName),
                    selected: _selectedImagePath == preset.sampleImagePath,
                    onSelected: _busy ? null : (_) => _runWithPreset(preset),
                  ),
              ],
            ),
            const SizedBox(height: 16),
            LayoutBuilder(
              builder: (context, constraints) {
                final photo = _buildPhoto(
                  result,
                  height: constraints.maxWidth >= 780 ? 360 : 260,
                );
                if (result == null) return photo;
                final details = _buildResult(result);
                if (constraints.maxWidth < 780) {
                  return Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [photo, const SizedBox(height: 14), details],
                  );
                }
                return Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Expanded(flex: 5, child: photo),
                    const SizedBox(width: 16),
                    Expanded(flex: 7, child: details),
                  ],
                );
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPhoto(PestDiagnosisResult? result, {required double height}) {
    final path = _selectedImagePath;
    final bytes = _imageBytes;
    final Widget? image =
        bytes != null
            ? Image.memory(
              bytes,
              semanticLabel: '선택한 해충 사진',
              errorBuilder:
                  (_, _, _) => const Text(
                    '사진 미리보기를 표시할 수 없습니다.',
                    style: TextStyle(color: Colors.white),
                  ),
            )
            : path == null
            ? null
            : Image.asset(path, semanticLabel: '샘플 해충 사진');
    final showToggle = result != null && result.boxes.isNotEmpty;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Container(
          width: double.infinity,
          height: height,
          decoration: BoxDecoration(
            color: const Color(0xFF17271F),
            borderRadius: BorderRadius.circular(18),
          ),
          clipBehavior: Clip.antiAlias,
          child: Stack(
            fit: StackFit.expand,
            children: [
              if (image == null)
                const Center(
                  child: Text(
                    '사진을 촬영하거나 이미지를 선택해 주세요.',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: Colors.white),
                  ),
                )
              else
                FittedBox(
                  fit: BoxFit.contain,
                  child: Stack(
                    children: [
                      image,
                      if (showToggle && _showBoxes)
                        Positioned.fill(
                          child: CustomPaint(
                            painter: VisualBBoxPainter(boxes: result.boxes),
                          ),
                        ),
                    ],
                  ),
                ),
              if (_running)
                ColoredBox(
                  color: Colors.black54,
                  child: Center(
                    child: Padding(
                      padding: const EdgeInsets.all(18),
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          const CircularProgressIndicator(
                            color: Color(0xFFCEF0BB),
                          ),
                          const SizedBox(height: 12),
                          Semantics(
                            liveRegion: true,
                            child: Text(
                              _status,
                              textAlign: TextAlign.center,
                              style: const TextStyle(
                                color: Colors.white,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
            ],
          ),
        ),
        if (showToggle)
          SwitchListTile(
            contentPadding: const EdgeInsets.symmetric(horizontal: 8),
            title: Text(_fromServer ? '탐지 영역 표시' : '시연 탐지 영역 표시'),
            value: _showBoxes,
            onChanged: (value) => setState(() => _showBoxes = value),
          ),
      ],
    );
  }

  Widget _buildResult(PestDiagnosisResult result) {
    final confidence = (result.yoloConfidence * 100).toStringAsFixed(1);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Card(
          margin: EdgeInsets.zero,
          color: const Color(0xFFE6F0E5),
          child: Padding(
            padding: const EdgeInsets.all(18),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  _fromServer ? '서버 분석 · 연구용' : '내장 시연 결과',
                  style: const TextStyle(
                    color: _muted,
                    fontWeight: FontWeight.w800,
                  ),
                ),
                const SizedBox(height: 6),
                _SectionHeading(result.finalPestName),
                const SizedBox(height: 4),
                Text(result.growthStage),
                if (!_fromServer) ...[
                  const SizedBox(height: 8),
                  const Text(
                    '시연 데이터입니다. 현장 진단이나 방제 결정에 사용하지 마세요.',
                    style: TextStyle(color: _muted, height: 1.45),
                  ),
                ],
                const Divider(height: 24),
                Text('YOLO 후보  ${result.yoloPredName} · 탐지 신뢰도 $confidence%'),
                const SizedBox(height: 4),
                Text(
                  '탐지 개체 ${result.detectedCount}개 · YOLO ${result.yoloLatencyMs.toStringAsFixed(1)} ms',
                ),
                if (result.isSelfCorrected) ...[
                  const SizedBox(height: 4),
                  const Text(
                    'VLM이 유사종 후보를 교정했습니다.',
                    style: TextStyle(fontWeight: FontWeight.w700),
                  ),
                ],
              ],
            ),
          ),
        ),
        const SizedBox(height: 10),
        _buildSectionCard(
          Icons.visibility_outlined,
          '형태 판별 근거',
          result.diagnosisSection,
        ),
        _buildSectionCard(
          Icons.warning_amber_rounded,
          '예상 작물 피해',
          result.symptomSection,
        ),
        _buildSectionCard(
          Icons.eco_outlined,
          '방제 참고 정보',
          result.controlSection,
        ),
      ],
    );
  }

  Widget _buildSectionCard(IconData icon, String title, String content) {
    return Card(
      margin: const EdgeInsets.only(bottom: 10),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(icon, color: _green, size: 20),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    title,
                    style: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w800,
                      color: _green,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 10),
            SelectableText(
              content,
              style: const TextStyle(fontSize: 15, height: 1.55),
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
    final boxPaint =
        Paint()
          ..color = const Color(0xFFFF1E1E)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 3.0;

    final fillPaint =
        Paint()
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

      final labelText =
          '${b.label} ${(b.confidence * 100).toStringAsFixed(1)}%';
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
  bool shouldRepaint(covariant VisualBBoxPainter oldDelegate) =>
      oldDelegate.boxes != boxes;
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
    return SingleChildScrollView(
      key: const PageStorageKey('settings'),
      padding: const EdgeInsets.symmetric(vertical: 20),
      child: _Content(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const _SectionHeading('마이페이지 및 설정'),
            const SizedBox(height: 12),
            // 농가 및 프로젝트 프로필 카드
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Row(
                  children: [
                    const CircleAvatar(
                      radius: 28,
                      backgroundColor: Color(0xFF1B5E20),
                      child: Icon(
                        Icons.agriculture_rounded,
                        color: Colors.white,
                        size: 30,
                      ),
                    ),
                    const SizedBox(width: 14),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            'MYSTIC 스마트 영농 연구팀',
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            '참여기업: (주)스톰힐 이노베이션\n지도교수: 최희열 교수님 | 누적 진단: ${widget.historyCount}건',
                            style: TextStyle(
                              fontSize: 12,
                              color: Colors.grey.shade700,
                              height: 1.4,
                            ),
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
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
                color: Color(0xFF1B5E20),
              ),
            ),
            const SizedBox(height: 8),
            const Card(
              child: Column(
                children: [
                  ListTile(
                    leading: Icon(
                      Icons.radar_rounded,
                      color: Color(0xFF1B5E20),
                    ),
                    title: Text(
                      '1단계 객체 탐지: YOLOv8-Large v5',
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 14,
                      ),
                    ),
                    subtitle: Text(
                      '800×800 Letterbox • 유사종 2.0x 오버샘플링 • 현장 데이터 추가 검증 중',
                    ),
                  ),
                  Divider(height: 1),
                  ListTile(
                    leading: Icon(
                      Icons.auto_awesome_rounded,
                      color: Color(0xFF1B5E20),
                    ),
                    title: Text(
                      '2단계 정밀 진단: Qwen2-VL-2B LoRA (Rank 16)',
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 14,
                      ),
                    ),
                    subtitle: Text(
                      '800장 확장 데이터셋 파인튜닝 • 시각적 BBox 가이딩 • 연구용 프로토타입',
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),

            // FastAPI 백엔드 서버 연동 설정
            const Text(
              '추론 백엔드 서버(FastAPI) 연동 설정',
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
                color: Color(0xFF1B5E20),
              ),
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
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          fontSize: 14,
                        ),
                      ),
                      subtitle: const Text(
                        '켜면 촬영·업로드 사진을 아래 서버로 전송합니다. 서버 분석이 실패하면 내장 시연 결과를 표시합니다.',
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
                        labelText: 'FastAPI 서버 주소',
                        hintText: 'http://127.0.0.1:8000',
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(10),
                        ),
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

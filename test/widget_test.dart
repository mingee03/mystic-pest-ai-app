import 'package:flutter/widgets.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mystic_pest_ai_app/main.dart';
import 'package:mystic_pest_ai_app/services/pipeline_api_service.dart';

void main() {
  testWidgets('Phone shows camera actions and bottom navigation', (
    tester,
  ) async {
    tester.view.physicalSize = const Size(360, 780);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    await tester.pumpWidget(const MysticPestAiApp());
    expect(tester.getCenter(find.text('홈')).dy, greaterThan(650));
    expect(find.text('사진 촬영'), findsOneWidget);
    expect(find.text('이미지 업로드'), findsOneWidget);

    await tester.tap(find.text('해충 카메라'));
    await tester.pumpAndSettle();
    expect(find.text('해충 사진 진단'), findsOneWidget);
    expect(find.text('사진을 촬영하거나 이미지를 선택해 주세요.'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  testWidgets('Desktop sample can run again and keeps its result on resize', (
    tester,
  ) async {
    tester.view.physicalSize = const Size(1440, 1000);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    await tester.pumpWidget(const MysticPestAiApp());

    expect(tester.getCenter(find.text('홈')).dy, lessThan(64));
    final pestName =
        PipelineApiService.samplePresets.first.presetResult.finalPestName;
    for (var attempt = 0; attempt < 2; attempt++) {
      final sample = find.text(pestName).first;
      await tester.ensureVisible(sample);
      await tester.tap(sample);
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 550));
      await tester.pump(const Duration(milliseconds: 450));
      await tester.pump(const Duration(milliseconds: 700));
      await tester.pumpAndSettle();

      expect(find.text('내장 시연 결과'), findsOneWidget);
      expect(find.text('시연 탐지 영역 표시'), findsOneWidget);
      if (attempt == 0) {
        await tester.tap(find.text('홈'));
        await tester.pumpAndSettle();
      }
    }

    tester.view.physicalSize = const Size(360, 780);
    await tester.pumpAndSettle();
    expect(tester.getCenter(find.text('홈')).dy, greaterThan(650));
    expect(find.text('내장 시연 결과'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });
}

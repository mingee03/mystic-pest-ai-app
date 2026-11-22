import 'package:flutter_test/flutter_test.dart';
import 'package:mystic_pest_ai_app/main.dart';

void main() {
  testWidgets('MYSTIC App navigation bar test', (WidgetTester tester) async {
    await tester.pumpWidget(const MysticPestAiApp());
    expect(find.text('홈'), findsOneWidget);
    expect(find.text('해충 카메라'), findsOneWidget);
    expect(find.text('마이페이지'), findsOneWidget);
  });
}

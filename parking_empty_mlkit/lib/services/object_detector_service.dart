import 'dart:io';
import 'package:google_mlkit_object_detection/google_mlkit_object_detection.dart';
import 'package:image/image.dart' as img;

class ObjectDetectorService {
  late final ObjectDetector _detector;

  /// 박스 필터링 파라미터
  final double minBoxArea;
  final double maxAspect;
  final double minAspect;

  /// 기타 옵션
  final bool enableClassification;
  final bool enableMultiple;
  final bool streamMode;

  ObjectDetectorService({
    this.minBoxArea = 900.0,
    this.maxAspect = 5.0,
    this.minAspect = 0.15,
    this.enableClassification = false,
    this.enableMultiple = true,
    this.streamMode = false,
  }) {
    final options = ObjectDetectorOptions(
      mode: streamMode ? DetectionMode.stream : DetectionMode.single,
      classifyObjects: enableClassification,
      multipleObjects: enableMultiple,
    );
    _detector = ObjectDetector(options: options);
  }

  /// 파일 경로로부터 감지 박스 반환
  Future<List<({int? id, double left, double top, double right, double bottom})>>
      detectBoxes(String framePath) async {
    final input = InputImage.fromFilePath(framePath);
    final detections = await _detector.processImage(input);

    // 원본 이미지 크기 파악 → 경계 보정(clamp)
    final bytes = await File(framePath).readAsBytes();
    final decoded = img.decodeImage(bytes);
    if (decoded == null) throw Exception('decodeImage failed: $framePath');
    final w = decoded.width.toDouble();
    final h = decoded.height.toDouble();

    final out = <({int? id, double left, double top, double right, double bottom})>[];

    for (final o in detections) {
      final r = o.boundingBox;

      // 필터링(면적/종횡비)
      final bw = r.width;
      final bh = r.height;
      final area = bw * bh;
      if (area < minBoxArea) continue;

      final aspect = (bw > bh ? bw / bh : bh / bw);
      if (aspect > maxAspect || aspect < minAspect) continue;

      // 경계 보정
      final left   = r.left.clamp(0.0, w);
      final top    = r.top.clamp(0.0, h);
      final right  = r.right.clamp(0.0, w);
      final bottom = r.bottom.clamp(0.0, h);

      out.add((id: o.trackingId, left: left, top: top, right: right, bottom: bottom));
    }
    return out;
  }

  Future<void> dispose() async => _detector.close();
}

// lib/main.dart
import 'dart:io';
import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:image/image.dart' as img;

import 'models/slot.dart';
import 'utils/geometry.dart';
import 'services/video_frame_extractor.dart';
import 'services/object_detector_service.dart';
import 'services/parking_analyzer.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const ParkingApp());
}

class ParkingApp extends StatelessWidget {
  const ParkingApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '주차장 빈자리 찾기',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
        useMaterial3: true,
      ),
      home: const HomePage(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final _ctrl = TextEditingController(text: '/sdcard/Download/1.mp4');
  final _logs = <String>[];
  bool _running = false;

  // 설정값 (필요 시 UI로 뺄 수 있음)
  static const int    EXTRACT_FPS      = 3;
  static const double IOU_THRESHOLD    = 0.20;
  static const double MIN_BOX_AREA     = 900.0;
  static const double MIN_ASPECT       = 0.15;
  static const double MAX_ASPECT       = 5.0;
  static const bool   SAVE_OVERLAYS    = false;

  void _log(String s) {
    setState(() { _logs.add(s); });
  }

  Future<void> _askPermissions() async {
    // Android 13+ : videos/photos, 12- : storage
    await Permission.videos.request();
    await Permission.photos.request();
    await Permission.storage.request();
  }

  /// 첫 프레임 해상도 읽기
  Future<({int w, int h})> _readSize(File f) async {
  final bytes = await f.readAsBytes();
  final decoded = img.decodeImage(bytes);
  if (decoded == null) throw Exception('이미지 디코드 실패: ${f.path}');
  return (w: decoded.width, h: decoded.height);
}

  /// ROI 템플릿: 첫 프레임 해상도에 맞춘 픽셀 좌표로 변환해서 사용
  List<Slot> _makeSlotsForSize(int w, int h) {
    // 예시: 사용자가 가진 영상(1280x720)에서 알려준 좌표(픽셀)를 기준으로
    // 해상도가 달라도 **비율**로 스케일링되도록 아래처럼 작성.
    Rect scaleRect(num x1, num y1, num x2, num y2) {
      // 원본 기준(예: 1280x720) 좌표라고 가정하고 비율 변환
      const baseW = 1280.0;
      const baseH = 720.0;
      final sx = w / baseW;
      final sy = h / baseH;
      return rectFromXYXY(x1 * sx, y1 * sy, x2 * sx, y2 * sy);
    }

    // ▼ 사용자가 예시로 준 ROI. (필요 시 원하는 만큼 추가)
    final rois = <Rect>[
      scaleRect(445, 614, 533, 693),
      scaleRect(576, 617, 765, 691),
      scaleRect(717, 617, 995, 692),
      // ... 더 추가 가능
    ];

    final slots = <Slot>[];
    for (int i = 0; i < rois.length; i++) {
      slots.add(Slot(id: i + 1, roi: rois[i]));
    }
    return slots;
  }

  Future<void> _run() async {
    if (_running) return;
    setState(() => _running = true);
    _logs.clear();

    try {
      final videoPath = _ctrl.text.trim();
      if (videoPath.isEmpty) {
        throw Exception('비디오 경로를 입력하세요.');
      }
      _log('권한 요청 중…');
      await _askPermissions();

      _log('프레임 추출 시작… ($EXTRACT_FPS fps)');
      final extractor = VideoFrameExtractor(fps: EXTRACT_FPS);
      final extracted = await extractor.extractFrames(videoPath);
      final frames = extracted.frames;
      _log('프레임 추출 완료 → ${frames.length}장 (${extracted.outDir.path})');

      // ROI 준비 (첫 프레임 크기 기준)
      final size = await _readSize(frames.first);
      _log('첫 프레임 해상도: ${size.w}x${size.h}');
      final slots = _makeSlotsForSize(size.w, size.h);
      _log('ROI 슬롯 수: ${slots.length}');

      // 감지기/분석기 실행
      final detector = ObjectDetectorService(
        minBoxArea: MIN_BOX_AREA,
        minAspect: MIN_ASPECT,
        maxAspect: MAX_ASPECT,
      );
      final analyzer = ParkingAnalyzer(
        detector: detector,
        iouThreshold: IOU_THRESHOLD,
        saveOverlayFrames: SAVE_OVERLAYS,
        maxFrames: 600,
      );

      final summary = await analyzer.run(
        slots: slots,
        frames: frames,
        workDir: extracted.outDir,
        onLog: _log,
      );

      _log(summary);
      await detector.dispose();
      _log('완료 ✅');
    } catch (e, st) {
      _log('에러: $e');
      _log(st.toString());
    } finally {
      setState(() => _running = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final runBtn = FilledButton.icon(
      onPressed: _running ? null : _run,
      icon: const Icon(Icons.play_arrow),
      label: Text(_running ? 'Running…' : 'Run on video'),
    );

    final pathField = TextField(
      controller: _ctrl,
      decoration: const InputDecoration(
        border: OutlineInputBorder(),
        labelText: '비디오 경로 (예: /sdcard/Download/1.mp4)',
      ),
    );

    final logView = Container(
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        border: Border.all(color: Colors.black12),
        borderRadius: BorderRadius.circular(8),
      ),
      height: 260,
      child: ListView.builder(
        itemCount: _logs.length,
        itemBuilder: (_, i) => Text(
          _logs[i],
          style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
        ),
      ),
    );

    return Scaffold(
      appBar: AppBar(title: const Text('주차장 빈자리 찾기')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Align(
              alignment: Alignment.centerLeft,
              child: Text(
                'Google ML Kit 기반 빈자리 탐지',
                style: Theme.of(context).textTheme.titleLarge,
              ),
            ),
            const SizedBox(height: 12),
            pathField,
            const SizedBox(height: 12),
            Row(
              children: [
                runBtn,
                const SizedBox(width: 12),
                Text('FPS=$EXTRACT_FPS, IoU≥$IOU_THRESHOLD, MinArea=$MIN_BOX_AREA'),
              ],
            ),
            const SizedBox(height: 12),
            Expanded(child: logView),
          ],
        ),
      ),
    );
  }
}

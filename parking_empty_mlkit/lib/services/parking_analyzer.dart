import 'dart:io';
import 'dart:math';
import 'dart:ui';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

import '../models/slot.dart';
import '../utils/geometry.dart';
import 'object_detector_service.dart';

class ParkingAnalyzer {
  final ObjectDetectorService detector;
  final double iouThreshold;
  final bool saveOverlayFrames;
  final int maxFrames;

  ParkingAnalyzer({
    required this.detector,
    this.iouThreshold = 0.20,
    this.saveOverlayFrames = false,
    this.maxFrames = 600,
  });

  static final _green = img.ColorRgb8(0, 200, 0);
  static final _red   = img.ColorRgb8(230, 0, 0);
  static final _cyan  = img.ColorRgb8(0, 170, 200);

  Future<String> run({
    required List<Slot> slots,
    required List<File> frames,
    required Directory workDir,
    void Function(String log)? onLog,
  }) async {
    final used = frames.take(maxFrames).toList();

    Directory? overlay;
    if (saveOverlayFrames) {
      overlay = Directory('${workDir.path}/overlays');
      if (!overlay.existsSync()) overlay.createSync(recursive: true);
    }

    int idx = 0;
    for (final f in used) {
      idx++;
      onLog?.call('Processing $idx / ${used.length} : ${f.path.split('/').last}');
      final boxes = await detector.detectBoxes(f.path);

      img.Image? im;
      if (saveOverlayFrames) {
        im = img.decodeImage(await f.readAsBytes());
      }

      for (final s in slots) {
        final rS = s.roi;
        double best = 0.0;
        for (final b in boxes) {
          final rB = rectFromXYXY(b.left, b.top, b.right, b.bottom);
          best = max(best, rectIou(rS, rB));
          if (saveOverlayFrames && im != null) _stroke(im, rB, _cyan);
        }
        final occ = best >= iouThreshold;
        if (occ) s.occupiedCount++; else s.emptyCount++;
        if (saveOverlayFrames && im != null) _stroke(im, rS, occ ? _red : _green, t: 5);
      }

      if (saveOverlayFrames && im != null && overlay != null) {
        final p = '${overlay.path}/${f.uri.pathSegments.last}';
        File(p).writeAsBytesSync(img.encodeJpg(im, quality: 85));
      }
    }

    final buf = StringBuffer('=== Summary ===\n');
    for (final s in slots) {
      buf.writeln('Slot ${s.id}: Occ=${s.occupiedCount} / Emp=${s.emptyCount} '
          '(OccRate=${(s.occRate*100).toStringAsFixed(1)}%)');
    }
    final summary = buf.toString();
    onLog?.call(summary);

    final csv = await _saveCsv(slots);
    onLog?.call('CSV saved → $csv');
    return summary;
  }

  Future<String> _saveCsv(List<Slot> slots) async {
    final ext = await getExternalStorageDirectory();
    final dir = Directory('${ext!.path}/parking_results');
    if (!dir.existsSync()) dir.createSync(recursive: true);
    final out = File('${dir.path}/summary_${DateTime.now().millisecondsSinceEpoch}.csv');

    final head = 'slot_id,x1,y1,x2,y2,occ,emp,occ_rate\n';
    final rows = slots.map((s) =>
      '${s.id},${s.roi.left.toStringAsFixed(1)},${s.roi.top.toStringAsFixed(1)},'
      '${s.roi.right.toStringAsFixed(1)},${s.roi.bottom.toStringAsFixed(1)},'
      '${s.occupiedCount},${s.emptyCount},${s.occRate.toStringAsFixed(4)}'
    ).join('\n');

    out.writeAsStringSync(head + rows);
    return out.path;
  }

  void _stroke(img.Image im, Rect r, img.ColorRgb8 c, {int t = 3}) {
  final x1 = r.left.round(), y1 = r.top.round();
  final x2 = r.right.round(), y2 = r.bottom.round();

  for (int i = 0; i < t; i++) {
    // 상단 가로선
    img.drawLine(im, x1: x1,     y1: y1 + i, x2: x2,     y2: y1 + i, color: c);
    // 하단 가로선
    img.drawLine(im, x1: x1,     y1: y2 - i, x2: x2,     y2: y2 - i, color: c);
    // 좌측 세로선
    img.drawLine(im, x1: x1 + i, y1: y1,     x2: x1 + i, y2: y2,     color: c);
    // 우측 세로선
    img.drawLine(im, x1: x2 - i, y1: y1,     x2: x2 - i, y2: y2,     color: c);
  }
}
}

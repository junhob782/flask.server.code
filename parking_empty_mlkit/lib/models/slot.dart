import 'dart:ui';

class Slot {
  final int id;
  final Rect roi;
  int occupiedCount = 0;
  int emptyCount = 0;

  Slot({required this.id, required this.roi});

  double get occRate {
    final total = occupiedCount + emptyCount;
    if (total == 0) return 0.0;
    return occupiedCount / total;
  }

  Map<String, dynamic> toMap() => {
    'id': id,
    'x1': roi.left,
    'y1': roi.top,
    'x2': roi.right,
    'y2': roi.bottom,
    'occ': occupiedCount,
    'emp': emptyCount,
    'occRate': occRate,
  };
}

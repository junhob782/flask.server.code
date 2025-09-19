import 'dart:ui';

double rectIou(Rect a, Rect b) {
  final inter = a.intersect(b);
  if (inter.isEmpty) return 0.0;
  final interArea = inter.width * inter.height;
  final unionArea = a.width * a.height + b.width * b.height - interArea;
  if (unionArea <= 0) return 0.0;
  return interArea / unionArea;
}

Rect rectFromXYXY(double x1, double y1, double x2, double y2) {
  return Rect.fromLTRB(
    x1 < x2 ? x1 : x2,
    y1 < y2 ? y1 : y2,
    x1 < x2 ? x2 : x1,
    y1 < y2 ? y2 : y1,
  );
}

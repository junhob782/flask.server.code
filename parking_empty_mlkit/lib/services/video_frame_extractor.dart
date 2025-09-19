import 'dart:io';
import 'package:ffmpeg_kit_flutter/ffmpeg_kit.dart';
import 'package:ffmpeg_kit_flutter/return_code.dart';
import 'package:path_provider/path_provider.dart';

class VideoFrameExtractor {
  final int fps;
  VideoFrameExtractor({this.fps = 3});

  Future<({Directory outDir, List<File> frames})> extractFrames(String videoPath) async {
    final tmp = await getTemporaryDirectory();
    final out = Directory('${tmp.path}/frames_${DateTime.now().millisecondsSinceEpoch}');
    if (!out.existsSync()) out.createSync(recursive: true);

    final pattern = '${out.path}/%05d.jpg';
    final cmd = '-y -i "$videoPath" -vf fps=$fps "$pattern"';

    final session = await FFmpegKit.execute(cmd);
    final rc = await session.getReturnCode();
    if (!ReturnCode.isSuccess(rc)) {
      final log = await session.getAllLogsAsString();
      throw Exception('FFmpeg failed: ${rc?.getValue()}\n$log');
    }

    final files = out.listSync().whereType<File>().toList()
      ..sort((a, b) => a.path.compareTo(b.path));
    if (files.isEmpty) throw Exception('No frames extracted');
    return (outDir: out, frames: files);
  }
}

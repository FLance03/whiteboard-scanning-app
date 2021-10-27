import 'dart:math';

import 'package:path/path.dart';
import 'dart:async';
import 'dart:io';

class FileHelpers {
  static Future<List<File>> dirContents(Directory dir) {
    var files = <File>[];
    var completer = Completer<List<File>>();
    var lister = dir.list(recursive: false);
    lister.listen ( 
        (FileSystemEntity file) {
          if (file is File) {
            files.add(file);
          }
        },
        onDone:   () => completer.complete(files)
        );
    return completer.future;
  }
  static String formatDate(DateTime dateTime) {
    List<String> addZeros = ['', '0', '00', '000'];
    List<String> nameParts = [dateTime.year.toString(), dateTime.month.toString(), dateTime.day.toString(), 
                              dateTime.hour.toString(), dateTime.minute.toString(), dateTime.second.toString(),
                              dateTime.millisecond.toString()];
    for (int i=0 ; i < 7 ; i++) {
      nameParts[i] = '${addZeros[4 - nameParts[i].length]}${nameParts[i]}';
    }
    // yyyy-mm-dd-hh-mm-ss-mss
    return '${nameParts[0]}-${nameParts[1].substring(2)}-${nameParts[2].substring(2)}' +
            '${nameParts[3].substring(2)}-${nameParts[4].substring(2)}-${nameParts[5].substring(2)}' +
            '${nameParts[3].substring(1)}';
  }
  static String getFileName(String path) {
    // Remove file extension
    return basename(path).split('.')[0];
  }
  static List<List<int>> getUniqueDates(List<String> fileNames) {
    List<String> mimic = List.from(fileNames)..sort((a, b) => b.compareTo(a));
    print(mimic);
    List<List<int>> uniqueDates = [];
    Map<String, int> currentDate = {
      'year': 0, 'month': 0, 'day': 0
    };
    Map<String, int> infos = {};
    for (int i=0 ; i < mimic.length ; i++) {
      infos = getDateTimesMap(mimic[i]);
      if (currentDate['year'] != infos['year'] || 
          currentDate['month'] != infos['month'] || 
        currentDate['day'] != infos['day']) {
        uniqueDates.add([infos['year'] as int, infos['month'] as int, infos['day'] as int]);
        currentDate['year'] = infos['year'] as int;
        currentDate['month'] = infos['month'] as int;
        currentDate['day'] = infos['day'] as int;
      }
    }
    return uniqueDates;
  }
  static List<File> sortByFileName(List<File> files) {
    List<File> mimic = List.from(files);
    return mimic..sort((a, b) => getFileName(b.path).compareTo(getFileName(a.path)));
  }
  static List<int> getDateTimes(String fileName) {
    List<String> infos = fileName.split('-');
    List<int> retVal = List.filled(7, 0, growable: false);

    for (int i=0 ; i < 7 ; i++) {
      retVal[i] = int.parse(infos[i]);
    }
    return retVal;
  }
  static Map<String, int> getDateTimesMap(String fileName) {
    List<int> infos = getDateTimes(fileName);
    return {
      'year': infos[0],
      'month': infos[1],
      'day': infos[2],
      'hour': infos[3],
      'minute': infos[4],
      'second': infos[5],
      'millisecond': infos[5],
    };
  }
  static List<File> filterFiles({required List<File> files, List<int> filters = const []}) {
    List<File> retVal = [];
    List<int> infos = [];
    int filterLength = min(filters.length, 7);
    bool willAccept;

    for (int i=0 ; i < files.length ; i++) {
      infos = getDateTimes(getFileName(files[i].path));
      willAccept = true;
      for (int j=0 ; j < filterLength && willAccept ; j++) {
        if (infos[j] != filters[j]) {
          willAccept = false;
        }
      }
      if (willAccept) {
        retVal.add(files[i]);
      }
    }
    return retVal;
  }
}
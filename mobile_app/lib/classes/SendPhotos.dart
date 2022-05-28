import 'dart:typed_data';
import 'dart:async';
import 'dart:io';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:http/http.dart' as http;

import '../classes/FileHelpers.dart';
import '../screens/Home.dart';
  
/* 
  Things left to do:

  1. Freeze screen (https://stackoverflow.com/questions/54220859/flutter-disable-touch-on-the-entire-screen).
  2. Fix the sendphotos change screen to make it easier.

*/
class SendPhotos {
  final serverURL = 'http://192.168.0.107:15024';

  void sendFiles(BuildContext context, List<File> selectedPhotos) async{
    OverlayEntry _overlayEntry = this._createOverlayEntry(context);
    Overlay.of(context)?.insert(_overlayEntry);
    // method to send files
    await sendJson(selectedPhotos);
    // not sure how to change screens without copy pasting exact same function from home
    _overlayEntry.remove();
    Navigator.popAndPushNamed(
      context, 
      '/files', 
      arguments: {
        'files': await FileHelpers.dirContents(await FileHelpers.getDirectoryFromFolder('Files')),
      }
    );
  }

  OverlayEntry _createOverlayEntry(BuildContext context) {
    var height = MediaQuery.of(context).size.height;
    var width = MediaQuery.of(context).size.width;
    return OverlayEntry(
      builder: (context) => Positioned(
        top: height / 2 - 50,
        left: width / 4,
        child: Material(
          elevation: 4.0,
          child: Column(
            children: [
              Container(
                height: 100,
                width: width / 2,
                child: Center(child: CircularProgressIndicator()),
              )
            ],
          ),
        ),
      )
    );
  }

  Future sendJson(List<File> selectedPhotos) async {
    List<Uint8List> imageBytes = [];
    List<String> base64String = [];
    // from files to uint8list to base64
    print("here");
    selectedPhotos = FileHelpers.sortByFileName(selectedPhotos);
    selectedPhotos.forEach((photos) => imageBytes.add(photos.readAsBytesSync()));
    imageBytes.forEach((image) => base64String.add(base64Encode(image)));
    print(imageBytes);
    final response = await http.post(
      Uri.parse(serverURL),
      headers: <String, String>{
        'Content-Type': 'application/json; charset=UTF-8',
      },
      body: jsonEncode(<String, List<String>>{
        'img': base64String,
      }),
    );

    // put what to do with response code here
    if (response.statusCode == 201) {
      final decodedBytes = base64Decode(response.body);
      serverIO(decodedBytes);
    } else {
      throw Exception('Failed to receive file from server.');
    }
  }

  void serverIO(var decodedBytes) async{
    // put code here to write a file
    final directory = await getExternalStorageDirectory();
    final filePath = '${directory!.path}/files' ;
    final fileDir = await new Directory(filePath).create();
    DateTime now = DateTime.now();
    print(fileDir);
    // yy-mm-dd-hh-min-sec-ms
    String fileName = FileHelpers.formatDate(now);
    print(fileName);
    final newFile = File('$filePath/$fileName.docx');
    newFile.writeAsBytes(decodedBytes);
  }
}





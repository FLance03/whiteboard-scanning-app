import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:async';
import 'dart:io';

import '../classes/FileHelpers.dart';

/*
kp flutter (future builder, circular progress indicator)
server takes the file, server gives it to python, server waits for python word output
flutter:
- user picks photos and use json and uploads it to server
- after receiving the word file, it saves the word file to its designated folder
- Put a screen that freezes app before downloading

*/

class Home extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final ButtonStyle style = ElevatedButton.styleFrom(
      textStyle: const TextStyle(fontSize: 20),
      minimumSize: Size(200, 40),
    );

    return Scaffold(
      appBar: AppBar(
        title: Text('Notes App'),
      ),
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ElevatedButton(
              style: style,
              onPressed: () => _camera(context),
              child: Text('Take a picture'),
            ),
            const SizedBox(height: 10),
            ElevatedButton(
              style: style,
              onPressed: () => _photos(context),
              child: Text('Existing photos'),
            ),
            const SizedBox(height: 10),
            ElevatedButton(
              style: style,
              onPressed: () => _file(context),
              child: Text('Existing files'),
            ),
            const SizedBox(height: 10),
            ElevatedButton(
              style: style,
              onPressed: () => _photos(context),
              child: Text('Process Photos'),
            ),
          ],
        ),
      ),
    );
  }

  void _camera(BuildContext context) {
    Navigator.pushNamed(context, '/camera');
  }
  void _photos(BuildContext context) async {
    final directory = await getExternalStorageDirectory();
    final imagePath = '${directory!.path}/photos' ;
    final imageDir = await new Directory(imagePath).create();
    Navigator.pushNamed(
      context, 
      '/photos', 
      arguments: {
        'photos': await FileHelpers.dirContents(imageDir),
      }
    );
  }
  void _file(BuildContext context) async {
    final directory = await getExternalStorageDirectory();
    final filePath = '${directory!.path}/Files';
    final fileDir = await new Directory(filePath).create();
    print(await FileHelpers.dirContents(fileDir));
    Navigator.pushNamed(
      context, 
      '/files', 
      arguments: {
        'files': await FileHelpers.dirContents(fileDir),
      }
    );
  }
}
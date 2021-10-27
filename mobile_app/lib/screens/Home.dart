import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:async';
import 'dart:io';

import '../classes/FileHelpers.dart';

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
    print(await FileHelpers.dirContents(imageDir));
    Navigator.pushNamed(
      context, 
      '/photos', 
      arguments: {
        'photos': await FileHelpers.dirContents(imageDir)
      }
    );
  }
}
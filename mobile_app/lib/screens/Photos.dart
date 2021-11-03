import 'package:flutter/material.dart';
import 'package:path/path.dart';
import 'dart:io';

import '../widgets/ListPhotos.dart';
import '../classes/FileHelpers.dart';

class Photos extends StatelessWidget {
  List<File> photos;
  bool toOpen;
  Photos({required List<File> files, required bool toOpen}):
    // Sort by filename in descending order
    this.photos = FileHelpers.sortByFileName(files),
    this.toOpen = toOpen;
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Notes App'),
      ),
      body: ListPhotos(
        photos: this.photos,
        toOpen: this.toOpen,
      )
    );
  }
}
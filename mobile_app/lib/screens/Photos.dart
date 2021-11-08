import 'package:flutter/material.dart';
import 'package:path/path.dart';
import 'dart:io';
import 'dart:convert';
import 'package:http/http.dart' as http;



import '../widgets/ListPhotos.dart';
import '../classes/FileHelpers.dart';

class Photos extends StatelessWidget {
  List<File> photos;
  Photos({required List<File> files}):
    // Sort by filename in descending order
    this.photos = FileHelpers.sortByFileName(files, reverse: true);
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Notes App'),
      ),
      body: ListPhotos(
        photos: this.photos,
      )
    );
  }
}
import 'package:flutter/material.dart';
import 'package:path/path.dart';
import 'package:http/http.dart' as http;
import 'dart:io';
import 'dart:convert';

import '../widgets/ListFiles.dart';
import '../classes/FileHelpers.dart';

class Files extends StatelessWidget {
  List<File> files;
  Files({required List<File> files}):
    // Sort by filename in descending order
    this.files = FileHelpers.sortByFileName(files);
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Notes App'),
      ),
      body: ListFiles( // make a listFiles.dart
        files: this.files,
      )
    );
  }
  factory Files.fromJson(Map<String, dynamic> json){
    return Files(
      files: json['listPhotos'],
    );
  }
}
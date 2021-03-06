import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:open_file/open_file.dart';
import 'package:mobile_app/screens/Files.dart';
import 'dart:io';
import 'dart:convert';

import '../classes/FileHelpers.dart';

class ListFiles extends StatefulWidget {
  List<File> files;
  ListFiles({required List<File> this.files});

  @override
  _ListFilesState createState() => _ListFilesState();
}

class _ListFilesState extends State<ListFiles> {
  @override
  String httpURL = "https://jsonplaceholder.typicode.com/albums";
  Widget build(BuildContext screenContext) {
    List<File> listFiles = this.widget.files;
    List<String> stringFiles = [];
    stringFiles = listFiles.map((file) => FileHelpers.getFileName(file.path)).toList();

    return ListView.builder(
      itemCount: stringFiles.length,
      itemBuilder: (BuildContext context, int i) {
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Container(
              child: GestureDetector(
                onTap: () => {
                  // here
                  OpenFile.open(listFiles[i].path)
                },
                child: Text(stringFiles[i])
              ),
              width: MediaQuery.of(context).size.width,
              color: Colors.grey[200],
              padding: EdgeInsets.only(
                left: 10,
                top: 8,
                bottom: 8,
              ),
            ),
          ],
        );
      }
    );
  }
}
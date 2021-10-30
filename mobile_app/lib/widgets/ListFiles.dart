import 'package:flutter/material.dart';
import 'package:mobile_app/screens/Files.dart';
import 'package:path/path.dart';
import 'package:open_file/open_file.dart';
import 'dart:io';

import '../classes/FileHelpers.dart';

class ListFiles extends StatefulWidget {
  List<File> files;

  ListFiles({required List<File> this.files});

  @override
  _ListFilesState createState() => _ListFilesState();
}

class _ListFilesState extends State<ListFiles> {
  @override
  Widget build(BuildContext screenContext) {
    List<File> listFiles = this.widget.files;
    List<String> stringFiles = [];
    stringFiles = listFiles.map((file) => FileHelpers.getFileName(file.path)).toList();
    print(stringFiles);

    return  ListView.builder(
      itemCount: stringFiles.length,
      itemBuilder: (BuildContext context, int i) {
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Container(
              child: GestureDetector(
                onTap: () => OpenFile.open(listFiles[i].path),
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

  openFile(String filePath) async {
    OpenFile.open(filePath);
  }
}
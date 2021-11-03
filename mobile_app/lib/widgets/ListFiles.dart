import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:open_file/open_file.dart';
import 'package:mobile_app/screens/Files.dart';
import 'dart:io';
import 'dart:convert';

import '../classes/FileHelpers.dart';

class ListFiles extends StatefulWidget {
  List<File> files;
  bool toOpen;

  ListFiles({required List<File> this.files, required bool this.toOpen});

  @override
  _ListFilesState createState() => _ListFilesState();
}

class _ListFilesState extends State<ListFiles> {
  @override
  String httpURL = "https://jsonplaceholder.typicode.com/albums";
  Widget build(BuildContext screenContext) {
    List<File> listFiles = this.widget.files;
    bool toOpen = this.widget.toOpen;
    List<String> stringFiles = [];
    stringFiles = listFiles.map((file) => FileHelpers.getFileName(file.path)).toList();

    return  ListView.builder(
      itemCount: stringFiles.length,
      itemBuilder: (BuildContext context, int i) {
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Container(
              child: GestureDetector(
                onTap: () => {
                  // function to open instead of sending to api
                  if(toOpen){
                    OpenFile.open(listFiles[i].path)
                  } else {
                    // code for sending file instead of
                    /*
                      - sending file
                      - receiving file
                    */
                    sendJSON(listFiles[i])

                  }
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
  Future<Files> sendJSON(File file) async {
    final response = await http.post(
      Uri.parse(httpURL),
      headers: <String, String>{
        'Content-Type': 'application/json; charset=UTF-8',
      },
      body: jsonEncode(<String, File>{
        'file': file,
      }),
    );
    if (response.statusCode == 201) {
      // If the server did return a 201 CREATED response,
      // then parse the JSON.
      return Files.fromJson(jsonDecode(response.body));
    } else {
      // If the server did not return a 201 CREATED response,
      // then throw an exception.
      throw Exception('Failed to create album.');
    }
  }

}
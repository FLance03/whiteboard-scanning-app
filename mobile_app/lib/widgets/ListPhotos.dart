import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:mobile_app/screens/Photos.dart';
import 'package:path/path.dart';
import 'package:http/http.dart' as http;
import 'package:open_file/open_file.dart';
import 'dart:io';
import 'dart:convert';

import '../classes/FileHelpers.dart';

class ListPhotos extends StatefulWidget {
  List<File> photos;
  ListPhotos({required List<File> this.photos});

  @override
  _ListPhotosState createState() => _ListPhotosState();
}

class _ListPhotosState extends State<ListPhotos> {
  int numShown = 15;
  List<File> selectedPhotos = [];
  double picPadding = 0.0;
  int picsPerRow = 4;
  String serverURL = 'http://192.168.0.13:15024';

  @override
  Widget build(BuildContext screenContext) {
    List<List<int>> uniqueDates = FileHelpers.getUniqueDates(
      this.widget.photos.map((photo) => FileHelpers.getFileName(photo.path)).toList()
    );
    print(selectedPhotos);
    return  Column(
      children: [
        Expanded(
          flex: 9,
          child: ListView.builder(
            itemCount: uniqueDates.length,
            itemBuilder: (BuildContext context, int i) {
              // photosBlock holds all photos at a specific Date
              List<Widget> photosBlock = WrapPhotos(screenContext, FileHelpers.filterFiles(
                files: this.widget.photos,
                filters: uniqueDates[i],
              ));
              print(i);
              return Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Container(
                    width: MediaQuery.of(context).size.width,
                    color: Colors.grey[200],
                    padding: EdgeInsets.only(
                      left: 10,
                      top: 2,
                      bottom: 2,
                    ),
                    child: Text(
                      FormatYearMonthDay(
                        year: uniqueDates[i][0],
                        month: uniqueDates[i][1],
                        day: uniqueDates[i][2],
                      ),
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 15,
                      ),
                    ),
                  ),
                  ConstrainedBox(
                    constraints: BoxConstraints(
                      minHeight: 50,
                    ),
                    child: Padding(
                      padding: EdgeInsets.all(picPadding),
                      child: Wrap(
                        spacing: 1,
                        runSpacing: 1,
                        direction: Axis.horizontal,
                        crossAxisAlignment: WrapCrossAlignment.center,
                        children: photosBlock,
                      ),
                    ),
                  ),
                ],
              );
            }
          ),
        ),
        Expanded(
          flex: 1,
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Padding(
                padding: const EdgeInsets.only(left: 15.0),
                child: Text(
                  'Selected: ${this.selectedPhotos.length}',
                  style: TextStyle(
                    color: Colors.grey,
                  ),
                ),
              ),
              Padding(
                padding: const EdgeInsets.only(right: 15.0),
                child: ElevatedButton(
                  onPressed: () {
                    sendJson(selectedPhotos);
                  },
                  child: Text('Submit'),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
  Future sendJson(List<File> selectedPhotos) async {
    List<Uint8List> imageBytes = [];
    List<String> base64String = [];
    // from files to uint8list to base64
    selectedPhotos.forEach((photos) => imageBytes.add(photos.readAsBytesSync()));
    imageBytes.forEach((image) => base64String.add(base64Encode(image)));
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
    print(response);
    if (response.statusCode == 201) {
      // If the server did return a 201 CREATED response,
      // then parse the JSON.
      print(response);
      // return Files.fromJson(jsonDecode(response.body));
    } else {
      // If the server did not return a 201 CREATED response,
      // then throw an exception.
      throw Exception('Failed to create album.');
    }
  }

  // void serverIO(List<File> selectedPhotos) async{
  //   final response = await sendJson(selectedPhotos);
  //   // put code here to write a file
  // }

  String FormatYearMonthDay({required int year, required int month, required int day}) {
    List<String> months = ['', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                    'September', 'October', 'November', 'December'];
    return '${months[month]} $day, $year';
  }
  List<GestureDetector> WrapPhotos(BuildContext context, List<File> photos) {
    double widthPerPic = (MediaQuery.of(context).size.width - 2*picPadding - (picsPerRow - 1)) / picsPerRow;
    List<GestureDetector> images = [];

    for (int i=0 ; i < photos.length ; i++) {
      images.add(
        GestureDetector(
          onTap: () {
            String fileName = FileHelpers.getFileName(photos[i].path);
            if (this.selectedPhotos.indexWhere(
              (photo) => FileHelpers.getFileName(photo.path) == fileName) == -1
            ){
              // Photo selected
              setState(() {
                this.selectedPhotos.add(photos[i]);
              });
            }else {
              // Photo deselected
              setState(() {
                this.selectedPhotos.removeWhere(
                  (photo) => FileHelpers.getFileName(photo.path) == fileName);
              });
            }
          },
          child: Container(
            // Put border if selected (contained in this.selectedPhotos)
            decoration: this.selectedPhotos.indexWhere(
              (photo) => FileHelpers.getFileName(photo.path) == FileHelpers.getFileName(photos[i].path)) != -1
            ? BoxDecoration(
              border: Border.all(
                width: 5.0,
                color: Colors.green[200] as Color,
              ),
              borderRadius: BorderRadius.all(
                  Radius.circular(5.0),
              ),
            )
            : BoxDecoration(),
            child: Image.file(
              photos[i],
              width: widthPerPic,
            ),
          ),
        )
      );
    }
    return images;
  }
}
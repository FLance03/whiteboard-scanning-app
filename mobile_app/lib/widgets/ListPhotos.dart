import 'package:flutter/material.dart';
import 'package:mobile_app/screens/Photos.dart';
import 'package:path/path.dart';
import 'dart:io';

import '../classes/FileHelpers.dart';

class ListPhotos extends StatefulWidget {
  List<File> photos;
  ListPhotos({required List<File> this.photos});

  @override
  _ListPhotosState createState() => _ListPhotosState();
}

class _ListPhotosState extends State<ListPhotos> {
  int numShown = 15;
  double picPadding = 0.0;
  int picsPerRow = 4;

  @override
  Widget build(BuildContext screenContext) {
    List<List<int>> uniqueDates = FileHelpers.getUniqueDates(
      this.widget.photos.map((photo) => FileHelpers.getFileName(photo.path)).toList()
    );
    
    return  ListView.builder(
      itemCount: uniqueDates.length,
      itemBuilder: (BuildContext context, int i) {
        // photosBlock holds all photos at a specific Date
        List<Widget> photosBlock = WrapPhotos(screenContext, FileHelpers.filterFiles(
          files: this.widget.photos,
          filters: uniqueDates[i],
        ));
        // List<Widget> photosBlock = WrapPhotos(this.widget.photos, i);
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
    );
  }
  String FormatYearMonthDay({required int year, required int month, required int day}) {
    List<String> months = ['', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                    'September', 'October', 'November', 'December'];
    return '${months[month]} $day, $year';
  }
  List<Image> WrapPhotos(BuildContext context, List<File> photos) {
    double widthPerPic = (MediaQuery.of(context).size.width - 2*picPadding - (picsPerRow - 1)) / picsPerRow;
    List<Image> images = [];

    for (int i=0 ; i < photos.length ; i++) {
      images.add(
        Image.file(
          photos[i],
          width: widthPerPic,
        )
      );
    }
    // images.add(
    //     Image.file(
    //       photos[i],
    //       height: 100,
    //       width: 100,
    //     )
    //   );
    return images;
  }
}
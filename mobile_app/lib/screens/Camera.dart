import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:camera/camera.dart';
import 'dart:async';
import 'dart:io';

import '../classes/FileHelpers.dart';

class Camera extends StatelessWidget {
  @override
  Widget build(BuildContext context){
    WidgetsFlutterBinding.ensureInitialized();
    return FutureBuilder<List<CameraDescription>>(
      future: availableCameras(),
      builder: (BuildContext context, AsyncSnapshot<List<CameraDescription>> snapshot) {
        if (snapshot.connectionState == ConnectionState.done) {
          final cameras = snapshot.data;
          final firstCamera = cameras!.first;
          return TakePictureScreen(
            camera: firstCamera,
          );
        }else {
          return const Center(child: CircularProgressIndicator());
        }
      },
    );
  }
}
class TakePictureScreen extends StatefulWidget {
  final CameraDescription camera;

  const TakePictureScreen({
    Key? key,
    required this.camera,
  }) : super(key: key);

  @override
  TakePictureScreenState createState() => TakePictureScreenState();
}
class TakePictureScreenState extends State<TakePictureScreen> {
  int numPics = 0;
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;

  @override
  void initState() {
    super.initState();
    _controller = CameraController(
      widget.camera,
      // Define the resolution to use.
      ResolutionPreset.medium,
    );
    // Initialize the controller. Returns a Future.
    _initializeControllerFuture = _controller.initialize();
  }
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Take a picture'),
        actions: [
          this.numPics == 0 ? SizedBox() : IconButton(
            icon: Icon(Icons.file_upload),
            onPressed: () async {
              List<File> sendPhotos = await getCurrentGroupedPhotos();
              print(sendPhotos.map((file) => FileHelpers.getFileName(file.path)));
              // Here
            },
          )
        ]
      ),
      body: FutureBuilder<void>(
        future: _initializeControllerFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.done) {
            _controller.setFlashMode(FlashMode.off);
            return Center(
              child: CameraPreview(_controller),
            );
          } else {
            return const Center(child: CircularProgressIndicator());
          }
        },
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () async {
          try {
            await _initializeControllerFuture;

            final image = await _controller.takePicture();
            print(image.path);
            final directory = await getExternalStorageDirectory();
            final imagePath = '${directory!.path}/photos' ;
            final imageDir = await new Directory(imagePath).create();
            DateTime now = DateTime.now();
            print(imageDir);
            // yy-mm-dd-hh-min-sec-ms
            String fileName = FileHelpers.formatDate(now);
            print(fileName);
            final newFile = File('$imagePath/$fileName.jpg');
            newFile.writeAsBytes(await image.readAsBytes());
            setState(() {
              numPics++;
            });
          } catch (e) {
            print(e);
          }
        },
        child: const Icon(Icons.camera_alt),
      ),
    );
  }
  Future<List<File>> getCurrentGroupedPhotos() async {
    final directory = await getExternalStorageDirectory();
    final imagePath = '${directory!.path}/photos' ;
    final imageDir = await new Directory(imagePath).create();
    List<File> photos = await FileHelpers.dirContents(imageDir);

    photos = FileHelpers.sortByFileName(photos, reverse: true);
    return photos.take(numPics).toList();
  }
}
// Widget to display the picture taken by the user.

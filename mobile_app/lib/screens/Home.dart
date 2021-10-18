import 'package:flutter/material.dart';

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
              onPressed: () {},
              child: Text('Existing photos'),
            ),
          ],
        ),
      ),
    );
  }

  void _camera(BuildContext context) {
    //If Login Authetication returns true
    Navigator.pushNamed(context, '/camera');
  }
}
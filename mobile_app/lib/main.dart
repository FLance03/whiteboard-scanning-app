import 'package:flutter/material.dart';
import 'package:mobile_app/screens/Photos.dart';
import 'package:mobile_app/screens/Files.dart';

import 'screens/screens.dart';

Future main() async{
  runApp(
    MaterialApp(
      debugShowCheckedModeBanner: false,
      onGenerateRoute: _routes(), // default routes is '/' which is Home()
      theme: _theme(),
    )
  );
}

ThemeData _theme() {
  return ThemeData(
    visualDensity: VisualDensity.adaptivePlatformDensity,
  );
}

RouteFactory _routes() {
  return (RouteSettings settings) {
    final dynamic arguments = settings.arguments; //Needed for passing data between screens
    Widget screen;
    switch (settings.name) {
      // case TestRoute:
      //   screen = HomePage();
      //   break;
      case '/':
        screen = Home();
        break;
      case '/camera':
        screen = Camera();
        break;
      case '/photos':
        screen = Photos(
          files: arguments['photos'],
          toOpen: arguments['toOpen']
        );
        break;
      case '/files':
        screen = Files(
          files: arguments['files'],
          toOpen: arguments['toOpen'],
        );
        break;
      default:
        return null;
    }
    return MaterialPageRoute(builder: (BuildContext context) => screen);
  };
}


import 'package:flutter/material.dart';

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
    final arguments = settings.arguments; //Needed for passing data between screens
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
      default:
        return null;
    }
    return MaterialPageRoute(builder: (BuildContext context) => screen);
  };
}


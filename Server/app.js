const express = require('express');
const app = express();
const bodyParser = require('body-parser');
const fs = require('fs');
app.use(bodyParser.json({
    type: (req) => true,
  }))
// app.use(express.json()) 
app.post('/', function(req, res) {
    // var base64Img = req.body.img;
    // console.log(req.body.img);
    fs.writeFileSync('test2.jpg',req.body.img, {encoding: 'base64'});
    res.send('s');
});

app.listen('9000', function(err) {
    if (err) throw err;
})
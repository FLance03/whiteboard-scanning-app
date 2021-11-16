const express = require('express');
const app = express();
const bodyParser = require('body-parser');
const fs = require('fs');
const {spawn} = require('child_process');

app.use(express.json({
    type: (req) => true,
    limit: '50mb',
  }))

// app.use(express.json()) 
app.post('/', async function(req, res)  {
    console.log('Hi');
    var base64Img = req.body.img;
    console.log(base64Img);
    for (var i = 0 ; i < base64Img.length ; i++) {
        fs.writeFileSync(`input${i}.jpg`,req.body.img[i], {encoding: 'base64'});
    }
    const python = spawn('python',["../test.py"]);
    python.stdout.on('data', function(data) {
        console.log(data.toString());
        var output = fs.readFileSync(`${__dirname}/../output.docx`, {encoding: 'base64'});
        // res.download(`${__dirname}/../output.docx`, 'output.docx');
        console.log(output);
        res.status(201);
	console.log('DONE!!');
        res.send(output);
    });
});

app.get('/', function(req, res){
    console.log('get works');
});

app.listen('15024', function(err) {
    if (err) throw err;
    console.log('Listening at port 15024');
})
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
    console.log('Received, Hello');
    var base64Img = req.body.img;
    for (var i = 0 ; i < base64Img.length ; i++) {
        fs.writeFileSync(`./Server/${i}.jpg`,req.body.img[i], {encoding: 'base64'});
    }
    const python = spawn('python',["./main.py"]);
    python.stdout.on('data', function(data) {
        var output = fs.readFileSync(`${__dirname}/output.docx`, {encoding: 'base64'});
        // res.download(`${__dirname}/../output.docx`, 'output.docx');
        res.status(201);
        console.log('Program done');
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
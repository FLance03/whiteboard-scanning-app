# Assuming input is png and txt file

import cv2 as cv
from docx import Document
from docx.shared import Inches

f = open('test.txt', 'r')
textlines = f.readlines()

document = Document()

document.add_picture('images/test.png', width=Inches(1.25))

for line in textlines:
    document.add_paragraph(line)

document.save('test.docx')
# Assuming input is png and txt file

import cv2 as cv
from docx import Document
from docx.shared import Inches

f = open('test.txt', 'r')
textlines = f.read()

document = Document()

table = document.add_table(rows=3, cols=3, style='Table Grid')

rowCellsA = table.rows[0].cells
rowCellsB = table.rows[2].cells
rowCellsA[0].text = textlines
cell = rowCellsA[2]
paragraph = cell.add_paragraph()
cell_r = paragraph.add_run()
cell_r.add_picture('images/test.png')
cellA = table.cell(2,0)
cellB = table.cell(2,2)
MergeCell = cellA.merge(cellB)

document.save('test.docx')
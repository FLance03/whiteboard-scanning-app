# import pandas as pd
#
#
# try:
#     df = pd.read_excel('a.xlsx')
# except FileNotFoundError:
#     pd.DataFrame([], columns=['a', 'b']).to_excel('a.xlsx')
# else:
#     print(len(df.index))
#     df = df.append({'a':0, 'b': 0}, ignore_index=True)
#     print(df)
#     df.to_excel('a.xlsx', index=False)
import cv2 as cv
import pandas as pd
dir_num = 1
data = []
while True:
    test = cv.imread(f'./{dir_num}/0o.jpg')
    print(test)
    if test is None:
        break
    img_num = 0
    while True:
        o = cv.imread(f'./{dir_num}/{img_num}o.jpg')
        if o is None:
            break
        p = cv.imread(f'./{dir_num}/{img_num}p.jpg')
        n = cv.imread(f'./{dir_num}/{img_num}n.jpg')
        data.append({
            'dir_num': dir_num,
            'img_num': img_num,
        })
        img_num += 1
    dir_num += 1
    print(dir_num)
pd.DataFrame(data).to_excel('a.xlsx')

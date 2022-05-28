import re
from os import walk, path, mkdir
import pathlib

import cv2 as cv

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv.resize(image, dim, interpolation=cv.INTER_AREA)

timestamps = []

with open('timestamps.txt', 'r') as f:
    while True:
        timestamp = f.readline()
        if timestamp == '':
            break
        timestamps.append(timestamp)

timeinfo = []
tags = []
for i, timestamp in enumerate(timestamps):
    p = re.match(r'((?P<hr>\d?\d)[^\w\s])?(?P<min>\d?\d)[^\w\s](?P<sec>\d?\d)( (?P<tag>[^\n]*))?(?P<nl>\\n)?', timestamp)
    if p is None and timestamp != '\n':
        raise Exception("Something wrong with line #", i + 1)
    elif timestamp != '\n':
        hr = 0 if p['hr'] is None else int(p['hr'])
        ms = hr * 3600 * 1000 + int(p['min']) * 60 * 1000 + int(p['sec']) * 1000
        if ms in timeinfo:
            check_exists = timeinfo.index(ms)
            raise Exception("Timestamps #" + str(check_exists + 1) + " and " + str(i + 1) + " are the same")
        timeinfo.append(ms)
        tags.append('' if p['tag'] is None else p['tag'])

video_name = ''
video_extension = ''
parent_dir = pathlib.Path(__file__).parent.resolve()
for (_, _, filenames) in walk(parent_dir):
    for filename in filenames:
        p = re.match(r'(?P<name>.*)?\.(?P<extension>\w*)', filename)
        if p is not None and p['extension'] in ['mkv', 'mp4', 'webm']:
            if video_extension != '':
                raise Exception('Multiple files with video extensions in current directory')
            video_name = '' if p['name'] is None else p['name']
            video_extension = p['extension']
new_dir = path.join(parent_dir, video_name)
new_dir_num = 1
while path.exists(new_dir + ' (' +str(new_dir_num) + ')'):
    new_dir_num += 1
new_dir += ' (' +str(new_dir_num) + ')'
mkdir(new_dir)

cap = cv.VideoCapture(video_name + '.' + video_extension)
fps = cap.get(cv.CAP_PROP_FPS)
for index, ms_stamp in enumerate(timeinfo):
    if cap.get(cv.CAP_PROP_FRAME_COUNT)/cap.get(cv.CAP_PROP_FPS)*1000 <= ms_stamp:
        break
    if index != len(timeinfo) - 1:
        next_frame_number = timeinfo[index + 1] / 1000 * cap.get(cv.CAP_PROP_FPS)

    key = ''
    cap.set(cv.CAP_PROP_POS_MSEC, ms_stamp)
    frame_exists, frame = cap.read()
    stack = [frame]
    stack_index = -1
    cv.imshow(str(index), frame if frame.shape[0] < 500 and frame.shape[1] < 700 else cv.resize(frame, (700, 500), interpolation=cv.INTER_AREA))
    while key != ord(' '):
        key = cv.waitKey(50)
        if key == ord('/') or key == ord('.'):
            current_frame_number = cap.get(cv.CAP_PROP_POS_FRAMES)
            if key == ord('/'):
                if index != len(timeinfo) - 1 and next_frame_number <= current_frame_number + len(stack) + 2:
                    print('Going to next frame will already overlap the frame for the next timestamp')
                elif current_frame_number >= int(cap.get(cv.CAP_PROP_FRAME_COUNT)) - 1:
                    print('Already at the last frame')
                elif stack_index < -1:
                    stack_index += 1
                else:
                    cap.set(cv.CAP_PROP_POS_FRAMES, current_frame_number + 1)
                    frame_exists, frame = cap.read()
                    stack.append(frame)
            elif key == ord('.'):
                if stack_index == -len(stack):
                    print('Cannot go back a frame, try to lessen by 1 the second for timestamp #', index + 1)
                else:
                    stack_index -= 1
            cv.imshow(str(index), stack[stack_index] if stack[stack_index].shape[0] < 500 and stack[stack_index].shape[1] < 700 else cv.resize(stack[stack_index], (700, 500), interpolation=cv.INTER_AREA))
    cv.destroyAllWindows()
    cv.imwrite(r'{new_dir}/{pic_num} - ({hr}-{min}-{sec})+{frame_shifts} frames - {tag}.png'.format(
        new_dir=new_dir,
        pic_num=index,
        hr=ms_stamp // 3600000,
        min=ms_stamp % 3600000 // 60000,
        sec=ms_stamp % 3600000 % 60000 // 1000,
        frame_shifts=len(stack) + stack_index,
        tag=tags[index]
    ), stack[stack_index])


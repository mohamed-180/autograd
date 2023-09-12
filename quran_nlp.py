import torch
from re import sub

with open('quran-final.txt') as f:
    s = f.read()

s = s.replace('\n', ' ')

h = sub('[^ةجحخهعغإفقثصضشسيىبلاآتنمكوؤرزأدءذئ طظ]', '', s)


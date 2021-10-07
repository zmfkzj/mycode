from exif import Image
from pathlib import Path
import chardet
import numpy as np

with open(Path.home()/'Downloads/task_210906_남서울_촬영_test-2021_09_27_13_26_35-coco 1.0/images/101_0382_0001.JPG','r+b') as f:
    img = Image(f)
latitude = np.sum(np.array(img.gps_latitude) / np.array((1,60,3600)))
longitude = np.sum(np.array(img.gps_longitude) / np.array((1,60,3600)))
print(latitude, longitude)

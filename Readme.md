A deep learning based opencv face detector.
-------------------------------------------
fast, accurate face detection with OpenCV using a pre-trained deep learning face detector model shipped with the library.

**libraries needed**
1. opencv (> v3.3)
2. numpy
3. imutils
4. argparse
5. time

**additional files needed**
Both files can be found in the opencv Github repo.
1. deploy.prototxt (defines the model architecture/actual layers)
2. caffemodel file (contains the weights for the actual layers)

I'hv included both of these files as well as some example pictures above for convinience.

**To run the face detection files satisfy all conditions above and then execute the following:**
1. open terminal in the directory location
2. for detect_faces file:
   ```python detect_faces.py --image rooster.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel```
3. for detect_faces_video file:
   ```python detect_faces_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel```


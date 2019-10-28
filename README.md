# Face-Classification
Face classification python script on videos

Input: Video
Output: all the faces detected represented as face image + list of frames appear and his coordinations. 

Usage:
```
python facedetector.py -h
python facetector.py -l VideoPath


  -h, --help            show this help message and exit
  -l LINK, --link LINK  provide video link
  -s STEPS, --steps STEPS
                        choose the frame steps (default = 1)
  -r RESIZE, --resize RESIZE
                        choose resize rate (default = 0.500000)
  -e {1,2}, --eyes {1,2}
                        Whether to clarify face with one or two eyes (default
                        = 1)

```

After proccessing the result should be named "result_ID", few examples added.


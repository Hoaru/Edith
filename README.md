Edith is an assistive system that recognize objects or text through point-detecting and give auditory feedback. This name comes from the movie “Spider-Man: Far From Home”, in which iron man Tony Stark sent spider man Peter Parker a smart glasses and Edith is the name of the artificial tactical intelligence system deployed in that glasses.
 
For Edith, we developed 3 modes, Object-Recognition Mode, Text-Recognition Mode and Assist Mode， amiming at helping different groups of people. 

Here is a brief introduction of our work:
1. Developed a smart recognition helper based on Raspberry Pi and PC with 3 modes (object-recognition mode, text-recognition mode and assist mode) that gives auditory feedback to help different groups of people.
2. Applied pyttsx3 for voice synthesis in MP4 format and broadcast according to text information on RPi. Applied MediaPipe and OpenCV for gesture recognition to trigger recognition.
3. Used YOLOv5, PyTorch and PaddleOCR to train an object-recognition model and a text-recognition model with Python, achieved accuracy of 90.1% and 81.4%. 
4. Established communication between RPi and PC for image collection and message feedback through Socket and multithread.

For some files with hierarchical structure, the sizes are over 100MB, which is limited to be upload by github. All of the hierarchical files has been uploaded to Google Drive. (https://drive.google.com/drive/u/3/folders/1kW9ZaXfIemfHjCNuADo9503cac80h5UU)

Thanks for reading.

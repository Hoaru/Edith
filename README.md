Edith is an assistive system that recognize objects or text through point-detecting and give auditory feedback. This name comes from the movie “Spider-Man: Far From Home”, in which iron man Tony Stark sent spider man Peter Parker a smart glasses and Edith is the name of the artificial tactical intelligence system deployed in that glasses.
 
For Edith, we developed 3 modes, Object-Recognition Mode, Text-Recognition Mode and Assist Mode， amiming at helping different groups of people. 

Here is a brief introduction of our work:
1. Developed a smart assist recognition application based on Linux (RPI) and Windows (PC) with 3 modes (object-recognition mode, text-recognition mode and assist mode) that gives auditory feedback.
2. Applied pyttsx3 for voice synthesis in MP4 format and read out according to text information on RPI. 
3. Applied MediaPipe and OpenCV for gesture recognition to trigger recognition.
4. Used YOLOv5, PyTorch and PaddleOCR to train an object recognition model and a character recognition model with Python, achieved accuracy of 94.1% and 97.4%. 
5. Established communication between RPI and PC for image collection and message feedback through Socket and multithreading.


For some files with hierarchical structure, the sizes are over 100MB, which is limited to be upload by github. All of the hierarchical files has been uploaded to Google Drive. (https://drive.google.com/drive/u/3/folders/1kW9ZaXfIemfHjCNuADo9503cac80h5UU)

Thanks for reading.

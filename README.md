Edith is an assistant system that recognize objects or text through point-detecting and give auditory feedback. This name comes from the movie “Spider-Man: Far From Home”, in which iron man Tony Stark sent spider man Peter Parker a smart glasses and Edith is the name of the artificial tactical intelligence system deployed in that glasses.
 
For Edith' users, we developed 3 modes, aiming at helping different groups of people:
1. Object-Recognition Mode: https://youtu.be/Uj8FFHap59U
2. Text-Recognition Mode: https://youtu.be/Uj8FFHap59U
3. Accessibility Mode: https://youtu.be/xdyWIbBVSe0

Here is a brief introduction of our work:
1. Developed a smart assistant system on Linux(RPI) and Windows(PC) with three modes (Object-recognition mode/Text-recognition mode/Assistive mode) that gives real-time auditory feedback.
2. Utilized pyttsx3 for voice synthesis with MP4 format and speak out the live-voice according to text information on RPi.
3. Applied MediaPipe and OpenCV for gesture recognition to wake the recognition.
4. Used YOLOv5, PyTorch and PaddleOCR to train object recognition model and character recognition model with Python, achieved accuracy of 94.1% and 97.4% respectively.
5. Applied Socket and Multithreading to established communication between Edge-devices and PC for image collection, and live-voice response.


For some files with hierarchical structure, the sizes are over 100MB, which is limited to be upload by github. All of the hierarchical files has been uploaded to Google Drive. (https://drive.google.com/drive/u/3/folders/1kW9ZaXfIemfHjCNuADo9503cac80h5UU)

Thanks for reading.

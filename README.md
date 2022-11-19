Edith is an assistive system that recognize obejects or an area of characters and give auditory feedback.

It has 3 modes, Object detection mode, Text detection mode and Auxiliary mode, aiming at helping different.

However, our original intention os to help blind people to know about this world using our system and edge-devices, also help those young children to know about this world, also giving auditory feedback for those people who prefer listening to reading.

Here are more details of our work.

1. Developed a smart recognition helper based on PC and Raspberry Pi with 3 modes (object-detecting mode, text-detecting mode and assist mode) that gives auditory feedback to help different groups of people.
2. Applied pyttsx3 for voice synthesis in MP4 format and broadcast according to text information on RPi. Applied MediaPipe and OpenCV for gesture recognition to trigger recognition.
3. Used YOLOv5, PyTorch and PaddleOCR to train an object recognition model and a character recognition model, achieved accuracy of 90.1% and 81.4%. 
4. Established communication between RPi and PC for image collection and message feedback through Socket and multithread.


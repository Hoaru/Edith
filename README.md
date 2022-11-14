Edith is an assistive system that recognize obejects or an area of characters and give auditory feedback.

It has 3 modes, Object detection mode, Text detection mode and Auxiliary mode, aiming at helping different.

However, our original intention os to help blind people to know about this world using our system and edge-devices, also help those young children to know about this world, also giving auditory feedback for those people who prefer listening to reading.

Here are more details of our work.

1. Developed a smart recognition helper on Jetson Nano that gives auditory feedback to users and sends notice to mobile phones.

2. Applied YOLOv5, PaddleOCR to train object and character recognition models with PyTorch, achieved accuracy of 90.1% and 81.4%. Supervised the training progress with Wandb.

3. Used pyttsx3 on voice synthesis in mp4 format and broadcast according to text information. Used MediaPipe and OpenCV for gesture recognition to control recognition patterns.

4. Designed and built database schema using Google Firebase to persist, manage text information. Utilized IFTTT to receive data from edge device and send notifications to mobile users.

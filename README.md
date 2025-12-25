Edith is an assistant system that recognize objects or text through point-detecting and give auditory feedback. This name comes from the movie “Spider-Man: Far From Home”, in which iron man Tony Stark sent spider man Peter Parker a smart glasses and Edith is the name of the artificial tactical intelligence system deployed in that glasses.
 
For Edith' users, we developed 3 modes, aiming at helping different groups of people:
1. Object Recognition Mode: https://youtu.be/Uj8FFHap59U
2. Text Recognition Mode: https://youtu.be/ui5UoAk-Maw
3. Accessibility Mode: https://youtu.be/xdyWIbBVSe0

Here is a brief introduction of our work:
1. Functional Module Design: Developed object recognition, text recognition, and accessibility modes, optimizing interaction logic for various user groups (children, visually impaired) and supporting multi-scenario applications.
2. Computer Vision Development: Implemented video stream reading using OpenCV and performed 3D hand gesture recognition with MediaPipe.
3. Multi-Model Integration: Fine-tuned YOLOv5 model with the COCO dataset for improved accuracy and integrated PaddleOCR for object and text recognition.
4. Real-time Communication: Designed multi-threading for parallel video processing, gesture parsing, and model inference. Used Socket for communication between Raspberry Pi and PC, creating a voice broadcast mechanism for real-time feedback.


Thanks for reading.

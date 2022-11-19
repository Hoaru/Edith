"""
--- Edith ---
1. Use OpenCV to read video stream from camera on Paspberry Pi
2. Identify 3-dimensional coordinates of key points of both hands
3. Activate circle-drawing when the index fingers stop moving
4. Trigger Recognition: when the circles corresponding to the index
fingers of both hands are fully formed, the recognition begins
5. If there are any characters, OCR begins; otherwise, Object-dection begins
"""

import cv2
import time
import math
import torch
import requests
import numpy as np
import mediapipe as mp
# new import for streaming
import queue
import threading
# new import for voice broadcast
import os
import pyttsx3
import socket
import hashlib
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont


font = cv2.FONT_HERSHEY_SIMPLEX
rtmp_str = 'rtsp://172.20.10.5:9554/webcam'
remote = False  # use remote camera or not
q = queue.Queue()  # store the frames
q_result = queue.Queue()  # store Lable and Description

def engine_init():
    engine_name = pyttsx3.init()  # initialize voice engine
    engine_name.setProperty('rate', 100)  # set voice rate
    engine_name.setProperty('volume', 1)  # set voice volume
    voices = engine_name.getProperty('voices')
    engine_name.setProperty('voice', voices[1].id)  # set first speech synthesizer
    return engine_name

def voice_broadcast():
    ip = "172.20.10.4"
    port = 6969
    ip_port = (ip, port)
    server = socket.socket()
    server.bind(ip_port)
    server.listen(5)
    engine = engine_init()
    print("connection start..\n")
    while True:
        conn, addr = server.accept()
        print("conn:", conn, "\naddr:", addr)
        while True:
            filename = "test.mp3"
            # generate vioce file(in this demo the context is fixed)
            # text = "Hello world, voice broadcast test 001."
            text = q_result.get()
            # engine.say(text)
            engine.save_to_file(text, filename)
            engine.runAndWait()
            engine.stop()

            if os.path.isfile(filename):  # if the file exist
                # 1.send file size
                size = os.stat(filename).st_size
                header = str(size) + ' ' + filename
                conn.send(header.encode("utf-8"))
                print("file size：", size)
                # 2.send file
                conn.recv(1024)  # wait for client's ready signal
                m = hashlib.md5()
                f = open(filename, "rb")
                for line in f:
                    conn.send(line)  # send data
                    m.update(line)
                f.close()
                # 3.send md5 to verify
                md5 = m.hexdigest()
                conn.send(md5.encode("utf-8"))
                print("md5:", md5)
            # sleep 15 second (for this demo)
            # time.sleep(15)
    server.close()

def Receive():
    print("start Reveive")
    cap = cv2.VideoCapture(rtmp_str)
    ret, frame = cap.read()
    q.put(frame)
    while ret:
        ret, frame = cap.read()
        q.put(frame)

# drawing class
class DrawSomeInfo:
    def __init__(self):
        self.hand_num = 0
        # record imformation of left and right hands
        # coordinates
        self.last_finger_cord_x = {'Left': 0, 'Right': 0}
        self.last_finger_cord_y = {'Left': 0, 'Right': 0}
        # circles degree
        self.last_finger_arc_degree = {'Left': 0, 'Right': 0}
        # right hand circle
        self.right_hand_circle_list = []
        # initialize retention time
        now = time.time()
        self.stop_time = {'Left': now, 'Right': now}
        # circles color
        self.handedness_color = {'Left': (90, 208, 10), 'Right': (50, 52, 53)}
        # finger movement range(adjust according to your camera)
        self.float_distance = 10
        # time to trigger
        self.activate_duration = 0.3
        self.last_thumb_img = None
        # OCR text recognition model
        self.pp_ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=False,
                                rec_model_dir='./inference/ch_ppocr_server_v2.0_rec_infer/',
                                cls_model_dir='./inference/ch_ppocr_mobile_v2.0_cls_infer/',
                                det_model_dir='./inference/ch_ppocr_server_v2.0_det_infer/')
        # Yolov5 object recognition model
        self.yolov5_dete = torch.hub.load('./yolov5', 'custom', path='./weights/yolov5n.pt', source='local')
        self.yolov5_dete.conf = 0.4
        # last detection result
        self.last_detect_res = {'detection': None, 'ocr': 'None'}
        # description of last detection result
        self.last_detect_des = {'detection': None, 'None': 'None'}

    # add text
    def cv2AddText(self, img, text, position, textColor=(0, 255, 0), textSize=30):
        # judge type of the image
        if (isinstance(img, np.ndarray)):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # instantiate a object to draw on the gave image
        draw = ImageDraw.Draw(img)
        # form of text
        fontStyle = ImageFont.truetype(
            "./fonts/FederationBold.TTF", textSize, encoding="utf-8")
        # draw text
        draw.text(position, text, textColor, font=fontStyle)
        # converse the type back to openCV
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    # generate text area on top right-hand corner
    def generateOcrTextArea(self, ocr_text, line_text_num, line_num, x, y, w, h, frame):
        # first we crop the sub-rect from the image
        sub_img = frame[y:y + h, x:x + w]
        green_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0
        res = cv2.addWeighted(sub_img, 0.5, green_rect, 0.5, 1.0)
        for i in range(line_num):
            text = ocr_text[(i * line_text_num):(i + 1) * line_text_num]
            res = self.cv2AddText(res, text, (10, 30 * i + 10), textColor=(242, 242, 230), textSize=18)
        return res

    # generate label area
    def generateLabelArea(self, text, x, y, w, h, frame):
        # crop the sub-rect from the image
        sub_img = frame[y:y + h, x:x + w]
        green_rect = np.ones(sub_img.shape, dtype = np.uint8) * 0
        res = cv2.addWeighted(sub_img, 0.5, green_rect, 0.5, 1.0)
        if res is not None:
            res = self.cv2AddText(res, text, (10, 10), textColor = (242, 242, 230), textSize = 25)
        else:
            return 0
        return res

    # generate thumbnail on top right-hand corner
    def generateThumb(self, raw_img, frame):
        # object recognition
        if self.last_detect_res['detection'] == None:
            results = self.yolov5_dete(raw_img)
            # nothing detected
            if len(results) == 0:
                print("nothing detected")
            if len(results) > 0:
                results_np = results.pandas().xyxy[0].to_numpy()
                if len(results_np) > 0:
                    label_id = results_np[0][5]
                    label_name = results_np[0][6]
                    self.last_detect_res['detection'] = [label_id, label_name]
                    # wiki search
                    keyword = self.last_detect_res['detection'][1]
                    url = "https://www.wikidata.org/w/api.php"
                    params = {
                        'action': 'wbsearchentities',
                        'format': 'json',
                        'search': keyword,
                        'language': 'en',
                        'type': 'item',
                        'limit': 1  # max num
                    }
                    # visit
                    get = requests.get(url=url, params=params)
                    # turn to json data
                    re_json = get.json()
                    self.last_detect_des['detection'] = re_json["search"][0]["display"]["description"]["value"]
                    lable = self.last_detect_res['detection'][1]
                    intro = self.last_detect_des['detection']
                    # print("Label Name: ", lable)
                    # print("Description: ", intro)
                    q_result.put(lable + ' ' + intro)
                else:
                    self.last_detect_res['detection'] = [0, 'None']
                    self.last_detect_des['detection'] = ['None', 'None']
            else:
                self.last_detect_res['detection'] = [0, 'None']
                self.last_detect_des['detection'] = ['None', 'None']

        # complete image
        frame_height, frame_width, _ = frame.shape
        # cover
        raw_img_h, raw_img_w, _ = raw_img.shape
        thumb_img_w = 200
        thumb_img_h = math.ceil(raw_img_h * thumb_img_w / raw_img_w)
        thumb_img = cv2.resize(raw_img, (thumb_img_w, thumb_img_h))
        rect_weight = 4
        # draw square frame on thumbnail
        thumb_img = cv2.rectangle(thumb_img, (0, 0), (thumb_img_w, thumb_img_h), (0, 139, 247), rect_weight)
        # generate label
        x, y, w, h = (frame_width - thumb_img_w), thumb_img_h, thumb_img_w, 50
        # put the image back to its position
        text = self.last_detect_res['detection'][1]
        frame = frame.copy()
        frame[y:y + h, x:x + w] = self.generateLabelArea(text, x, y, w, h, frame)

        # OCR
        ocr_text = ''
        if self.last_detect_res['ocr'] == 'None':
            result = self.pp_ocr.ocr(raw_img, cls=True)
            src_im = raw_img
            text_list = [line[1][0] for line in result]
            thumb_img = cv2.resize(src_im, (thumb_img_w, thumb_img_h))
            if len(text_list) > 0:
                ocr_text = ''.join(text_list)
                # record detected text
                self.last_detect_res['ocr'] = ocr_text
            else:
                # nothing detected
                self.last_detect_res['ocr'] = 'checked_no'
        else:
            ocr_text = self.last_detect_res['ocr']
        frame[0:thumb_img_h, (frame_width - thumb_img_w):frame_width, :] = thumb_img
        # whether required display
        if ocr_text != '' and ocr_text != 'checked_no':
            line_text_num = 15
            line_num = math.ceil(len(ocr_text) / line_text_num)
            y, h = (y + h + 20), (32 * line_num)
            frame[y:y + h, x:x + w] = self.generateOcrTextArea(ocr_text, line_text_num, line_num, x, y, w, h, frame)
        self.last_thumb_img = thumb_img
        return frame

    # draw circles
    def drawArc(self, frame, point_x, point_y, arc_radius=150, end=360, color=(255, 0, 255), width=20):
        img = Image.fromarray(frame)
        shape = [(point_x - arc_radius, point_y - arc_radius),
                 (point_x + arc_radius, point_y + arc_radius)]
        img1 = ImageDraw.Draw(img)
        img1.arc(shape, start=0, end=end, fill=color, width=width)
        frame = np.asarray(img)
        return frame

    # clear current mode
    def clear(self):
        self.right_hand_circle_list = []
        self.last_finger_arc_degree = {'Left': 0, 'Right': 0}

    # check whether the retention period is longer than 0.3s. Start to draw independently is time is over 0.3s
    def checkIndexFingerMove(self, handedness, finger_cord, frame, frame_copy):
        # calculate distance
        x_distance = abs(finger_cord[0] - self.last_finger_cord_x[handedness])
        y_distance = abs(finger_cord[1] - self.last_finger_cord_y[handedness])
        # lock mode
        # if not moving
        if (x_distance <= self.float_distance) and (y_distance <= self.float_distance):
            # retention time longer than time of trigger
            if (time.time() - self.stop_time[handedness]) > self.activate_duration:
                # draw circles, increase by 5 degrees/0.01s
                arc_degree = 5 * ((time.time() - self.stop_time[handedness] - self.activate_duration) // 0.01)
                if arc_degree <= 360:
                    frame = self.drawArc(
                        frame, finger_cord[0], finger_cord[1], arc_radius=50, end=arc_degree,
                        color=self.handedness_color[handedness], width=15)
                else:
                    frame = self.drawArc(
                        frame, finger_cord[0], finger_cord[1], arc_radius=50, end=360,
                        color=self.handedness_color[handedness], width=15)
                    # make the degrees be 360
                    self.last_finger_arc_degree[handedness] = 360
                    # recognition starts when 2 circles are formed
                    if (self.last_finger_arc_degree['Left'] >= 360) and (
                            self.last_finger_arc_degree['Right'] >= 360):
                        # get the coordinates
                        if self.last_finger_cord_x['Right'] < self.last_finger_cord_x['Left']:
                            rect_l = (self.last_finger_cord_x['Right'], self.last_finger_cord_y['Left'])
                            rect_r = (self.last_finger_cord_x['Right'], self.last_finger_cord_y['Left'])
                        else:
                            rect_l = (self.last_finger_cord_x['Left'], self.last_finger_cord_y['Right'])
                            rect_r = (self.last_finger_cord_x['Left'], self.last_finger_cord_y['Right'])
                        # generate square frame of the recognition area
                        frame = cv2.rectangle(frame, rect_l, rect_r, (0, 255, 0), 2)
                        # generate label for square frame of the recognition area
                        if self.last_detect_res['detection']:
                            x, y, w, h = self.last_finger_cord_x['Left'], (
                                    self.last_finger_cord_y['Left'] - 50), 200, 50
                            frame = frame.copy()
                            text = self.last_detect_res['detection'][1]
                            frame[y:y + h, x:x + w] = self.generateLabelArea(text, x, y, w, h, frame)
                        # initialize recognition results
                        self.last_detect_res = {'detection': None, 'ocr': 'None'}
                        # transmit results to thumbnail
                        y1 = min(self.last_finger_cord_y['Right'], self.last_finger_cord_y['Left'])
                        y2 = max(self.last_finger_cord_y['Right'], self.last_finger_cord_y['Left'])
                        x1 = min(self.last_finger_cord_x['Right'], self.last_finger_cord_x['Left'])
                        x2 = max(self.last_finger_cord_x['Right'], self.last_finger_cord_x['Left'])
                        raw_img = frame_copy[
                                  y1:y2,
                                  x1:x2, ]
                        frame = self.generateThumb(raw_img, frame)

                    # if (self.hand_num == 1) and (self.last_finger_arc_degree['Right'] == 360):
                    #     self.right_hand_circle_list.append((finger_cord[0], finger_cord[1]))
        else:
            # restart clock when palms start moving dramatically
            self.stop_time[handedness] = time.time()
            self.last_finger_arc_degree[handedness] = 0

        # refresh position
        self.last_finger_cord_x[handedness] = finger_cord[0]
        self.last_finger_cord_y[handedness] = finger_cord[1]
        return frame


# recognition-controlling class
class VirtualFingerReader:
    def __init__(self):
        # initialize mediapipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.image = None

    def process(self):
        th1 = threading.Thread(target=VirtualFingerReader.recognize, args=(self,))
        th1.start()
        th1.join()

    # check the index of left and right hands in the array
    # NOTICE: Mediapipe use mirror image
    def checkHandsIndex(self, handedness):
        # judge number of palms
        if len(handedness) == 1:
            handedness_list = ['Left' if handedness[0].classification[0].label == 'Right' else 'Right']
        else:
            handedness_list = [handedness[1].classification[0].label, handedness[0].classification[0].label]
        return handedness_list

    # main function
    def recognize(self):
        # initialize drawing class
        drawInfo = DrawSomeInfo()
        # calculate FPS
        fpsTime = time.time()
        # use OpenCV to read video stream
        if remote == True:
            print("Using web camera")
            cap = cv2.VideoCapture('rtsp://172.20.10.5:9554/webcam')
        else:
            print("Using local camera")
            cap = cv2.VideoCapture(0)
        # resolution ratio
        resize_w = 960
        resize_h = 720

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = 18
        videoWriter = cv2.VideoWriter('./record_video/out' + str(time.time()) + '.mp4', cv2.VideoWriter_fourcc(*'H264'),
                                      fps, (resize_w, resize_h))

        with self.mp_hands.Hands(min_detection_confidence=0.7,
                                 min_tracking_confidence=0.5,
                                 max_num_hands=2) as hands:
            flag_empty = False
            while cap.isOpened():
                if remote == True:
                    if q.qsize() > 5:
                        flag_empty = False
                    if flag_empty != True:
                        # empty the queue when the init is done
                        print("empty")
                        while q.qsize() > 1:
                            q.get()
                        flag_empty = True
                    if q.empty() != True:
                        self.image = q.get()
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    success, self.image = cap.read()
                    if not success:
                        print("Blank Frame")
                        continue
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                # resize image
                self.image = cv2.resize(self.image, (resize_w, resize_h))
                # adjust according to camera position
                # self.image = cv2.rotate( self.image, cv2.ROTATE_180)
                # improve performance
                self.image.flags.writeable = False
                # BGR to RGB
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                # mirror image(adjust according to the camera)
                # self.image = cv2.flip(self.image, 1)
                # process image with mediapipe
                results = hands.process(self.image)
                self.image.flags.writeable = True
                self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
                # save thumbnail
                if isinstance(drawInfo.last_thumb_img, np.ndarray):
                    self.image = drawInfo.generateThumb(drawInfo.last_thumb_img, self.image)
                hand_num = 0
                # whether palms exists
                if results.multi_hand_landmarks:
                    # record the index of hands
                    handedness_list = self.checkHandsIndex(results.multi_handedness)
                    hand_num = len(handedness_list)
                    drawInfo.hand_num = hand_num
                    # copy the original clean frame
                    frame_copy = self.image.copy()
                    # traverse all the palms
                    for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        # fault-tolerance
                        if hand_index > 1:
                            hand_index = 1
                        # draw finger skeleton
                        self.mp_drawing.draw_landmarks(
                            self.image,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style())
                        # used for store finger coordiates
                        landmark_list = []
                        # coordinates used for store the moving range
                        paw_x_list = []
                        paw_y_list = []
                        for landmark_id, finger_axis in enumerate(
                                hand_landmarks.landmark):
                            landmark_list.append([
                                landmark_id, finger_axis.x, finger_axis.y,
                                finger_axis.z
                            ])
                            paw_x_list.append(finger_axis.x)
                            paw_y_list.append(finger_axis.y)
                        if landmark_list:
                            # ratio resize
                            ratio_x_to_pixel = lambda x: math.ceil(x * resize_w)
                            ratio_y_to_pixel = lambda y: math.ceil(y * resize_h)
                            # top-left and bottom-right coordinates of palms
                            paw_left_top_x, paw_right_bottom_x = map(ratio_x_to_pixel,
                                                                     [min(paw_x_list), max(paw_x_list)])
                            paw_left_top_y, paw_right_bottom_y = map(ratio_y_to_pixel,
                                                                     [min(paw_y_list), max(paw_y_list)])
                            # get the coordinates of index finger
                            index_finger_tip = landmark_list[8]
                            index_finger_tip_x = ratio_x_to_pixel(index_finger_tip[1])
                            index_finger_tip_y = ratio_y_to_pixel(index_finger_tip[2])

                            # get the coordinates of middle finger
                            middle_finger_tip = landmark_list[12]
                            middle_finger_tip_x = ratio_x_to_pixel(middle_finger_tip[1])
                            middle_finger_tip_y = ratio_y_to_pixel(middle_finger_tip[2])

                            # draw coordinates
                            label_height = 30
                            label_wdith = 130
                            cv2.rectangle(self.image, (paw_left_top_x - 30, paw_left_top_y - label_height - 30),
                                          (paw_left_top_x + label_wdith, paw_left_top_y - 30), (0, 139, 247), -1)
                            l_r_hand_text = handedness_list[hand_index][:1]
                            cv2.putText(self.image,
                                        "{hand} x:{x} y:{y}".format(
                                            hand = l_r_hand_text,
                                            x = index_finger_tip_x,
                                            y = index_finger_tip_y),
                                        (paw_left_top_x - 30 + 10, paw_left_top_y - 40),
                                        cv2.FONT_HERSHEY_PLAIN, 1, (242, 242, 230), 2)
                            # draw square frame for palms
                            cv2.rectangle(self.image, (paw_left_top_x - 30, paw_left_top_y - 30),
                                          (paw_right_bottom_x + 30, paw_right_bottom_y + 30), (0, 139, 247), 1)
                            # release current mode
                            line_len = math.hypot((index_finger_tip_x - middle_finger_tip_x),
                                                  (index_finger_tip_y - middle_finger_tip_y))
                            if line_len < 50 and handedness_list[hand_index] == 'Right':
                                drawInfo.clear()
                                drawInfo.last_thumb_img = None
                            # transfer image to drawing class. start to draw circles when retention rime is longer than 0.3s
                            self.image = drawInfo.checkIndexFingerMove(handedness_list[hand_index],
                                                                       [index_finger_tip_x, index_finger_tip_y],
                                                                       self.image, frame_copy)
                # display FPS and palms number
                cTime = time.time()
                fps_text = 1 / (cTime - fpsTime)
                fpsTime = cTime
                self.image = drawInfo.cv2AddText(self.image, "FPS: " + str(int(fps_text)),
                                                 (10, 10), textColor = (242, 242, 230), textSize = 35)
                self.image = drawInfo.cv2AddText(self.image, "Palm num: " + str(hand_num),
                                                 (10, 45), textColor = (242, 242, 230), textSize = 35)
                # display image
                # self.image = cv2.resize(self.image, (resize_w//2, resize_h//2))
                cv2.imshow('EDITH', self.image)
                videoWriter.write(self.image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            cap.release()

def test():
    while True:
        text = input("input:\t")
        q_result.put(text)

if __name__ == '__main__':
    control = VirtualFingerReader()
    p1 = threading.Thread(target=Receive)
    p2 = threading.Thread(target=voice_broadcast)
    p3 = threading.Thread(target=test)
    if remote == True:
        p1.start()
        p2.start()
        # p3.start()
        control.process()
    else:
        # p2.start()
        control.recognize()

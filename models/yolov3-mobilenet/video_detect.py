# -*- coding: utf-8 -*-#
# Author:       weiz
# Date:         2019/10/14 17:44
# Name:         video_detect
# Description:  使用yolov3提供的接口检测视频的目标

import cv2
import numpy as np
import copy
from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer
import threading

MAXTRACK = 50  # 最长显示的追踪的点
IOUTHRESHOLD = 0.4  # 检测和追踪框的IOU阈值

loc = []  # 位置信息 x, y, w, h
scores = []  # 置信度
lab = []  # 目标的标签
initImg = []  # 跟踪算法的初始化图片
kcf_trcaker = []
track = {}
tracker = cv2.MultiTracker_create()


def draw(yolo, lab, boxes, scores, img, txt, isDetect, isDebug=False, drawTrack=False):
    """
    用于画框
    :param yolo:
    :param lab:
    :param boxes:
    :param scores:
    :param img:
    :param txt:
    :param isDetect:
    :param isDebug:是否处于调试模式
    :return:
    """
    for i, newbox in enumerate(boxes):
        color = yolo.getColor(lab[i])  # [int(c) for c in COLORS[i]]
        x, y, w, h = 0, 0, 0, 0
        if isDetect:
            x, y, w, h = int(newbox[0]), int(newbox[1]), int(newbox[2] - newbox[0]), int(newbox[3] - newbox[1])
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)

        else:
            x, y, w, h = int(newbox[0]), int(newbox[1]), int(newbox[2]), int(newbox[3])
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
        if track is not None and len(track) > 0:
            if i >= len(kcf_trcaker):
                bug = 0
            points = track[kcf_trcaker[i]]
            if w > 0 and h > 0:
                point = [int(x + w / 2), int(y + h)]
                points.append(point)
            l = len(points)
            length = min(l, MAXTRACK)  # 轨迹点最多显示近MAXTRACK个
            points = points[(l - length):]

            line = np.array(points, np.int32).reshape((-1, 1, 2))
            for p in points:
                if p[0] == 0 or p[1] == 0:
                    bug = 0
            if drawTrack:
                cv2.polylines(img, [line], False, color, thickness=2, lineType=cv2.LINE_AA)
            # for p in points:
            #   cv2.circle(img, p, 3, color, -1)
        if isDebug == True:
            text = ' {} {}: {:.3f}'.format(txt, yolo.getClass_names(lab[i]), scores[i])
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 3)
            cv2.rectangle(img, (x, y - text_h - baseline), (x + text_w, y), color, -1)
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return img


class TrackerThread(threading.Thread):
    """
    用于执行追踪算法
    """

    def __init__(self, yolo, initImg, img, loc, scores, lab, frame, isDraw):
        threading.Thread.__init__(self)

        self.yolo = yolo
        self.initImg = initImg
        self.img = img
        self.loc = loc
        self.scores = scores
        self.lab = lab
        self.frame = frame
        self.isDraw = isDraw

    def run(self):
        self.result = tracking(self.yolo, self.initImg, self.img, self.loc, self.scores, self.lab, self.frame,
                               self.isDraw)

    def get_result(self):
        return self.result


def detect_camera3(yolo, videoPath=None, loop=None, drop=3, output_path="", trackingAlt=True, isDebug=False, drawTrack=False):
    """
    检测本地视频目标或者检测开启摄像头后视频的目标
    :param videoPath: 视频的本地地址；默认或者为0表示开启摄像头
    :param loop: 每隔（loop-1）帧做一次检测
    :param output_path: 是否保存地址
    :param trackingAlt: 是否使用跟踪算法
    :return:
    """
    if (videoPath == None) or (videoPath == 0):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("The camera is not opening!")
            return
    else:
        cap = cv2.VideoCapture(videoPath)
        if not cap.isOpened():
            print("Invalid address entered!")
            return
    if loop == None:
        loop = 5
    if loop < 1:
        print("Please enter an integer greater than one.loop:%f", loop)
    video_FourCC = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')  ##int(cap.get(cv2.CAP_PROP_FOURCC))#c vp09 avc1 hvc1 #
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print('the video video_FourCC = {},video_fps= {},  video_size = {}  output = {}'.format(video_FourCC, video_fps,
                                                                                            video_size, output_path))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
        # 目标的标签
    frame = 0  # 帧数，循环控制参数
    initImg = []  # 跟踪算法的初始化图片
    start1_time = timer()
    global loc
    global scores
    global lab
    while True:

        pro_time = timer()
        return_value, img = cap.read()
        if img is None:
            break
        img_array = Image.fromarray(img)
        ''''
        if frame2 % drop == 0:
            frame2 = frame2 + 1
            frame2 = frame2 % drop
            continue
        else:
            frame2 = frame2 + 1
            frame2 = frame2 % drop
        '''''

        if frame % loop == 0:
            if isDebug == True:
                print('start  detect ')
            initImg = copy.deepcopy(img)
            trackerThread = TrackerThread(yolo, initImg, img, loc, scores, lab, frame, isDraw=False)
            # 启动一个线程,并发去做追踪
            trackerThread.start()
            # 如果是全局第一针
            newLoc, newScores, newLab = yolo.detect_image_value(img_array)
            ''''
            # write here
            head = []
            body = []
            belt = []
            for i in range(len(newLab)):
                clsInfo = []
                clsInfo.append(newLab[i])
                clsInfo.append(newScores[i])
                clsInfo.append(newLoc[i])
                # 0,1 is the index of 'hat' 'nohat'
                if newLab[i]==0 or newLab[i]==1:
                    head.append(clsInfo)
                # 0,1 is the index of 'hat' 'nohat'
                elif newLab[i]==4:
                    body.append(clsInfo)

            for cls in newLab:
                # 0,1 is the index of 'hat' 'nohat'
                if cls in [0, 1]:
            '''


            trackerThread.join()
            ret = trackerThread.get_result()
            if ret is not None:
                loc, scores, lab = ret[0], ret[1], ret[2]

            # 对比检测和追踪画面，更新tracker信息
            updateTracker(newLab, newLoc, newScores, initImg)
            draw(yolo, newLab, newLoc, newScores, img, '', True, isDebug=isDebug, drawTrack=drawTrack)
            result = img
            start = False
        else:
            if trackingAlt:
                ret = tracking(yolo, initImg, img, loc, scores, lab, frame, isDebug=isDebug)  # 目标跟踪
                if ret is not None:
                    loc, scores, lab = ret[0], ret[1], ret[2]
                    result = img
                else:
                    if isDebug == True:
                        print('start  detect ')
                    loc, scores, lab = yolo.detect_image_value(img_array)
                    draw(yolo, lab, loc, scores, img, '', True, isDebug=isDebug, drawTrack=drawTrack)
                    result = img

            else:
                if isDebug == True:
                    print('start  detect ')
                loc, scores, lab = yolo.detect_image_value(img_array)
                draw(yolo, lab, loc, scores, img, '', True, isDebug=isDebug, drawTrack=drawTrack)
                result = img

        post_time = timer()
        exec_time = post_time - pro_time
        curr_fps = 1 / exec_time
        fps = '{}: {:.3f}'.format('FPS', curr_fps)
        if isDebug == True:

            if frame % loop == 0:
                print('detect time = {}  '.format(exec_time))
            else:
                if trackingAlt:
                    print('tracker time = {}  '.format(exec_time))
                else:
                    print('detect time = {}  '.format(exec_time))

        # (fps_w, fps_h), baseline = cv2.getTextSize(fps, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        # cv2.rectangle(img, (2, 20 - fps_h - baseline), (2 + fps_w, 18), color=(0, 0, 0), thickness=-1)

        # cv2.putText(img, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #           fontScale=0.5, color=(255, 255, 255), thickness=2)
        if isOutput:
            out.write(result)
        if isDebug == True:
            cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 0, 0), thickness=2)

            cv2.imshow("result", result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # quit on ESC button
            if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
                break
        frame = frame + 1
        frame = frame % loop
        start = True
    end1_time = timer()
    print('end time is ', (end1_time - start1_time))
    cap.release()
    if isOutput:
        out.release()
    # yolo.close_session()


def detect_camera2(yolo, videoPath=None, loop=None, output_path="", trackingAlt=True):
    """
    检测本地视频目标或者检测开启摄像头后视频的目标
    :param videoPath: 视频的本地地址；默认或者为0表示开启摄像头
    :param loop: 每隔（loop-1）帧做一次检测
    :param output_path: 是否保存地址
    :param trackingAlt: 是否使用跟踪算法
    :return:
    """
    if (videoPath == None) or (videoPath == 0):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("The camera is not opening!")
            return
    else:
        cap = cv2.VideoCapture(videoPath)
        if not cap.isOpened():
            print("Invalid address entered!")
            return
    if loop == None:
        loop = 5
    if loop < 1:
        print("Please enter an integer greater than one.loop:%f", loop)
    video_FourCC = int(cap.get(cv2.CAP_PROP_FOURCC))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
        # 目标的标签
    frame = 0  # 帧数，循环控制参数
    initImg = []  # 跟踪算法的初始化图片
    start = False
    start1_time = timer()
    while True:

        pro_time = timer()
        return_value, img = cap.read()
        if img is None:
            break
        img_array = Image.fromarray(img)

        if frame % loop == 0:
            # loc, scores, lab = yolov3.yolo_detect(img)      # 目标检测
            print('start  detect ')
            initImg = copy.deepcopy(img)
            loc, scores, lab = yolo.detect_image_value(img_array)
            draw(yolo, lab, loc, scores, img, '', True)
            result = img


        else:
            if trackingAlt:

                ret = tracking(yolo, initImg, img, loc, scores, lab, frame)  # 目标跟踪
                if ret is not None:
                    loc, scores, lab = ret[0], ret[1], ret[2]
                    result = img
                else:
                    print('start  detect ')
                    loc, scores, lab = yolo.detect_image_value(img_array)
                    draw(yolo, lab, loc, scores, img, '', True)
                    result = img

            else:
                # loc, scores, lab = yolov3.yolo_detect(img)
                print('start  detect ')
                # loc, scores, lab = yolo.detect_image_value(img_array)
                # draw(yolo, lab, loc, scores, img, '',True)
                tpmLoc, tmpScores, tmpLab = yolo.detect_image_value(img_array)

                result = img
                # r_image, loc, scores, lab = yolo.predict(img_array)
                # result = np.asarray(r_image)

        post_time = timer()
        exec_time = post_time - pro_time
        curr_fps = 1 / exec_time
        fps = '{}: {:.3f}'.format('FPS', curr_fps)
        if frame % loop == 0:
            print('detect time = {}  '.format(exec_time))
        else:
            if trackingAlt:
                print('tracker time = {}  '.format(exec_time))
            else:
                print('detect time = {}  '.format(exec_time))

        # (fps_w, fps_h), baseline = cv2.getTextSize(fps, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        # cv2.rectangle(img, (2, 20 - fps_h - baseline), (2 + fps_w, 18), color=(0, 0, 0), thickness=-1)

        # cv2.putText(img, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #           fontScale=0.5, color=(255, 255, 255), thickness=2)

        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)

        cv2.imshow("result", result)

        if isOutput:
            writeTime = timer()
            out.write(result)
            writeTime2 = timer()
            print('write time = {}  '.format((writeTime2 - writeTime)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # quit on ESC button
        if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            break
        frame = frame + 1
        frame = frame % loop
        start = True
    end1_time = timer()
    print('end time is ', (end1_time - start1_time))
    yolo.close_session()


def detect_camera(yolo, videoPath=None, loop=None, output_path="", trackingAlt=True):
    """
    检测本地视频目标或者检测开启摄像头后视频的目标
    :param videoPath: 视频的本地地址；默认或者为0表示开启摄像头
    :param loop: 每隔（loop-1）帧做一次检测
    :param output_path: 是否保存地址
    :param trackingAlt: 是否使用跟踪算法
    :return:
    """
    if (videoPath == None) or (videoPath == 0):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("The camera is not opening!")
            return
    else:
        cap = cv2.VideoCapture(videoPath)
        if not cap.isOpened():
            print("Invalid address entered!")
            return
    if loop == None:
        loop = 5
    if loop < 1:
        print("Please enter an integer greater than one.loop:%f", loop)
    video_FourCC = int(cap.get(cv2.CAP_PROP_FOURCC))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    loc = []  # 位置信息 x, y, w, h
    scores = []  # 置信度
    lab = []  # 目标的标签
    frame = 0  # 帧数，循环控制参数
    initImg = []  # 跟踪算法的初始化图片
    start = False
    start1_time = timer()
    while True:

        pro_time = timer()
        return_value, img = cap.read()
        if img is None:
            break
        img_array = Image.fromarray(img)

        if frame % loop == 0:
            # loc, scores, lab = yolov3.yolo_detect(img)      # 目标检测
            print('start  detect ')

            r_image, loc, scores, lab = yolo.predict(img_array)
            result = np.asarray(r_image)
            initImg = img
        else:
            if trackingAlt:

                ret = tracking(yolo, initImg, img, loc, scores, lab, frame)  # 目标跟踪
                if ret is not None:
                    loc, scores, lab = ret[0], ret[1], ret[2]
                    result = img
                else:
                    print('start  detect ')
                    r_image, loc, scores, lab = yolo.predict(img_array)
                    result = np.asarray(r_image)
            else:
                # loc, scores, lab = yolov3.yolo_detect(img)
                print('start  detect ')
                r_image, loc, scores, lab = yolo.predict(img_array)
                result = np.asarray(r_image)

        post_time = timer()
        exec_time = post_time - pro_time
        curr_fps = 1 / exec_time
        fps = '{}: {:.3f}'.format('FPS', curr_fps)
        if frame % loop == 0:
            print('detect time = {}  '.format(exec_time))
        else:
            if trackingAlt:
                print('tracker time = {}  '.format(exec_time))
            else:
                print('detect time = {}  '.format(exec_time))

        # (fps_w, fps_h), baseline = cv2.getTextSize(fps, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        # cv2.rectangle(img, (2, 20 - fps_h - baseline), (2 + fps_w, 18), color=(0, 0, 0), thickness=-1)

        # cv2.putText(img, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #           fontScale=0.5, color=(255, 255, 255), thickness=2)

        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)

        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # quit on ESC button
        if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            break
        frame = frame + 1
        frame = frame % loop
        start = True
    end1_time = timer()
    print('end time is ', (end1_time - start1_time))
    yolo.close_session()


# COLORS = #np.random.randint(0, 255, size=(80, 3), dtype='uint8')

def union(au, bu, area_intersection):
    """
    并集
    :param au:
    :param bu:
    :param area_intersection:
    :return:
    """
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union


def intersection(ai, bi):
    """
    交集
    :param ai:
    :param bi:
    :return:
    """
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w * h


def iou(a, b):
    """
     交并比：a and b should be (x1,y1,x2,y2)
    :param a:
    :param b:
    :return:
    """

    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)


def updateTracker(newLab, newBoxes, newScores, initImg=None):
    """
    更新原来trcacker的相关信息
    :param lab:trcacker中的的类别信息
    :param boxes:trcacker中的的位置信息
    :param scores:trcacker中的的分值信息
    :param newLab:检测结果的类别信息
    :param newBoxes:检测结果的位置信息
    :param newScores:检测结果的分值信息
    :return:
    """
    global loc
    global scores
    global lab
    global track
    global kcf_trcaker
    newTracker = []
    newTrack = {}

    for i, newbox in enumerate(newBoxes):
        maxIou = 0.0
        index = 0
        for j, box in enumerate(loc):
            iouv = iou(newbox, box)
            if iouv >= maxIou:
                maxIou = iouv
                index = j
        if maxIou >= IOUTHRESHOLD:  # 如果iou大于阈值代表是同一个目标
            # 原有的目标依然在图中，需要继续保留原目标的位置中心点，tracker
            kcf = kcf_trcaker[index]
            # newTracker.append(kcf)
            if track[kcf]:

                # 如果原来里面有这个目标，同样需要更新一下Tracker，目的是为了纠正追踪框和检测框不一致的问题
                newKcf = cv2.TrackerKCF_create()
                if (newbox[2] - newbox[0]) == 0 or (newbox[3] - newbox[1]) == 0:
                    bug = 0
                ok = newKcf.init(initImg, (newbox[0], newbox[1], newbox[2] - newbox[0], newbox[3] - newbox[1]))
                if not ok:
                    print("1 The tracker initialization failed!")
                    newTrack[kcf] = track[kcf]
                    newTracker.append(kcf)
                else:
                    newTrack[newKcf] = track[kcf]
                    newTracker.append(newKcf)

            else:
                newTrack[kcf] = []
                newTracker.append(kcf)
        elif initImg is not None:  # 初始化跟踪器
            if maxIou > 0 and maxIou < IOUTHRESHOLD:
                print('maxIou = ', maxIou)
            kcf = cv2.TrackerKCF_create()
            # kcf = cv2.TrackerCSRT_create()
            if (newbox[2] - newbox[0]) == 0 or (newbox[3] - newbox[1]) == 0:
                bug = 0
            ok = kcf.init(initImg, (newbox[0], newbox[1], newbox[2] - newbox[0], newbox[3] - newbox[1]))
            if not ok:
                print("The tracker initialization failed!")
                return
            newTracker.append(kcf)
            newTrack[kcf] = []
        else:
            print('initImg error')
            return

    loc = newLab
    scores = newScores
    lab = newLab
    kcf_trcaker = newTracker
    track = newTrack


def tracking(yolo, initImg, img, loc, scores, lab, frame, isDraw=True, isDebug=False):
    """
    使用跟踪算法跟踪目标
    :param initImg: 初始化的图片
    :param img: 需要跟踪的图片
    :param loc: 初始化的位置信息
    :param scores: 初始化的置信度
    :param lab: 初始化的类别
    :param frame: 判断是否需要初始化
    :param isDraw: 判断是否在当前桢画框
    :return:
    """
    ''' 
    # 初始化跟踪器
    if frame == 1:

        global tracker
        global kcf_trcaker
        global track
        # 这步很重要，每次初始化跟踪时需要清除原先所跟踪的目标；否则，跟踪的目标会累加
        tracker = cv2.MultiTracker_create()   # tracker.clear()不能清除原先所跟踪的目标；暂时只能写成这样
        kcf_trcaker = []
        track = {}
        for i, newbox in enumerate(loc):
            # TrackerCSRT_create(),TrackerKCF_create(),TrackerMOSSE_create()
            kcf = cv2.TrackerKCF_create()
            ok = kcf.init(initImg,  (newbox[0], newbox[1], newbox[2]-newbox[0],newbox[3]- newbox[1]))
            kcf_trcaker.append(kcf)
            track[kcf] = []
            #ok = tracker.add(kcf, initImg, (newbox[0], newbox[1], newbox[2]-newbox[0],newbox[3]- newbox[1]))
            if not ok:
                print("The tracker initialization failed!")
                return

    ok, boxes = tracker.update(img)
    '''
    boxes = []
    for tr in kcf_trcaker:
        tr.update(img)
        ok, box = tr.update(img)
        boxes.append((box[0], box[1], int(box[0] + box[2]), int(box[1] + box[3])))

    if isDraw == True:
        img = draw(yolo, lab, boxes, scores, img, '', True, isDebug=isDebug)
    '''
    for i, newbox in enumerate(boxes):
        color =  yolo.getColor(lab[i])#[int(c) for c in COLORS[i]]
        #x, y, w, h = int(newbox[0]), int(newbox[1]), int(newbox[2]), int(newbox[3])
        x, y, w, h = int(newbox[0]), int(newbox[1]), int(newbox[2]), int(newbox[3])
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        text = 'tracker {}: {:.3f}'.format(yolo.getClass_names(lab[i]), scores[i])
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (x, y - text_h - baseline), (x + text_w, y), color, -1)
        cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    '''
    return boxes, scores, lab


if __name__ == "__main__":
    detect_camera3(videoPath="D:\GitHub_Repository\\test.mp4",
                  loop=10)  # videoPath="./video/12.avi"
# Import required libraries
from PIL import Image
import cv2
import numpy as np
import streamlit as st
import requests
import os
import pandas as pd
import time
import pandas as pd
from imutils.video import FPS
from sendEmail import sendEmail
from datetime import datetime
import pytz
# Detection confidence threshold


# Home UI
def main():
    st.set_page_config(layout="wide")

    font_css = """
        <style>
        button[data-baseweb="tab"] {
        font-size: 26px;
        }
        </style>
        """
    st.write(font_css, unsafe_allow_html=True)
    st.header("Major Project Problem Statement")
    st.title("Face Mask Detection System ")
    tabs = st.sidebar.selectbox(
        'Choose one of the following',
        ('WebCam', 'Image','Video',),
        key="main_menu"
    )

    # UI Options
    if tabs == 'WebCam':
        cam()
    if tabs == 'Image':
        main1()
    if tabs == 'Video':
        main2()

def cam():
    takeWebCam()

def takeWebCam():
    st.header("Webcam Live Feed")
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])

    next_frame_towait = 5  # for sms
    # it will get the time zone
    # of the specified location
    IST = pytz.timezone('Asia/Kolkata')

    # # construct the argument parse and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-y", "--yolo", required=True,help="base path to YOLO directory")
    # ap.add_argument("-i", "--input", type=str, default="",help="path to (optional) input video file")
    # ap.add_argument("-o", "--output", type=str, default="",help="path to (optional) output video file")
    # ap.add_argument("-d", "--display", type=int, default=1,help="whether or not output frame should be displayed")
    # ap.add_argument("-c", "--confidence", type=float, default=0.45,help="minimum probability to filter weak detections")
    # ap.add_argument("-t", "--threshold", type=float, default=0.3,help="threshold when applyong non-maxima suppression")
    # ap.add_argument("-u", "--use-gpu", type=bool, default=0,help="boolean indicating if CUDA GPU should be used")
    # ap.add_argument("-e", "--use-email", type=bool, default=0,help="boolean indicating if Emails need to be send or not")
    # args = vars(ap.parse_args())

    # setup e-mail config if yes.
    # if args["use_email"]:
    from sendEmail import sendEmail

    # load the class labels our YOLO model was trained on
    labelsPath = "obj.names"
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label(red and green)
    COLORS = [[0, 0, 255], [0, 255, 0]]

    # derive the paths to the YOLO weights and model configuration
    # weightsPath = os.path.sep.join([args["yolo"], "yolov4_face_mask.weights"])
    # configPath = os.path.sep.join([args["yolo"], "yolov4-obj.cfg"])
    ## Model Files
    configPath = 'yolov4-obj.cfg'
    weightsPath = 'yolov4_face_mask.weights'

    # load our YOLO object detector trained on mask dataset
    print("[INFO] loading YOLO from disk...")
    # configure the network model
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    confThreshold = 0.2
    nmsThreshold = 0.2

    font_color = (0, 0, 255)
    font_size = 0.5
    font_thickness = 2

    # Configure the network backend
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # initialize the width and height of the frames in the video file
    W = None
    H = None
    # determine only the *output* layer names that we need from YOLO
    # ln = net.getLayerNames()
    # ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    cap = cv2.VideoCapture(0)
    while run:
        frames_count, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(
            cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = int(width)
        height = int(height)
        print(frames_count, fps, width, height)

        # creates a pandas data frame with the number of rows the same length as frame count
        df = pd.DataFrame(index=range(int(frames_count)))
        df.index.name = "Frames"

        framenumber = 0  # keeps track of current frame

        # information to start saving a video file
        ret, frame = cap.read()  # import image
        ratio = .5  # resize ratio
        image = cv2.resize(frame, (0, 0), None, ratio, ratio)  # resize image
        width2, height2, channels = image.shape
        # video = cv2.VideoWriter('traffic_counter.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (height2, width2), 1)
        # writer= cv2.VideoWriter('video1.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width,height))
        video = cv2.VideoWriter('traffic_counter4.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 1000, (height2, width2), True)

        while True:
            ret, frame = cap.read()  # import image
            if ret:  # if there is a frame continue with code
                # if the frame dimensions are empty, grab them
                if W is None or H is None:
                    (H, W) = frame.shape[:2]
                # image = cv2.resize(frame, (0, 0), None, ratio, ratio)  # resize image
                input_size = 320
                # blob = cv2.dnn.blobFromImage(image, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)
                blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (190, 190), swapRB=True, crop=False)
                # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converts image to gray
                # Set the input of the network
                net.setInput(blob)
                ln = net.getLayerNames()
                ln = [(ln[i - 1]) for i in net.getUnconnectedOutLayers()]
                # Feed data to the network
                layerOutputs = net.forward(ln)
                # initialize our lists of detected bounding boxes, confidences,
                # and class IDs, respectively
                boxes = []
                confidences = []
                classIDs = []
                # loop over each of the layer outputs
                for output in layerOutputs:
                    # loop over each of the detections
                    for detection in output:
                        # extract the class ID and confidence (i.e., probability)
                        # of the current object detection
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]
                        # print(confidence)
                        # filter out weak predictions by ensuring the detected
                        # probability is greater than the minimum probability
                        if confidence > 0.45:
                            # scale the bounding box coordinates back relative to
                            # the size of the image, keeping in mind that YOLO
                            # actually returns the center (x, y)-coordinates of
                            # the bounding box followed by the boxes' width and
                            # height
                            box = detection[0:4] * np.array([W, H, W, H])
                            (centerX, centerY, width1, height1) = box.astype("int")
                            # use the center (x, y)-coordinates to derive the top
                            # and and left corner of the bounding box
                            x = int(centerX - (width1 / 2))
                            y = int(centerY - (height1 / 2))
                            # update our list of bounding box coordinates,
                            # confidences, and class IDs
                            boxes.append([x, y, int(width1), int(height1)])
                            confidences.append(float(confidence))
                            classIDs.append(classID)
                # apply NMS to suppress weak, overlapping
                # bounding boxes
                idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.3)

                # Add top-border to frame to display stats
                border_size = 100
                border_text_color = [255, 255, 255]
                frame = cv2.copyMakeBorder(frame, border_size, 0, 0, 0, cv2.BORDER_CONSTANT)
                # calculate count values
                filtered_classids = np.take(classIDs, idxs)
                mask_count = (filtered_classids == 1).sum()
                nomask_count = (filtered_classids == 0).sum()
                # display count
                text = "NoMaskCount: {}  MaskCount: {}".format(nomask_count, mask_count)
                cv2.putText(frame, text, (0, int(border_size - 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, border_text_color,
                            2)
                # display status
                text = "Status:"
                cv2.putText(frame, text, (W - 200, int(border_size - 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            border_text_color, 2)
                ratio = nomask_count / (mask_count + nomask_count + 0.000001)

                if ratio >= 0.1 and nomask_count >= 3:
                    text = "Danger !"
                    cv2.putText(frame, text, (W - 100, int(border_size - 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                [26, 13, 247], 2)
                    if fps >= next_frame_towait:  # to send danger sms again,only after skipping few seconds
                        msg = "**Face Mask System Alert** \n\n"
                        msg += "Camera ID: C001 \n\n"
                        msg += "Status: Danger! \n\n"
                        msg += "No_Mask Count: " + str(nomask_count) + " \n"
                        msg += "Mask Count: " + str(mask_count) + " \n"
                        datetime_ist = datetime.now(IST)
                        msg += "Date-Time of alert: \n" + datetime_ist.strftime('%Y-%m-%d %H:%M:%S %Z')
                        # sendSMS(msg,[7041677471])
                        # print('Sms sent')
                        sendEmail(msg)
                        next_frame_towait = fps + (5 * 25)
                elif ratio != 0 and np.isnan(ratio) != True:
                    text = "Warning !"
                    cv2.putText(frame, text, (W - 100, int(border_size - 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                [0, 255, 255], 2)

                else:
                    text = "Safe "
                    cv2.putText(frame, text, (W - 100, int(border_size - 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                [0, 255, 0],
                                2)

                # ensure at least one detection exists
                if len(idxs) > 0:
                    # loop over the indexes we are keeping
                    for i in idxs.flatten():
                        # extract the bounding box coordinates
                        (x, y) = (boxes[i][0], boxes[i][1] + border_size)
                        (w, h) = (boxes[i][2], boxes[i][3])
                        # draw a bounding box rectangle and label on the image
                        color = [int(c) for c in COLORS[classIDs[i]]]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
                        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # check to see if the output frame should be displayed to our
                # screen
                # if 1 > 0:
                #     # show the output frame
                #     # displays images and transformations
                #     cv2.imshow("Frame", frame)
                #     cv2.moveWindow("Frame", 0, 0)
                #     video.write(image)  # save the current image to video file from earlier
                #
                #     # adds to framecount
                #     framenumber = framenumber + 1
                #
                #     k = cv2.waitKey(int(1000 / fps)) & 0xff  # int(1000/fps) is normal speed since waitkey is in ms
                #     if k == ord("q"):  # if the `q` key was pressed, break from the loop
                #         break
                # show the output frame
                # displays images and transformations
                cv2.imshow("Frame", frame)
                FRAME_WINDOW.image(frame)
                cv2.moveWindow("Frame", 0, 0)
                # fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                # video = cv2.VideoWriter("op.mp4", fourcc, 24, (frame.shape[1], frame.shape[0]), True)
                # video = cv2.VideoWriter('traffic_counter4.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame.shape[1], frame.shape[0]), True)
                video.write(frame)  # save the current image to video file from earlier

                # adds to framecount
                framenumber = framenumber + 1

                k = cv2.waitKey(int(1000 / fps)) & 0xff  # int(1000/fps) is normal speed since waitkey is in ms
                if k == ord("q"):  # if the `q` key was pressed, break from the loop
                    break
            else:  # if video is finished then break loop
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        st.write("Stopped")


def main1():
    st.header("With the help of Image Files")
    filename = uploadImage()
    print(filename)
    print("uploadImage working")
    finalop = output1(filename.name)
    print("output Working")
    #st.video('traffic_counter1.avi','rb')
    #displayOutput(finalop)
    displayOutput(finalop)

def uploadImage():
    image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])
    if image_file is not None:
        # To See details
        file_details = {"filename": image_file.name, "filetype": image_file.type,"filesize": image_file.size}
        st.write(file_details)
        # To View Uploaded Image
        image = Image.open(image_file)

    print("Work1")
    return image_file

def output1(filename):

    next_frame_towait = 5  # for sms
    # it will get the time zone
    # of the specified location
    IST = pytz.timezone('Asia/Kolkata')

    # # construct the argument parse and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-y", "--yolo", required=True,help="base path to YOLO directory")
    # ap.add_argument("-i", "--input", type=str, default="",help="path to (optional) input video file")
    # ap.add_argument("-o", "--output", type=str, default="",help="path to (optional) output video file")
    # ap.add_argument("-d", "--display", type=int, default=1,help="whether or not output frame should be displayed")
    # ap.add_argument("-c", "--confidence", type=float, default=0.45,help="minimum probability to filter weak detections")
    # ap.add_argument("-t", "--threshold", type=float, default=0.3,help="threshold when applyong non-maxima suppression")
    # ap.add_argument("-u", "--use-gpu", type=bool, default=0,help="boolean indicating if CUDA GPU should be used")
    # ap.add_argument("-e", "--use-email", type=bool, default=0,help="boolean indicating if Emails need to be send or not")
    # args = vars(ap.parse_args())

    # setup e-mail config if yes.
    # if args["use_email"]:
    from sendEmail import sendEmail

    # load the class labels our YOLO model was trained on
    labelsPath = "obj.names"
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label(red and green)
    COLORS = [[0, 0, 255], [0, 255, 0]]

    # derive the paths to the YOLO weights and model configuration
    # weightsPath = os.path.sep.join([args["yolo"], "yolov4_face_mask.weights"])
    # configPath = os.path.sep.join([args["yolo"], "yolov4-obj.cfg"])
    ## Model Files
    configPath = 'yolov4-obj.cfg'
    weightsPath = 'yolov4_face_mask.weights'

    # load our YOLO object detector trained on mask dataset
    print("[INFO] loading YOLO from disk...")
    # configure the network model
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # load our input image and get it height and width

    image = cv2.imread(filename)
    (H, W) = image.shape[:2]
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()

    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (832, 832), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)  # list of 3 arrays, for each output layer.
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:

        # loop over each of the detections
        for detection in output:

            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]  # last 2 values in vector
            classID = np.argmax(scores)
            confidence = scores[classID]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > 0.45:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply NMS to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.3)
    border_size = 100
    border_text_color = [255, 255, 255]
    # Add top-border to image to display stats
    image = cv2.copyMakeBorder(image, border_size, 0, 0, 0, cv2.BORDER_CONSTANT)
    # calculate count values
    filtered_classids = np.take(classIDs, idxs)
    mask_count = (filtered_classids == 1).sum()
    nomask_count = (filtered_classids == 0).sum()
    # display count
    text = "NoMaskCount: {}  MaskCount: {}".format(nomask_count, mask_count)
    cv2.putText(image, text, (0, int(border_size - 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, border_text_color, 2)
    # display status
    text = "Status:"
    cv2.putText(image, text, (W - 300, int(border_size - 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, border_text_color, 2)
    ratio = nomask_count / (mask_count + nomask_count)

    if ratio >= 0.1 and nomask_count >= 3:
        text = "Danger !"
        cv2.putText(image, text, (W - 200, int(border_size - 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [26, 13, 247], 2)

    elif ratio != 0 and np.isnan(ratio) != True:
        text = "Warning !"
        cv2.putText(image, text, (W - 200, int(border_size - 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [0, 255, 255], 2)

    else:
        text = "Safe "
        cv2.putText(image, text, (W - 200, int(border_size - 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [0, 255, 0], 2)

    # ensure at least one detection exists
    if len(idxs) > 0:

        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1] + border_size)
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    #cv2.imwrite("opimage", image)
    # Filename
    filename = 'savedImage.jpg'
    # Saving the image
    cv2.imwrite(filename, image)

    print(filename)
    return filename

def displayOutput(finalop):
    st.write("Final Output")
    st.write(finalop)
    #uploaded_file = st.file_uploader("Choose a Video file", type=["mp4"],accept_multiple_files=True)
    #if uploaded_file is None:
    #    st.write("Upload a Output Video File")
    #st.write(uploaded_file)
    st.image(finalop,use_column_width=True)
    print("Work2")
    exit(0)

def main2():
    st.header("With the help of Video Files")
    filename = uploadVideo()
    print(filename)
    print("uploadVideo working")
    finalop = output(filename.name)
    print("output Working")
    # st.video('traffic_counter1.avi','rb')
    # displayOutput(finalop)
    #displayOutput1(finalop)

def uploadVideo():
    uploaded_file = st.file_uploader("Choose a Video file",type=["mp4"])
    if uploaded_file is None:
        st.write("Upload a Video File")
    st.write(uploaded_file)
    st.video(uploaded_file,'rb')
    print("Work1")
    return uploaded_file

def output(filename):
    FRAME_WINDOW = st.image([])
    next_frame_towait = 5  # for sms
    # it will get the time zone
    # of the specified location
    IST = pytz.timezone('Asia/Kolkata')

    # # construct the argument parse and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-y", "--yolo", required=True,help="base path to YOLO directory")
    # ap.add_argument("-i", "--input", type=str, default="",help="path to (optional) input video file")
    # ap.add_argument("-o", "--output", type=str, default="",help="path to (optional) output video file")
    # ap.add_argument("-d", "--display", type=int, default=1,help="whether or not output frame should be displayed")
    # ap.add_argument("-c", "--confidence", type=float, default=0.45,help="minimum probability to filter weak detections")
    # ap.add_argument("-t", "--threshold", type=float, default=0.3,help="threshold when applyong non-maxima suppression")
    # ap.add_argument("-u", "--use-gpu", type=bool, default=0,help="boolean indicating if CUDA GPU should be used")
    # ap.add_argument("-e", "--use-email", type=bool, default=0,help="boolean indicating if Emails need to be send or not")
    # args = vars(ap.parse_args())

    # setup e-mail config if yes.
    # if args["use_email"]:
    from sendEmail import sendEmail

    # load the class labels our YOLO model was trained on
    labelsPath = "obj.names"
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label(red and green)
    COLORS = [[0, 0, 255], [0, 255, 0]]

    # derive the paths to the YOLO weights and model configuration
    # weightsPath = os.path.sep.join([args["yolo"], "yolov4_face_mask.weights"])
    # configPath = os.path.sep.join([args["yolo"], "yolov4-obj.cfg"])
    ## Model Files
    configPath = 'yolov4-obj.cfg'
    weightsPath = 'yolov4_face_mask.weights'

    # load our YOLO object detector trained on mask dataset
    print("[INFO] loading YOLO from disk...")
    # configure the network model
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    confThreshold = 0.2
    nmsThreshold = 0.2

    font_color = (0, 0, 255)
    font_size = 0.5
    font_thickness = 2

    # Configure the network backend
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # initialize the width and height of the frames in the video file
    W = None
    H = None
    # determine only the *output* layer names that we need from YOLO
    # ln = net.getLayerNames()
    # ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    cap = cv2.VideoCapture(filename)
    frames_count, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(
        cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = int(width)
    height = int(height)
    print(frames_count, fps, width, height)

    # creates a pandas data frame with the number of rows the same length as frame count
    df = pd.DataFrame(index=range(int(frames_count)))
    df.index.name = "Frames"

    framenumber = 0  # keeps track of current frame

    # information to start saving a video file
    ret, frame = cap.read()  # import image
    ratio = .5  # resize ratio
    image = cv2.resize(frame, (0, 0), None, ratio, ratio)  # resize image
    width2, height2, channels = image.shape
    # video = cv2.VideoWriter('traffic_counter.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (height2, width2), 1)
    # writer= cv2.VideoWriter('video1.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width,height))
    video = cv2.VideoWriter('traffic_counter4.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 1000, (height2, width2), True)

    while True:
        ret, frame = cap.read()  # import image
        if ret:  # if there is a frame continue with code
            # if the frame dimensions are empty, grab them
            if W is None or H is None:
                (H, W) = frame.shape[:2]
            # image = cv2.resize(frame, (0, 0), None, ratio, ratio)  # resize image
            input_size = 320
            # blob = cv2.dnn.blobFromImage(image, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)
            #blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (864, 864), swapRB=True, crop=False)
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (190, 190), swapRB=True, crop=False)
            # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converts image to gray
            # Set the input of the network
            net.setInput(blob)
            ln = net.getLayerNames()
            ln = [(ln[i - 1]) for i in net.getUnconnectedOutLayers()]
            # Feed data to the network
            layerOutputs = net.forward(ln)
            # initialize our lists of detected bounding boxes, confidences,
            # and class IDs, respectively
            boxes = []
            confidences = []
            classIDs = []
            # loop over each of the layer outputs
            for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                    # extract the class ID and confidence (i.e., probability)
                    # of the current object detection
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    # print(confidence)
                    # filter out weak predictions by ensuring the detected
                    # probability is greater than the minimum probability
                    if confidence > 0.45:
                        # scale the bounding box coordinates back relative to
                        # the size of the image, keeping in mind that YOLO
                        # actually returns the center (x, y)-coordinates of
                        # the bounding box followed by the boxes' width and
                        # height
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width1, height1) = box.astype("int")
                        # use the center (x, y)-coordinates to derive the top
                        # and and left corner of the bounding box
                        x = int(centerX - (width1 / 2))
                        y = int(centerY - (height1 / 2))
                        # update our list of bounding box coordinates,
                        # confidences, and class IDs
                        boxes.append([x, y, int(width1), int(height1)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
            # apply NMS to suppress weak, overlapping
            # bounding boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.3)

            # Add top-border to frame to display stats
            border_size = 100
            border_text_color = [255, 255, 255]
            frame = cv2.copyMakeBorder(frame, border_size, 0, 0, 0, cv2.BORDER_CONSTANT)
            # calculate count values
            filtered_classids = np.take(classIDs, idxs)
            mask_count = (filtered_classids == 1).sum()
            nomask_count = (filtered_classids == 0).sum()
            # display count
            text = "NoMaskCount: {}  MaskCount: {}".format(nomask_count, mask_count)
            cv2.putText(frame, text, (0, int(border_size - 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, border_text_color, 2)
            # display status
            text = "Status:"
            cv2.putText(frame, text, (W - 200, int(border_size - 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        border_text_color, 2)
            ratio = nomask_count / (mask_count + nomask_count + 0.000001)

            if ratio >= 0.1 and nomask_count >= 3:
                text = "Danger !"
                cv2.putText(frame, text, (W - 100, int(border_size - 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            [26, 13, 247], 2)
                if fps >= next_frame_towait:  # to send danger sms again,only after skipping few seconds
                    msg = "**Face Mask System Alert** \n\n"
                    msg += "Camera ID: C001 \n\n"
                    msg += "Status: Danger! \n\n"
                    msg += "No_Mask Count: " + str(nomask_count) + " \n"
                    msg += "Mask Count: " + str(mask_count) + " \n"
                    datetime_ist = datetime.now(IST)
                    msg += "Date-Time of alert: \n" + datetime_ist.strftime('%Y-%m-%d %H:%M:%S %Z')
                    # sendSMS(msg,[7041677471])
                    # print('Sms sent')
                    sendEmail(msg)
                    next_frame_towait = fps + (5 * 25)
            elif ratio != 0 and np.isnan(ratio) != True:
                text = "Warning !"
                cv2.putText(frame, text, (W - 100, int(border_size - 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            [0, 255, 255], 2)

            else:
                text = "Safe "
                cv2.putText(frame, text, (W - 100, int(border_size - 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, [0, 255, 0],
                            2)

            # ensure at least one detection exists
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1] + border_size)
                    (w, h) = (boxes[i][2], boxes[i][3])
                    # draw a bounding box rectangle and label on the image
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
                    text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # check to see if the output frame should be displayed to our
            # screen
            # if 1 > 0:
            #     # show the output frame
            #     # displays images and transformations
            #     cv2.imshow("Frame", frame)
            #     cv2.moveWindow("Frame", 0, 0)
            #     video.write(image)  # save the current image to video file from earlier
            #
            #     # adds to framecount
            #     framenumber = framenumber + 1
            #
            #     k = cv2.waitKey(int(1000 / fps)) & 0xff  # int(1000/fps) is normal speed since waitkey is in ms
            #     if k == ord("q"):  # if the `q` key was pressed, break from the loop
            #         break
            # show the output frame
            # displays images and transformations
            cv2.imshow("Frame", frame)
            cv2.moveWindow("Frame", 0, 0)
            # fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            # video = cv2.VideoWriter("op.mp4", fourcc, 24, (frame.shape[1], frame.shape[0]), True)
            # video = cv2.VideoWriter('traffic_counter4.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame.shape[1], frame.shape[0]), True)
            FRAME_WINDOW.image(frame)
            video.write(frame)  # save the current image to video file from earlier

            # adds to framecount
            framenumber = framenumber + 1

            k = cv2.waitKey(int(1000 / fps)) & 0xff  # int(1000/fps) is normal speed since waitkey is in ms
            if k == ord("q"):  # if the `q` key was pressed, break from the loop
                break
        else:  # if video is finished then break loop
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


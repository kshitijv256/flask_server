"""
@ on 4/14/2022 by kshitijv256
Driver file for input image processing and result prediction

@ on 10/24/2022 by kshitijv256
Updated the code to use CNN model for prediction 

@ on 11/02/2022 by kshitijv256
Updated the code to be use in Flask app

"""
import cv2
import os
import numpy as np
from keras.models import load_model
import pickle
from tensorflow.keras.optimizers import Adamax


## preparing
def prepareEncoder():
    with open("./Assets/models/encoder.pickle", "rb") as handle:
        label_encoder = pickle.load(handle)
    return label_encoder


# Load model
def loadModel():
    model = load_model("./Assets/models/modelv3.h5", compile=False)
    model.compile(
        Adamax(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# Load Image
def loadImg(path):
    y = cv2.imread(path)
    img = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)
    return img


# Find contours
def findContours(img):
    if img is not None:
        img = ~img
        ret, thresh = cv2.threshold(img, 120, 255, 0)
        ctrs, ret = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

        w = int(128)
        h = int(128)
        testset = []

        rects = []
        for c in cnt:
            x, y, w, h = cv2.boundingRect(c)
            rect = [x - 5, y - 10, w + 10, h + 20]
            rects.append(rect)

        bool_rect = []
        for r in rects:
            l = []
            for rec in rects:
                flag = 0
                if rec != r:
                    if (
                        r[0] < (rec[0] + rec[2] + 10)
                        and rec[0] < (r[0] + r[2] + 10)
                        and r[1] < (rec[1] + rec[3] + 10)
                        and rec[1] < (r[1] + r[3] + 10)
                    ):
                        flag = 1
                    l.append(flag)
                if rec == r:
                    l.append(0)
            bool_rect.append(l)

        dump_rect = []
        for i in range(0, len(cnt)):
            for j in range(0, len(cnt)):
                if bool_rect[i][j] == 1:
                    area1 = rects[i][2] * rects[i][3]
                    area2 = rects[j][2] * rects[j][3]
                    if area1 == min(area1, area2):
                        dump_rect.append(rects[i])

        final_rect = [i for i in rects if i not in dump_rect]
        count = 1
        for r in final_rect:
            x = r[0]
            y = r[1]
            w = r[2]
            h = r[3]
            if w < 50 and h < 50:
                continue
            h = max(h, w)
            w = max(h, w)
            im_crop = thresh[y - h // 2 : y + h + 10, x : x + w + 10]

            # height, width = im_crop.shape
            # x = height if height > width else width
            # y = height if height > width else width

            # ## To make contours square
            # square = np.zeros((x, y), np.uint8)
            # square[
            #     int((y - height) / 2) : int(y - (y - height) / 2),
            #     int((x - width) / 2) : int(x - (x - width) / 2),
            # ] = im_crop
            # im_crop = square

            ## To resize image
            img = cv2.resize(im_crop, (96, 96))
            img = ~img
            # img = cv2.blur(img, (5, 5))
            cv2.imwrite("./contours/contours{}.jpg".format(count), img)
            count += 1
            img = np.array(img)
            img = np.expand_dims(img, axis=0)
            stacked_img = np.stack((img,) * 3, axis=-1)
            testset.append(stacked_img)
        testset = np.array(testset)
        return testset
    return None


# Predict
def predictValues(features, model):
    res = []
    for i in range(len(features)):
        image = features[i]
        image = image / 255.0
        prediction = model.predict(image)
        value = np.argmax(prediction)
        res.append(value)
        print("Prediction is {}.".format(value))
    return res


def predictResult(result, model):
    result = predictValues(result, model)
    equation = ""
    for i in range(len(result)):
        if result[i] < 10:
            equation = equation + str(result[i])
        if result[i] == 10:
            equation = equation + "+"
        if result[i] == 11:
            equation = equation + "."
        if result[i] == 12:
            equation = equation + "/"
        if result[i] == 13:
            equation = equation + "*"
        if result[i] == 14:
            equation = equation + "-"
    try:
        out = eval(equation)
    except:
        out = "Incorrect equation"
    return equation, out
    # s = ""
    # for i in range(len(testset)):
    #     pred = model.predict(testset[i])
    #     result = np.argsort(pred)
    #     result_temp = result[0][::-1]
    #     # print(result_temp)

    #     result = label_encoder.inverse_transform(np.array(result_temp))

    #     if result[0] == "div" or result[0] == "forward_slash":
    #         s = s + "/"
    #     elif result[0] == "times":
    #         s = s + "*"
    #     ## NEW CODE
    #     elif result[0] == "(" or result[0] == "{" or result[0] == "[":
    #         s = s + "("
    #     elif result[0] == ")" or result[0] == "}" or result[0] == "]":
    #         s = s + ")"
    #     elif result[0] == "=":
    #         s = s + "-"
    #     else:
    #         s = s + result[0]
    # try:
    #     x = eval(s)
    #     return s, x
    # except:
    #     return s, None

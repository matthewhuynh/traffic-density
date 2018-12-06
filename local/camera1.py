import cv2
import numpy as np
import urllib.request
import time
import requests
import json

"""<-----------------------Functions----------------------->"""
def getCars(image, Height, Width):
    classes = None
    with open('yolov3.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

    blob = cv2.dnn.blobFromImage(
        image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.4
    nms_threshold = 0.07
    cars = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            if class_id != 2 and class_id != 5 and class_id != 7 and class_id != 3:
                continue
            confidence = scores[class_id]
            if confidence > 0:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        if str(classes[class_ids[i]]) != 'car' and str(classes[class_ids[i]]) != 'truck' and str(classes[class_ids[i]]) != 'bus' and str(classes[class_ids[i]]) != 'motorcycle':
            continue
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        cars.append(str(int(x)) + '||' + str(int(y)) +
                    '||' + str(int(w)) + '||' + str(int(h)))

    return cars

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1]
                     for i in net.getUnconnectedOutLayers()]
    return output_layers

def getStateOfRoad(img,np_array, count):
    img = cv2.imread(img)
    h, w = img.shape[:2]

    AREA_PTS = np.array(np_array)

    cars = getCars(img, h, w)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img)

    edge = ~cv2.Canny(cv2.bilateralFilter(cl1, 10, 200, 200), 45, 70)

    #blur = cv2.bilateralFilter(cv2.blur(edge,(21,21), 100),9,200,200)
    blur = cv2.GaussianBlur(cv2.blur(edge, (21, 21), 100), (5, 5), 1)
    _, threshold = cv2.threshold(blur, 220, 255, cv2.THRESH_BINARY)

    base = np.zeros((h, w) + (3,), dtype='uint8')
    area_mask = cv2.fillPoly(base, [AREA_PTS], (255, 255, 255))[:, :, 0]

    vrx = np.array(np_array, np.int32)
    vrx = vrx.reshape((-1,1,2))
    cv2.polylines(threshold, [vrx], True, (255, 255, 255),1)

    for k in cars:
        x = int(k.split('||')[0])
        y = int(k.split('||')[1])
        w = int(k.split('||')[2])
        h = int(k.split('||')[3])
        vrx1 = np.array([[int(x + (w/2)), int(y)], [int(x + w), int(y + (h/2))],
                        [int(x + (w/2)), int(y+h)], [int(x), int(y + (h/2))]], np.int32)
        cv2.fillPoly(threshold, np.int_([vrx1]), (0, 255, 255))

    t = cv2.bitwise_and(threshold, threshold, mask=area_mask)
    free = np.count_nonzero(t)
    allBackground = np.count_nonzero(area_mask)
    capacity = 1 - float(free)/allBackground
    cv2.putText(t,"Density: "+str((int)(capacity * 100)) +" %",(10,37), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 3)

    if (np_array == [[98,287], [136,55], [170,55], [450,287]]):
        # cl1 = cv2.resize(cl1, (0, 0), None, .85, .85)
        # edge = cv2.resize(edge, (0, 0), None, .85, .85)
        # blur = cv2.resize(blur, (0, 0), None, .85, .85)
        # t = cv2.resize(t, (0, 0), None, .85, .85)
        for i in cars:
            x = int(i.split('||')[0])
            y = int(i.split('||')[1])
            w = int(i.split('||')[2])
            h = int(i.split('||')[3])
            cv2.rectangle(cl1, (int(round(x)),int(round(y))), (int(round(x+w)),int(round(y+h))), (255, 0, 0), 2)
            cv2.rectangle(t, (int(round(x)),int(round(y))), (int(round(x+w)),int(round(y+h))), (255, 0, 0), 2)


        numpy_horizontal1 = np.concatenate((cl1, edge), axis=1)
        numpy_horizontal2 = np.concatenate((blur, t), axis=1)
        numpy_vertical = np.concatenate((numpy_horizontal1, numpy_horizontal2), axis=0)
        cv2.imwrite("result1/result1_" + str(count) + ".jpg", numpy_vertical)
    if (np_array == [[195,450], [595,180], [670,180], [625,450]]):
        # cl1 = cv2.resize(cl1, (0, 0), None, .85, .85)
        # edge = cv2.resize(edge, (0, 0), None, .85, .85)
        # blur = cv2.resize(blur, (0, 0), None, .85, .85)
        # t = cv2.resize(t, (0, 0), None, .85, .85)
        for i in cars:
            x = int(i.split('||')[0])
            y = int(i.split('||')[1])
            w = int(i.split('||')[2])
            h = int(i.split('||')[3])
            cv2.rectangle(cl1, (int(round(x)),int(round(y))), (int(round(x+w)),int(round(y+h))), (255, 0, 0), 2)
            cv2.rectangle(t, (int(round(x)),int(round(y))), (int(round(x+w)),int(round(y+h))), (255, 0, 0), 2)


        numpy_horizontal1 = np.concatenate((cl1, edge), axis=1)
        numpy_horizontal2 = np.concatenate((blur, t), axis=1)
        numpy_vertical = np.concatenate((numpy_horizontal1, numpy_horizontal2), axis=0)
        cv2.imwrite("result2/result2_" + str(count) + ".jpg", numpy_vertical)
    if (np_array == [[285,80], [185,83], [100,85], [0,140], [0,287], [511,287], [511,128], [450,100]]):
        # cl1 = cv2.resize(cl1, (0, 0), None, .85, .85)
        # edge = cv2.resize(edge, (0, 0), None, .85, .85)
        # blur = cv2.resize(blur, (0, 0), None, .85, .85)
        # t = cv2.resize(t, (0, 0), None, .85, .85)
        for i in cars:
            x = int(i.split('||')[0])
            y = int(i.split('||')[1])
            w = int(i.split('||')[2])
            h = int(i.split('||')[3])
            cv2.rectangle(cl1, (int(round(x)),int(round(y))), (int(round(x+w)),int(round(y+h))), (255, 0, 0), 2)
            cv2.rectangle(t, (int(round(x)),int(round(y))), (int(round(x+w)),int(round(y+h))), (255, 0, 0), 2)


        numpy_horizontal1 = np.concatenate((cl1, edge), axis=1)
        numpy_horizontal2 = np.concatenate((blur, t), axis=1)
        numpy_vertical = np.concatenate((numpy_horizontal1, numpy_horizontal2), axis=0)
        cv2.imwrite("result3/result3_" + str(count) + ".jpg", numpy_vertical)
    if (np_array == [[610,114], [583,135], [420,165], [140,450], [800,450], [665,170], [675,110]]):
        # cl1 = cv2.resize(cl1, (0, 0), None, .85, .85)
        # edge = cv2.resize(edge, (0, 0), None, .85, .85)
        # blur = cv2.resize(blur, (0, 0), None, .85, .85)
        # t = cv2.resize(t, (0, 0), None, .85, .85)
        for i in cars:
            x = int(i.split('||')[0])
            y = int(i.split('||')[1])
            w = int(i.split('||')[2])
            h = int(i.split('||')[3])
            cv2.rectangle(cl1, (int(round(x)),int(round(y))), (int(round(x+w)),int(round(y+h))), (255, 0, 0), 2)
            cv2.rectangle(t, (int(round(x)),int(round(y))), (int(round(x+w)),int(round(y+h))), (255, 0, 0), 2)


        numpy_horizontal1 = np.concatenate((cl1, edge), axis=1)
        numpy_horizontal2 = np.concatenate((blur, t), axis=1)
        numpy_vertical = np.concatenate((numpy_horizontal1, numpy_horizontal2), axis=0)
        cv2.imwrite("result4/result4_" + str(count) + ".jpg", numpy_vertical)
    return (capacity*100)

# def downloadImageFromCamera(link, folder, count):
#     urllib.request.urlretrieve(link, folder+"/" + folder +"_"+ str(count) + ".jpg") 


def sendJsonREST(URL, data):
    headers = {'content-type': 'application/json'}
    response = requests.post(url = URL,  data=json.dumps(data), headers=headers)
    return response 


def getDataOfCamera(folder, np_array, count):
    # folder = ""
    #tran quoc hoan - hoang van thu
    # if np_array == [[610,114], [583,135], [420,165], [140,450], [800,450], [665,170], [675,110]]:
    #     folder = "camera4"
    # #tran hung dao    
    # elif np_array == [[285,80], [185,83], [100,85], [0,140], [0,287], [511,287], [511,128], [450,100]]:
    #     folder = "camera3"
    # #pham van dong - phan van tri
    # elif np_array == [[195,450], [595,180], [670,180], [625,450]]:
    #     folder = "camera2"
    # #ly tu trong - 2 ba trung    
    # elif np_array == [[98,287], [136,55], [170,55], [450,287]]:
    #     folder = "camera1"
    data ={}
    begin = time.time()
    state = getStateOfRoad(folder + "/" + folder +"_"+ str(count) + ".jpg", np_array, count)
    
    data[folder + "_" + str(time.strftime("%Y-%m-%d %H:%M:%S"))] = int(state)

    end = time.time()
    print (folder +" :" + str(state)[0:5] +"\t" + str(end - begin)[0:5])
    return data
# http://giaothong.hochiminhcity.gov.vn/render/ImageHandler.ashx?id=58af9670bd82540010390c34&t=1536658750404
#AREA_PTS = np.array([[195,450], [590,180], [670,180], [625,450]])

# http://giaothong.hochiminhcity.gov.vn/render/ImageHandler.ashx?id=58af8eb2bd82540010390c30&t=1539670377904
#AREA_PTS = np.array([[450,130], [590,40], [615,50], [600,155]])


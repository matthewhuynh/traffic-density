import cv2
import numpy as np
from matplotlib import pyplot as plt
import urllib.request
import time


begin_time = time.time()

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1]
                     for i in net.getUnconnectedOutLayers()]
    return output_layers

def getCars(image,Height,Width):
    scale = 0.00392
    classes = None

    with open('yolov3.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

    blob = cv2.dnn.blobFromImage(
        image, scale, (416, 416), (0, 0, 0), True, crop=False)

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
            if class_id != 2 and class_id != 5 and class_id != 7:
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
        if str(classes[class_ids[i]]) != 'car' and str(classes[class_ids[i]]) != 'truck' and str(classes[class_ids[i]]) != 'bus':
            continue
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        cars.append(str(int(x)) + '||' + str(int(y)) +
                    '||' + str(int(w)) + '||' + str(int(h)))

    return cars


#urllib.request.urlretrieve('http://giaothong.hochiminhcity.gov.vn/render/ImageHandler.ashx?id=587c8183b807da0011e33d3f&t=1539764591446', 'video.jpg')

# http://giaothong.hochiminhcity.gov.vn/render/ImageHandler.ashx?id=58af9670bd82540010390c34&t=1536658750404
#AREA_PTS = np.array([[195,450], [590,180], [670,180], [625,450]])

# http://giaothong.hochiminhcity.gov.vn/render/ImageHandler.ashx?id=58af8eb2bd82540010390c30&t=1538463084177
#AREA_PTS = np.array([[450,130], [590,40], [615,50], [600,155]])

np_array = [[195,450], [595,180], [670,180], [625,450]]
AREA_PTS = np.array(np_array)

img = cv2.imread('camera2_17.jpg')
h, w = img.shape[:2]
print (w,h)

cars = getCars(img,h,w)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl1 = clahe.apply(img)

smooth = cv2.bilateralFilter(cl1, 10, 200, 200)

# get edge from image
edge = ~cv2.Canny(cv2.bilateralFilter(cl1, 10, 200, 200), 45, 70)

blur = cv2.bilateralFilter(cv2.blur(edge,(21,21), 100),5,200,200)
#blur = cv2.GaussianBlur(cv2.blur(edge, (21, 21), 100), (5, 5), 1)
_, threshold = cv2.threshold(blur, 220, 255, cv2.THRESH_BINARY)

base = np.zeros((h, w) + (3,), dtype='uint8')
area_mask = cv2.fillPoly(base, [AREA_PTS], (255, 255, 255))[:, :, 0]

# for k in cars:
#     x = int(k.split('||')[0])
#     y = int(k.split('||')[1])
#     w = int(k.split('||')[2])
#     h = int(k.split('||')[3])
#     vrx = np.array([[int(x + (w/2)), int(y)], [int(x + w), int(y + (h/2))],
#                     [int(x + (w/2)), int(y+h)], [int(x), int(y + (h/2))]], np.int32)
#     cv2.fillPoly(threshold, np.int_([vrx]), (0, 255, 255))

vrx = np.array(np_array, np.int32)
vrx = vrx.reshape((-1, 1, 2))
cv2.polylines(cl1, [vrx], True, (255, 255, 255), 2)
cv2.polylines(threshold, [vrx], True, (255, 255, 255), 2)

t = cv2.bitwise_and(threshold, threshold, mask=area_mask)
free = np.count_nonzero(t)
allBackground = np.count_nonzero(area_mask)
capacity = 1 - float(free)/allBackground
cv2.putText(t,"Density: "+str((int)(capacity * 100)) +" %",(180,150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 3)
# fig = plt.figure()
# fig.suptitle("Capacity: {}%".format(capacity*100), fontsize=16)
# plt.subplot(221), plt.imshow(cl1), plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(222), plt.imshow(edge), plt.title('Canny Edge')
# plt.xticks([]), plt.yticks([])
# plt.subplot(223), plt.imshow(blur), plt.title('Blur')
# plt.xticks([]), plt.yticks([])
# plt.subplot(224), plt.imshow(t), plt.title('Threshold with ROI mask')
# plt.xticks([]), plt.yticks([])

# fig.savefig('count.png', dpi=500)
for i in cars:
    x = int(i.split('||')[0])
    y = int(i.split('||')[1])
    w = int(i.split('||')[2])
    h = int(i.split('||')[3])
    cv2.rectangle(cl1, (int(round(x)),int(round(y))), (int(round(x+w)),int(round(y+h))), (255, 0, 0), 2)
    cv2.rectangle(t, (int(round(x)),int(round(y))), (int(round(x+w)),int(round(y+h))), (255, 0, 0), 2)
cv2.imwrite("1.jpg", cl1)
cv2.imwrite("2.jpg", edge)
cv2.imwrite("4.jpg", blur)
cv2.imwrite("3.jpg", t)
cv2.imwrite("5.jpg", img)

end_time = time.time()
print("Time: " + str(end_time - begin_time) +
      " s\t" + str(int(capacity*100)) + " %")


#vrx = np.array([[int(x + (w/2)),int(y)], [int(x + w),int(y + (h/2))], [int(x + (w/2)), int(y+h)], [int(x),int(y + (h/2))]], np.int32)
#cv2.fillPoly(image, np.int_([vrx]), (0,255,255))

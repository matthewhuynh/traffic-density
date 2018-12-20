import os                                                                       
from multiprocessing import Pool
import camera1
import json
import time
import cv2

#processes = ('camera1.py', 'camera2.py', 'camera3.py', 'camera4.py')                                    

# link1 = "http://giaothong.hochiminhcity.gov.vn/render/ImageHandler.ashx?id=5a823d555058170011f6eaa2&t=1538726297130"
folder1 = "camera1"
np_array1 = [[98,287], [130,77], [190,77], [450,287]]

# link2 = "http://giaothong.hochiminhcity.gov.vn/render/ImageHandler.ashx?id=58af9670bd82540010390c34&t=1536658750404"
folder2 = "camera2"
np_array2 = [[195,450], [595,180], [670,180], [625,450]]

# link3 = "http://giaothong.hochiminhcity.gov.vn/render/ImageHandler.ashx?id=5b0b7aba0e517b00119fd800&t=1539760306008"
folder3 = "camera3"
np_array3 = [[285,80], [185,83], [100,85], [0,140], [0,287], [511,287], [511,128], [450,100]]

# link4 = "http://giaothong.hochiminhcity.gov.vn/render/ImageHandler.ashx?id=587c8183b807da0011e33d3f&t=1539764591446"
folder4 = "camera4"
np_array4 = [[610,114], [583,135], [420,165], [140,450], [800,450], [665,170], [675,110]]


input = {
    folder1 : np_array1,
    folder2 : np_array2,
    folder3 : np_array3,
    folder4 : np_array4
} 
                                                                                
pool = Pool(processes=4)    
count = 0                              
while(1):
    begin = time.time()
    datas = pool.starmap(camera1.getDataOfCamera, [(folder, np_array, count) for folder, np_array in input.items()])

    jsonData = {}
    for data in datas:
        jsonData.update(data)
        
    URL ="http://localhost:8081/getState"
    response = camera1.sendJsonREST(URL, jsonData)
    
    end = time.time()

    img = cv2.imread("result1/result1_"+str(count)+".jpg")
    cv2.imshow("Result", img)
    cv2.waitKey(3000)

    if (count == 59):
        count = 0
    else: count += 1
    

    print("-------" + str(end-begin)[0:4] + "s-------" +str(count))

    # time.sleep(3)


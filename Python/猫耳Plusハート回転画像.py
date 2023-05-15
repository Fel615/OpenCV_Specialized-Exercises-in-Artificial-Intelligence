import cv2
import numpy as np
from pngoverlay import PNGOverlay




face_cascade_path = 'haarcascade_frontalface_alt.xml'
eye_cascade_path = 'haarcascade_eye.xml'

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

src = cv2.imread('arashi1.jpg')
png_image = cv2.imread("cat_ear2.png", cv2.IMREAD_UNCHANGED)  # アルファチャンネル込みで読み込む
height, width,channels = png_image.shape

png_image2 = cv2.imread("eye.png", cv2.IMREAD_UNCHANGED)  # アルファチャンネル込みで読み込む
height2, width2,channels2 = png_image2.shape

#change_background("eye.png", "eyewhite.png")

#動画ヴァージョン
#fname="testmovie3.avi"
#cap = cv2.VideoCapture(fname)


#USBカメラヴァージョン
cap = cv2.VideoCapture(0)




#png_image2 = cv2.bitwise_not(png_image2)
#cv2.imwrite("eyewhite.png",png_image2)


ratio = 0.05

heart_kosuu=9 #出したいハートの個数
heart_kosuu=int(np.round(360/heart_kosuu,0))

#ハートの角度
angle=0

#ハートと顔の距離の倍率
come_and_go=1
d=0.1


while True:
    #VideoCaptureから1フレーム読み込む
    ret, src = cap.read()
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(src_gray)
    for x, y, w, h in faces:

        rate = w / width  #顔の横幅Widthにより、縮小
        png_image = PNGOverlay("cat_ear2.png")
        png_image.resize(rate)
        png_image.show(src, int(x+w/2), int(y-h*0.05))

        for i in range(angle,angle+360,heart_kosuu):
            

            png_image3 = PNGOverlay("heart3.png")
            png_image3.resize(rate)
            png_image3.rotate(15)
            png_image3.show(src, int(np.round((np.cos(np.radians(i))*w*come_and_go+(x+w/2)) ,0)), int(np.round((np.sin(np.radians(i))*h*come_and_go+((y+h/2))) ,0)) )



        #cv2.rectangle(src, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = src[y: y + h, x: x + w]
        face_gray = src_gray[y: y + h, x: x + w]
        eyes = eye_cascade.detectMultiScale(face_gray)


        for (ex, ey, ew, eh) in eyes:
            #print(ey,y,h)
            if ey<=h/3:
                rate2 = ew / width2  #顔の横幅Widthにより、縮小
                png_image2 = PNGOverlay("eye.png")
                png_image2.resize(rate2)
            
                #png_image2.show(src, int(x+ex+ew/2), int(y+ey+eh/2))


    cv2.imshow("face&eye",src)
    cv2.waitKey(20)
    angle+=15
    come_and_go+=d
    if come_and_go>=1.5 or come_and_go<=1:
                d*=-1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#キャプチャをリリースして、ウィンドウをすべて閉じる
cap.release()
cv2.destroyAllWindows()





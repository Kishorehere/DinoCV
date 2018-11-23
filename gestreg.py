import cv2
import numpy as np
from pynput.keyboard import Key,Controller
keyboard = Controller()

tx,ty,tw,th=(0,0,0,0)
#abc=cv2.imread('images.jpg')
#cv2.imshow('Image',abc)
#cv2.waitKey(0)


cap = cv2.VideoCapture(0)

kernelOpen=np.ones((5,5))
kernelClose=np.ones((20,20))
pinchflag=0


f=0

while True:
    green_co = []
    yellow_co = []
    ret,frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dist = frame.shape[1]/10

    lower_yellow = np.array([20,100,100])
    upper_yellow = np.array([30,255,255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    res=cv2.bitwise_and(frame, frame, mask = mask)
    maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
    maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)
    maskFinal=maskClose

    
    lower_green = np.array([35,100,100])
    upper_green = np.array([85,255,255])

    mask_1 = cv2.inRange(hsv, lower_green, upper_green)
    res_2=cv2.bitwise_and(frame, frame, mask = mask_1)
    maskOpen_1=cv2.morphologyEx(mask_1,cv2.MORPH_OPEN,kernelOpen)
    maskClose_1=cv2.morphologyEx(maskOpen_1,cv2.MORPH_CLOSE,kernelClose)
    maskFinal_1=maskClose_1


    _,counto,h= cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    _,counto_1,h_1= cv2.findContours(maskFinal_1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    #cv2.drawContours(frame,counto,-1,(255,0,0),3)
    
    for i in range(len(counto)):
        x,y,w,h=cv2.boundingRect(counto[i])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        cx=x+w/2
        cy=y+h/2
        cv2.circle(frame,(cx,cy),2,(0,0,255),3)
        yellow_co.append(cx)
        yellow_co.append(cy)

    for i in range(len(counto_1)):
        x,y,w,h=cv2.boundingRect(counto_1[i])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        cx=x+w/2
        cy=y+h/2
        cv2.circle(frame,(cx,cy),2,(255,0,0),3)
        green_co.append(cx)
        green_co.append(cy)
        

    if(len(green_co)==4 and len(yellow_co)==4 ):
        d1 = [green_co[0]-yellow_co[0],green_co[1]-yellow_co[1]]
        d2 = [green_co[0]-yellow_co[2],green_co[1]-yellow_co[3]]
        d3 = [green_co[2]-yellow_co[0],green_co[3]-yellow_co[1]]
        d4 = [green_co[2]-yellow_co[2],green_co[3]-yellow_co[3]]

        a =0
        if(d1[0]<dist and d1[1]<dist):
            a+=1
        if(d2[0]<dist and d2[1]<dist):
            a+=1
        if(d3[0]<dist and d3[1]<dist):
            a+=1
        if(d4[0]<dist and d4[1]<dist):
            a+=1

        if(a==2 and f==0):
            print('screenshot')
            keyboard.press(Key.print_screen)
            f=1
        if(a!=2):
            f=0
    
    
    if(len(counto)==2):
        if(pinchflag==1):
            pinchflag=0
            keyboard.release(Key.space)
        x1,y1,w1,h1=cv2.boundingRect(counto[0])
        x2,y2,w2,h2=cv2.boundingRect(counto[1])
        cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(255,0,0),2)
        cv2.rectangle(frame,(x2,y2),(x2+w2,y2+h2),(255,0,0),2)
        tx,ty,tw,th=cv2.boundingRect(np.array([[[x1,y1],[x1+w1,y1+h1],[x2,y2],[x2+w2,y2+h2]]]))
        cv2.rectangle(frame,(tx,ty),(tx+tw,ty+th),(255,0,0),2)

   # if(len(counto)==3):
        
         #x1,y1,w1,h1=cv2.boundingRect(counto[0])
         #x2,y2,w2,h2=cv2.boundingRect(counto[1])
         #x3,y3,w3,h3=cv2.boundingRect(counto[2])
         #cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(255,0,0),2)
         #cv2.rectangle(frame,(x2,y2),(x2+w2,y2+h2),(255,0,0),2)
         #cv2.rectangle(frame,(x3,y3),(x3+w3,y3+h3),(255,0,0),2)
     
        #tx,ty,tw,th=cv2.boundingRect(np.array([[[x1,y1],[x1+w1,y1+h1],[x2,y2],[x2+w2,y2+h2]]]))
        #cv2.rectangle(frame,(tx,ty),(tx+tw,ty+th),(255,0,0),2)


    if(len(counto)==1):
        x,y,w,h=cv2.boundingRect(counto[0])
        if(pinchflag==0):
            if(abs((tw*th-w*h)*100/(w*h))<30):
                pinchflag=1
                keyboard.press(Key.space)
                tx,ty,tw,th=(0,0,0,0)

            else:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                cx=x+w/2
                cy=y+h/2
                cv2.circle(frame,(cx,cy),2,(0,0,255),3)
        
    cv2.imshow('frame' , frame)
    cv2.imshow('mask' , mask)
    cv2.imshow('res' , res)
    cv2.imshow('res_2' , res_2)
    #cv2.imshow('maskFinal',maskFinal)

    if cv2.waitKey(1) & 0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()


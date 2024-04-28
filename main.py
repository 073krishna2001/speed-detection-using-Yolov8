import cv2 # opencv library
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import time
from math import dist
from tracker import*

# Load YOLO model
model=YOLO('yolov8s.pt')

# Function to handle mouse events
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)
  
# Create window and set mouse callback
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Open video capture
cap=cv2.VideoCapture('test_video.mp4')

# Read class names
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

#initialize counter
count=0

# Initialize trackers
tracker=Tracker()   #  Create Tracker object for car
tracker1=Tracker()  #  Create Tracker object for bus

# Define line positions for speed estimation
cy1=150
cy2=230
offset=8

# Initialize dictionaries to store vehicle positions and speeds
upcar={}
countercarup=[]
downcar={}
countercardown=[]

downbus={}
counterbusdown=[]
upbus={}
counterbusup=[]

# Add VideoWriter object orsaving the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1020, 550))

# Dictionary to store the previous positions of each vehicle
previous_positions = {}

# Process video frames
while True:    
    ret,frame = cap.read()
    if not ret:
        break
    # Process every third frame to optimize speed
    count += 1
    if count % 3 != 0:
        continue
    # Resize frame
    frame=cv2.resize(frame,(1020,500))
   
    # Perform object detection
    results=model.predict(frame) 
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    
    # Initialize lists for vehicle bounding boxes
    list=[] # for car
    list1=[]  # for bus
    
    # Extract bounding boxes for cars and buses
    for index,row in px.iterrows():
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        
        if 'car' in c:
           list.append([x1,y1,x2,y2])
       
        elif'bus' in c:
           list1.append([x1,y1,x2,y2])
    
    # Update car tracker
    bbox_idx=tracker.update(list)
    # Update bus tracker
    bbox1_idx=tracker1.update(list1)
    
    
###################################### carup ###############################

    # Iterate through car bounding boxes
    for bbox in bbox_idx:
        x3,y3,x4,y4,id1=bbox
        # for getting rectangle center coordinates
        cx3=int(x3+x4)//2
        cy3=int(y3+y4)//2
        
        # Draw the trail by connecting previous positions with lines
        if id1 in previous_positions:
            for prev_pos in previous_positions[id1]:
                cv2.line(frame, prev_pos, (cx3, cy3), (0, 255, 255), 1)

        # Update previous positions
        if id1 not in previous_positions:
            previous_positions[id1] = []
        previous_positions[id1].append((cx3, cy3))
        if len(previous_positions[id1]) > 10:
            previous_positions[id1].pop(0)
        
        
        if cy1<(cy3+offset) and cy1>(cy3-offset):
            upcar[id1]=(cx3,cy3)  # save the id of the car when it touches the green color line (line1)
            upcar[id1] = time.time()
        if id1 in upcar:
            if cy2<(cy3+offset) and cy2>(cy3-offset): # do futher processing when the same id car touches the red color line (line2)
                elapsed1_time = time.time() - upcar[id1]
                if(countercarup).count(id1)==0:
                    countercarup.append(id1)
                    distance1 = 10  # meters
                    a_speed_ms1 = distance1 / elapsed1_time
                    a_speed_kh1 = a_speed_ms1 * 3.6
                cv2.circle(frame,(cx3,cy3),4,(255,0,0),-1)
                cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,255),2)
                cvzone.putTextRect(frame,f'{id1}',(x3,y3),1,1)
                #cv2.putText(frame, str(id1), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, str(int(a_speed_kh1)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

##################################### cardown ###############################

        if cy2<(cy3+offset) and cy2>(cy3-offset):
            downcar[id1]=(cx3,cy3)  # save the id of the car when it touches the red color line (line2)
            downcar[id1] = time.time()
        if id1 in downcar:
            if cy1<(cy3+offset) and cy1>(cy3-offset): # do futher processing when the same id car touches the green color line (line1)
                elapsed_time = time.time() - downcar[id1]
                if(countercardown).count(id1)==0:
                    countercardown.append(id1)
                    distance = 10  # meters
                    a_speed_ms = distance / elapsed_time
                    a_speed_kh = a_speed_ms * 3.6
                cv2.circle(frame,(cx3,cy3),4,(255,0,255),-1)
                cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,0),2)
                cvzone.putTextRect(frame,f'{id1}',(x3,y3),1,1)
                #cv2.putText(frame, str(id1), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, str(int(a_speed_kh)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
        
        
##################################### up bus #######################################

    # Iterate through bus bounding boxes
    for bbox1 in bbox1_idx:
        x5,y5,x6,y6,id2=bbox1
        cx4=int(x5+x6)//2
        cy4=int(y5+y6)//2
        
        # Draw the trail by connecting previous positions with lines
        if id2 in previous_positions:
            for prev_pos in previous_positions[id2]:
                cv2.line(frame, prev_pos, (cx4, cy4), (0, 255, 255), 1)

        # Update previous positions
        if id2 not in previous_positions:
            previous_positions[id2] = []
        previous_positions[id2].append((cx4, cy4))
        if len(previous_positions[id2]) > 10:
            previous_positions[id2].pop(0)
        
        if cy1<(cy4+offset) and cy1>(cy4-offset):
            upbus[id2]=(cx4,cy4)  # save the id of the bus when it touches the green color line (line1)
            upbus[id2] = time.time()
        if id2 in upbus:
            if cy2<(cy4+offset) and cy2>(cy4-offset): # do futher processing when the same id bus touches the red color line (line2)
                elapsed2_time = time.time() - upbus[id2]
                if(counterbusup).count(id2)==0:
                    counterbusup.append(id2)
                    distance2 = 10  # meters
                    a_speed_ms2 = distance2 / elapsed2_time
                    a_speed_kh2 = a_speed_ms2 * 3.6
                cv2.circle(frame,(cx4,cy4),4,(255,0,0),-1)
                cv2.rectangle(frame,(x5,y5),(x6,y6),(255,0,255),2)
                cvzone.putTextRect(frame,f'{id2}',(x5,y5),1,1) 
                #cv2.putText(frame, str(id2), (x5, y5), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, str(int(a_speed_kh2)) + 'Km/h', (x6, y6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)             
        
##################################### down bus ###################################

        if cy2<(cy4+offset) and cy2>(cy4-offset):
            downbus[id2]=(cx4,cy4)  # save the id of the bus when it touches the red color line (line2)
            downbus[id2] = time.time()
        if id2 in downbus:
            if cy1<(cy4+offset) and cy1>(cy4-offset): # do futher processing when the same id bus touches the green color line (line1)
                elapsed3_time = time.time() - downbus[id2]
                if(counterbusdown).count(id2)==0:
                    counterbusdown.append(id2) 
                    distance3 = 10  # meters
                    a_speed_ms3 = distance3 / elapsed3_time
                    a_speed_kh3 = a_speed_ms3 * 3.6
                cv2.circle(frame,(cx4,cy4),4,(255,0,255),-1)
                cv2.rectangle(frame,(x5,y5),(x6,y6),(255,0,0),2)
                cvzone.putTextRect(frame,f'{id2}',(x5,y5),1,1)
                #cv2.putText(frame, str(id2), (x5, y5), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, str(int(a_speed_kh3)) + 'Km/h', (x6, y6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                 
    # Display lines for speed estimation        
    cv2.line(frame,(1,cy1),(1018,cy1),(0,255,0),2)
    cv2.line(frame,(3,cy2),(1016,cy2),(0,0,255),2)
    # print(upcar) ---- displaying the list of car entry in the editor.
    
    # Calculate and display vehicle counts
    cup=len(countercarup)
    #cvzone.putTextRect(frame,f'upcar:-{cup}',(50,60),2,2) # ---- display number of upcars
    cv2.putText(frame, ('Car going To Town:-') + str(cup), (680,350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
    
    cdown=len(countercardown)
    #cvzone.putTextRect(frame,f'downcar:-{cdown}',(50,100),2,2) # ---- displaying number of downcar
    cv2.putText(frame, ('Car leaving The Town:-') + str(cdown), (680,380), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
    
    cbusup=len(counterbusup)
    #cvzone.putTextRect(frame,f'busup:-{cbusup}',(790,60),2,2) # ---- display number of busup
    cv2.putText(frame, ('Bus going To Town:-') + str(cbusup), (680,430), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
    
    cbusdown=len(counterbusdown)
    #cvzone.putTextRect(frame,f'busdown:-{cbusdown}',(790,100),2,2) # ---- display number of busdown 
    cv2.putText(frame, ('Bus leaving The Town:-') + str(cbusdown), (680,460), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
    
    out.write(frame)
    
    # Show frame
    cv2.imshow("RGB", frame)
    
    # Exit on ESC key
    if cv2.waitKey(1)&0xFF==27:
        break

# Release resources  
cap.release()
cv2.destroyAllWindows()

import os 

import cv2 
import mediapipe as mp
import argparse

def process_img(img, face_detection):
    
    H, W, _ = img.shape 

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)


    if out.detections is not None: 
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box 

            x1,y1,w,h = bbox.xmin, bbox.ymin,bbox.width, bbox.height

            x1 = int(x1*W)        
            y1 = int(y1*H)
            w = int(w*W)
            h = int(h*H)    
            # Clamp coordinates to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x1 + w)
            y2 = min(H, y1 + h)

            # Only draw if valid region
            if x2 > x1 and y2 > y1:

                # Draw black filled rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)

                # Text to display
                text = "Mogged"

                # Font settings
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 2

                # Get text size
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

                # Compute centered position
                text_x = x1 + (x2 - x1 - text_width) // 2
                text_y = y1 + (y2 - y1 + text_height) // 2

                # Keep text inside image (extra safety)
                text_x = max(0, text_x)
                text_y = max(text_height, text_y)

                # Put text
                cv2.putText(img, text, (text_x, text_y),
                            font, font_scale, (255, 255, 255), thickness)
    return img

args = argparse.ArgumentParser()
args.add_argument("--mode", default ='webcam')
args.add_argument("--filepath",default=None)

args = args.parse_args()


output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir,exist_ok=True)

#read image

mp_face_detection = mp.solutions.face_detection 


with mp_face_detection.FaceDetection(model_selection= 0, min_detection_confidence = 0.5) as face_detection: 
    
    if args.mode in ["image"]:
        # read image
        img = cv2.imread(args.filepath)

        H, W, _ = img.shape
    
        img = process_img(img, face_detection)

        #save image
        cv2.imwrite(os.path.join(output_dir,'output.png'),img)


    elif args.mode in ['video']:
        
        cap = cv2.VideoCapture(args.filepath)    
        ret, frame = cap.read()

        output_video = cv2.VideoWriter(os.path.join(output_dir,'output.mp4'),
                                       cv2.VideoWriter_fourcc(*'MP4V'),
                                       25,
                                       (frame.shape[1],frame.shape[0]))

        while ret:
            frame = process_img(frame, face_detection)

            output_video.write(frame)

            ret,frame = cap.read()
            

        cap.release()
        output_video.release()

    elif args.mode in ['webcam']:
        cap = cv2.VideoCapture(0)

        ret, frame = cap.read()
        
        while ret:
            frame = process_img(frame,face_detection)

            cv2.imshow('Blur',frame)
            cv2.waitKey(25)      

            ret, frame = cap.read()
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()    

    #cv2.imshow('Mogged',img)
    #cv2.waitKey(0)



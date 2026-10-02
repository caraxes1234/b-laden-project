import cv2
import time
import numpy as np
recognizer=cv2.FaceRecognizerSF.create(
    "models/face_recognition_sface_2021dec.onnx", "" 
)
face=cv2.imread("Screenshot 2026-09-11 191313.png")
cap=cv2.VideoCapture(0)

hf,wf=face.shape[:2]
fcount=0
detector=cv2.FaceDetectorYN.create(
  "models/face_detection_yunet_2023mar.onnx", "", (wf,hf)
)

retval,face_t=detector.detect(face)
if face_t is not None:
    aligned_face=recognizer.alignCrop(face,face_t[0:1])
    embedding_face=recognizer.feature(aligned_face)
else:
    print("couldnt detect face in reference photo ")
    exit()
fps=0
start=time.time()
while True:
    face_entry=[]
    ret,crowed_img=cap.read()
    if not ret:
        exit()
    fcount=fcount+1
    hc,wc=crowed_img.shape[:2]
    detector.setInputSize((wc,hc))
    retlev_c,face_c=detector.detect(crowed_img)
    if face_c is not None:
        for f in face_c:
            x,y,w,h=np.int32(f[:4])
            f_p=f[None,:]
            aligned_cface=recognizer.alignCrop(crowed_img,f_p)
            embedding_faceC=recognizer.feature(aligned_cface)
            cosinValue=recognizer.match(embedding_face,embedding_faceC,cv2.FaceRecognizerSF_FR_COSINE)
            face_ent={
                "Box":np.int32(f_p[0][:4]),
                "value":cosinValue
                
            }
            face_entry.append(face_ent)
    max_val=0.40
    best_box=None
    for i in face_entry:
        current_box = i["Box"]   
        current_val = i["value"]
       
        if max_val <current_val:
            best_box=i["Box"]
            max_val=i["value"]

    if max_val >0.40 and best_box is not None:
        cx,cy,cw,ch=best_box
        
        points=np.array([
             [cx, cy],          # Top-Left
            [cx + cw, cy],      # Top-Right
            [cx + cw, cy + ch],  # Bottom-Right
            [cx, cy + ch]
        ],dtype=np.int32)
        center=np.mean(points,axis=0)
        target_x=int(center[0])
        target_y=int(center[1])
        frame_y,frame_x=crowed_img.shape[:2]
        frame_cx=frame_x//2
        frame_cy=frame_y//2
        error_X=int(target_x-frame_cx)
        error_Y=int(target_y-frame_cy)
        points=points.reshape(-1,1,2)
        cv2.polylines(crowed_img,[points],True,(0,255,0),2)
        cv2.putText(crowed_img,f"{error_X},{error_Y}",(target_x,target_y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,2.0,(0,255,0),2)
    if time.time()-start>=1:
                    fps=fcount
                    start=time.time()
                    fcount=0
    cv2.putText(crowed_img, f"FPS: {fps}",
                                    (100, 190),cv2.FONT_HERSHEY_PLAIN,1.0,(0,255,0),1)
    cv2.imshow("j",crowed_img)
   
    if cv2.waitKey(1) & 0xFF==ord('q'):
       break
    
cap.release()
cv2.destroyAllWindows()

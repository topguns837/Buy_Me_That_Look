import cv2
import mediapipe as mp
import numpy as np



class PoseEstimation :
    def __init__(self) :
        
        self.BG_COLOR = (192, 192, 192) 
        self.IMAGE_FILES = []
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        self.init__mp_pose()
        

    def init__mp_pose(self) :

        with self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.9) as pose:

            for idx, file in enumerate(self.IMAGE_FILES):
                image = cv2.imread(file)
                image_height, image_width, _ = image.shape
                # Convert the BGR image to RGB before processing.
                results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
                if not results.pose_landmarks:
                  continue
                print(
                    f'Nose coordinates: ('
                    f'{results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].x * image_width}, '
                    f'{results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].y * image_height})'
                )
    
                annotated_image = image.copy()
                # Draw segmentation on the image.
                # To improve segmentation around boundaries, consider applying a joint
                # bilateral filter to "results.segmentation_mask" with "image".
                condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                bg_image = np.zeros(image.shape, dtype=np.uint8)
                bg_image[:] = self.BG_COLOR
                annotated_image = np.where(condition, annotated_image, bg_image)
                # Draw pose landmarks on the image.
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
                cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
                # Plot pose world landmarks.
                self.mp_drawing.plot_landmarks(
                    results.pose_world_landmarks, self.mp_pose.POSE_CONNECTIONS)

    def findPosition(self,img, results ,draw=True):

        self.lmList = []
        if results.pose_landmarks:
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def is_full_shot(self) :
        #self.image = image
        
        try :       
            if 0<= self.lmlist[1][1] <= self.image.shape[1] and 0<= self.lmlist[29][1] <= self.image.shape[1] and 0<= self.lmlist[30][1] <= self.image.shape[1]  :
                if 0<= self.lmlist[1][2] <= self.image.shape[0] and 0<= self.lmlist[29][2] <= self.image.shape[0] and 0<= self.lmlist[30][2] <= self.image.shape[0] :
                    if abs(self.lmlist[1][2] - self.lmlist[29][2]) > self.image.shape[1] and abs(self.lmlist[1][2] - self.lmlist[30][2]) > self.image.shape[1]  : 
                        return True
                
        except :
            pass

        return False    

    def driver(self, img_path) :

        with self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

            self.image = cv2.imread(img_path)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            self.image.flags.writeable = False
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            results = pose.process(self.image)

            self.lmlist = self.findPosition(self.image, results)

            
            # Draw the pose annotation on the image.
            self.image.flags.writeable = True
            self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
            self.mp_drawing.draw_landmarks(
                self.image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
            # Flip the image horizontally for a selfie-view display.
            #cv2.imshow('MediaPipe Pose', self.image)
            #if cv2.waitKey(0) & 0xFF == 27:
              #exit()

            
            if self.is_full_shot() :
                return True
            else:
                return False
            

            

    
if __name__ == "__main__" :
    ps = PoseEstimation()
    img_path = 'Web_Scraping/Sample_Images/sample2.jpg'
    result = ps.driver(img_path)
    print(result)
    print(ps.lmlist)
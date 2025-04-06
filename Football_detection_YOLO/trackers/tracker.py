from ultralytics import YOLO
import supervision as sv
import pickle
import numpy as np
import torch
import os
from utils import get_center_of_bbox, get_bbox_width, get_foot_position
import cv2
import sys
sys.path.append('/..')  # Add the parent directory to the path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker=sv.ByteTrack()  # Load a pretrained YOLOv8 model
    def detect_frames(self, frames):
        batch_size = 8  
        detections = []
        for frame in range(0, len(frames), batch_size):
            batch = frames[frame:frame + batch_size]
            detections_batch = self.model.predict(batch, conf=0.1)
            detections+= detections_batch
            
            
        return detections 


    def get_object_track(self, frames,read_from_stubs=False,stub_path=None):
        if read_from_stubs and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
            
        detections = self.detect_frames(frames)##detections stored in the list for all the frames
        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }
        for frame_num,detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inverse = {v: k for k, v in cls_names.items()} #maps class id to class name
            # convert to supervision detections
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            #convert goalkeeper to 0 and player to 1
            for object_ind, class_id in enumerate(detection_supervision.class_id):##into the supervision format
                if cls_names[class_id] == "goalkeeper":
                      detection_supervision.class_id[object_ind] = cls_names_inverse["player"]
            
            ##tracker
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            #print(detection_with_tracks)
            
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            for frame_detection in detection_with_tracks:
                 # Check the type of the object

                bbox = frame_detection[0].tolist()
                print(type(bbox))  # This will output: <class 'list'>

                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                if cls_id == cls_names_inverse['player']:
                    tracks["players"][frame_num][track_id] = {"bbox" : bbox}
                if cls_id == cls_names_inverse['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox" : bbox}    
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                if cls_id == cls_names_inverse['ball']:
                    tracks["ball"][frame_num][1] = {"bbox" : bbox}   
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
                
        return tracks
    
    ##annotations
    def draw_ellipse(self, frame, bbox, color=(0, 255, 0),track_id=None,):
        y2 = int(bbox[3])       
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        cv2.ellipse(frame, (x_center, y2),
                axes=(int(width / 2), int(0.35*width)),
                      angle=0,
                      startAngle=0,
                      endAngle=235,
                      color=color,
                      thickness=2,
                      lineType=cv2.LINE_4)
        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )
        return frame
    def add_position_to_tracks(sekf,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position
    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame
                  
    def draw_annotations(self, video_frames, tracks):
        output_video_frames=[]
        for frame_num, frame in enumerate(video_frames):
            frame=frame.copy()
            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            for track_id, player in player_dict.items():
                # color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"],(0,255,0), track_id)

                # if player.get('has_ball',False):
                #     frame = self.draw_traingle(frame, player["bbox"],(0,0,255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))
            output_video_frames.append(frame)    
        return output_video_frames   
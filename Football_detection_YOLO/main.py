from utils import read_video, save_video
from trackers import Tracker
from cam_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed import SpeedAndDistance_Estimator
def main():
    input_video = 'input_videos/08fd33_4.mp4'
    output_video = 'output_video/output.avi'

    # Read video frames
    video_frames = read_video(input_video)
      
    # initialize the tracker
    model_path = r"C:\Users\91823\Downloads\archive\footballDetection\models\best (4).pt"

    tracker = Tracker(model_path)
    tracks = tracker.get_object_track(video_frames,read_from_stubs=True,stub_path='stubs/track_stubs.pkl')
    tracker.add_position_to_tracks(tracks)
    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    # speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    
    #draw output
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)
    # Save processed video frames

    save_video(output_video_frames, output_video)
if __name__ == '__main__':
    main()
# This script reads a video, processes it, and saves the output video.    
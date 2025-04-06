from utils import read_video, save_video
from trackers import Tracker
def main():
    input_video = 'input_videos/08fd33_4.mp4'
    output_video = 'output_video/output.avi'

    # Read video frames
    video_frames = read_video(input_video)
      
    # initialize the tracker
    model_path = r"C:\Users\91823\Downloads\archive\footballDetection\models\best (4).pt"

    tracker = Tracker(model_path)

    tracks = tracker.get_object_track(video_frames,read_from_stubs=True,stubs_path='stubs/track_stubs.pkl')
    #draw output
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    # Save processed video frames

    save_video(output_video_frames, output_video)
if __name__ == '__main__':
    main()
# This script reads a video, processes it, and saves the output video.    
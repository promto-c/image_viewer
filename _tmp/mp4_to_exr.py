import cv2
import os

def convert_mp4_to_exr(video_path, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {frame_count}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to EXR format and save
        exr_filename = os.path.join(output_dir, f"frame.{frame_idx:04d}.exr")
        cv2.imwrite(exr_filename, frame.astype('float32'))
        print(f"Saved: {exr_filename}")
        frame_idx += 1

    # Release the video capture object
    cap.release()
    print("Conversion complete!")

# Example usage
video_path = 'path_to_your_video.mp4'
output_dir = 'path_to_output_directory'
convert_mp4_to_exr(video_path, output_dir)

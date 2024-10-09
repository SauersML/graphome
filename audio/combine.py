import argparse
from moviepy.editor import VideoFileClip, AudioFileClip
import os

def combine_audio_video(audio_file_path, video_file_path, output_file_path):
    """Combine the specified audio and video files by adjusting the video speed to match the audio duration."""
    if not os.path.exists(audio_file_path) or not os.path.exists(video_file_path):
        print("Audio or video file not found.")
        return

    # Load the video and audio clips
    video_clip = VideoFileClip(video_file_path)
    audio_clip = AudioFileClip(audio_file_path)

    # Adjust video speed to match the audio duration
    video_duration = video_clip.duration
    audio_duration = audio_clip.duration
    speed_factor = video_duration / audio_duration

    # Speed up or slow down the video to match the audio duration
    adjusted_video_clip = video_clip.fx(VideoFileClip.speedx, final_duration=audio_duration)

    # Set the adjusted audio to the video
    final_clip = adjusted_video_clip.set_audio(audio_clip)

    # Write the final video with audio
    final_clip.write_videofile(output_file_path, codec="libx264", audio_codec="aac")

    print(f"Combined video and audio file saved as {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine audio and video by adjusting video speed to match audio")
    parser.add_argument('--audio', required=True, help="Path to the audio file (wav)")
    parser.add_argument('--video', required=True, help="Path to the video file (mp4)")
    parser.add_argument('--output', required=True, help="Path for the output video file")

    args = parser.parse_args()

    # Call the function with provided arguments
    combine_audio_video(args.audio, args.video, args.output)

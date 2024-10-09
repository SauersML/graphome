import argparse
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_audioclips
import os

def combine_audio_video(audio_file_path, video_file_path, output_file_path):
    """Combine the specified audio and video files into one output video file."""
    if not os.path.exists(audio_file_path) or not os.path.exists(video_file_path):
        print("Audio or video file not found.")
        return

    # Load the video and audio clips
    video_clip = VideoFileClip(video_file_path)
    audio_clip = AudioFileClip(audio_file_path)

    # Calculate the number of times to loop the audio (should be no loops)
    loops_required = int(video_clip.duration // audio_clip.duration) + 1
    print(loops_required)
    audio_clips = [audio_clip] * loops_required

    # Make the looped audio
    looped_audio_clip = concatenate_audioclips(audio_clips)

    # Set the looped audio to the video
    final_clip = video_clip.set_audio(looped_audio_clip.subclip(0, video_clip.duration))

    # Write the final video with audio
    final_clip.write_videofile(output_file_path, codec="libx264", audio_codec="aac")

    print(f"Combined video and audio file saved as {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine audio and video into one output file")
    parser.add_argument('--audio', required=True, help="Path to the audio file (wav)")
    parser.add_argument('--video', required=True, help="Path to the video file (mp4)")
    parser.add_argument('--output', required=True, help="Path for the output video file")

    args = parser.parse_args()

    # Call the function with provided arguments
    combine_audio_video(args.audio, args.video, args.output)

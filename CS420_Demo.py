import streamlit as st
import os
from pytube import YouTube
import tempfile
from moviepy.video.io.VideoFileClip import VideoFileClip

import openai
import whisper
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz

import subprocess

import random
import subprocess
import tempfile

from yt_dlp import YoutubeDL

import streamlit as st
import tempfile

openai.api_key = ""


def summarize_transcript_with_timestamps(video_path):
    """Summarize a given transcript using GPT-4o and extract timestamps for the summary."""

    whisper_model = whisper.load_model("base")

    def transcribe_video(video_path):
        """Transcribe a video file using Whisper with timestamps."""
        print(f"Transcribing video: {video_path}")
        result = whisper_model.transcribe(video_path, word_timestamps=True)
        segments = result['segments']
        print("Transcript with timestamps extracted.")
        return segments

    segments = transcribe_video(video_path)

    transcript = " ".join([segment['text'] for segment in segments])

    prompt = (
        f"You are a highly intelligent assistant tasked with summarizing transcripts. "
        f"Please provide an extractive summary using exact sentences from the provided text, "
        f"sorted in chronological order, and has the same language as the transcripts. Ensure the summary is concise and captures the key points. "
        f"Transcript:\n{transcript}\n\n"
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert assistant summarizing transcripts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return None, None, None

    summary_text = response.choices[0].message.content.strip()

    summarized_timestamps = []
    summary_sentences = summary_text.split('. ')

    min_score = 75

    for i, summary_sentence in enumerate(summary_sentences):
        best_score = -1
        best_match_idx = -1
        
        for idx, segment in enumerate(segments):
            score = fuzz.partial_ratio(summary_sentence, segment['text'])
            if score > best_score:
                best_score = score
                best_match_idx = idx
        
        if best_score >= min_score and best_match_idx != -1:
            best_segment = segments[best_match_idx]
            summarized_timestamps.append({
                "start": best_segment['start'],
                "end": best_segment['end'],
                "text": best_segment['text']
            })
        else:
            print(f"Sentence not matched with sufficient score ({best_score}): {summary_sentence}")

    return transcript, summary_text, summarized_timestamps

def create_summary_video(input_video_path, output_video_path, timestamps, temp_dir="D:/CS420/LLM/temp_segments"):
    """
    Create a summary video using FFmpeg based on provided timestamps.

    Parameters:
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path where the summarized video will be saved.
        timestamps (list): List of dictionaries containing 'start' and 'end' times for segments.
        temp_dir (str): Temporary directory to store extracted video segments.
    """
    timestamps = sorted(timestamps, key=lambda x: x['start'])

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    segment_files = []

    for i, segment in enumerate(timestamps):
        start_time = segment['start']
        end_time = segment['end']

        segment_file = os.path.join(temp_dir, f"segment_{i + 1}.mp4")

        command = [
            'ffmpeg', '-i', input_video_path, 
            '-ss', str(start_time), '-to', str(end_time), 
            '-c:v', 'libx264', '-c:a', 'aac', '-strict', 'experimental', '-y', segment_file
        ]

        subprocess.run(command, check=True)
        segment_files.append(os.path.abspath(segment_file))

    concat_file_path = os.path.join(temp_dir, "segments.txt")
    with open(concat_file_path, 'w') as f:
        for segment_file in segment_files:
            f.write("file '{}'\n".format(segment_file.replace("\\", "/")))

    if not output_video_path.endswith(".mp4"):
        output_video_path += ".mp4"

    concat_command = [
        'ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_file_path, 
        '-c:v', 'libx264', '-c:a', 'aac', '-strict', 'experimental', '-y', output_video_path
    ]

    subprocess.run(concat_command, check=True)

    for segment_file in segment_files:
        os.remove(segment_file)

    os.remove(concat_file_path)
    print(f"Summary video saved to {output_video_path}")


def add_random_segments(input_video, output_video, summary_timestamps, segment_percent=0.20, max_segment_percent=0.3):
    """
    Thêm các đoạn ngẫu nhiên vào video tóm tắt và tự động gộp các đoạn chồng lấn.

    Tham số:
        input_video (str): Đường dẫn đến file video gốc.
        output_video (str): Đường dẫn nơi lưu video tóm tắt đã được tạo.
        summary_timestamps (list): Danh sách chứa các đoạn tóm tắt ban đầu với 'start' và 'end'.
        segment_percent (float): Phần trăm tổng thời lượng video cần tóm tắt.
        max_segment_percent (float): Phần trăm tối đa cho mỗi đoạn ngẫu nhiên.
    
    Trả về:
        merged_segments (list): Danh sách các đoạn video sau khi thêm các đoạn ngẫu nhiên và gộp.
    """
    
    def get_video_duration(video_path):
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return float(result.stdout.strip())

    def extract_segment(input_video, start_time, duration, output_path):
        subprocess.run([  
            "ffmpeg", "-y", "-ss", str(start_time), "-i", input_video,
            "-t", str(duration), "-c", "copy", output_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    def merge_segments(segments):
        segments.sort(key=lambda x: x['start'])
        merged_segments = []
        current = segments[0]

        for segment in segments[1:]:
            if current['end'] >= segment['start']:
                current['end'] = max(current['end'], segment['end'])
            else:
                merged_segments.append(current)
                current = segment

        merged_segments.append(current)
        return merged_segments

    total_duration = get_video_duration(input_video)
    if total_duration <= 0:
        raise ValueError("Thời lượng video không hợp lệ.")

    target_duration = total_duration * segment_percent
    max_segment_duration = total_duration * max_segment_percent

    covered_duration = sum(segment['end'] - segment['start'] for segment in summary_timestamps)

    with tempfile.TemporaryDirectory() as temp_dir:
        segments = summary_timestamps[:]
        remaining_duration = target_duration - covered_duration

        while remaining_duration > 0:
            start_time = random.uniform(0, total_duration)
            segment_duration = random.uniform(1, min(max_segment_duration, remaining_duration))

            if start_time + segment_duration > total_duration:
                segment_duration = total_duration - start_time

            segments.append({'start': start_time, 'end': start_time + segment_duration})
            remaining_duration -= segment_duration

        merged_segments = merge_segments(segments)

        concat_file_path = os.path.join(temp_dir, "segments.txt")
        with open(concat_file_path, 'w') as f:
            for i, segment in enumerate(merged_segments):
                segment_output = os.path.join(temp_dir, f"segment_{i}.mp4")
                extract_segment(input_video, segment['start'], segment['end'] - segment['start'], segment_output)
                f.write("file '{}'\n".format(segment_output.replace("\\", "/")))

        subprocess.run([  
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_file_path, "-c:v", "libx264", "-c:a", "aac",
            "-strict", "experimental", "-b:a", "192k", output_video
        ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    return merged_segments

def download_youtube_video(youtube_url, download_path="D:/CS420/LLM/temp_download"):
    """
    Download a YouTube video using yt-dlp and save it as an .mp4 file.
    
    Parameters:
        youtube_url (str): URL of the YouTube video to download.
        download_path (str): Path to save the downloaded video.
    
    Returns:
        tuple: (str, str) - Title of the video and path to the downloaded video file.
    """
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4',
        'outtmpl': os.path.join(download_path, '%(title)s.mp4'),
    }
    
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=True)
            video_title = info_dict.get('title', 'Unknown Title')
            video_path = ydl.prepare_filename(info_dict)
        
        return video_title, video_path
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

st.title("YouTube Video Summarizer")

youtube_url = st.text_input("Enter the YouTube URL:", "")

if youtube_url:
    with st.spinner("Downloading video..."):
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                title, video_path = download_youtube_video(youtube_url)
                st.success("Video downloaded successfully!")

                st.video(video_path)

                with st.spinner("Processing video and generating summary..."):
                    transcript, summary_text, timestamps = summarize_transcript_with_timestamps(video_path)

                    summarized_video_path = os.path.join(temp_dir, "summarized_video.mp4")
                    create_summary_video(video_path, summarized_video_path, timestamps)

                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.subheader("Summarized Video")
                        st.video(summarized_video_path)

                        st.markdown(f"**Video Title:** {title}")

                    with col2:
                        st.subheader("Summarized Text")
                        st.write(summary_text)

            except Exception as e:
                st.error(f"An error occurred: {e}")

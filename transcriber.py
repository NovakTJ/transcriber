import os
import logging
import asyncio
import time
import json
from datetime import datetime
from uuid import uuid4 # Still useful for unique temp filenames, though not for user IDs

# Install necessary libraries if not already present
# pip install yt-dlp faster-whisper

try:
    import yt_dlp
    from faster_whisper import WhisperModel
except ImportError:
    print("Required libraries not found. Please install them using pip:")
    print("pip install yt-dlp faster-whisper")
    exit()

# --- Configuration and Initialization ---

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base directory for storing transcripts
TRANSCRIPTS_OUTPUT_DIR = "transcripts"

# --- Helper Functions ---

def download_youtube_audio(url: str, output_path: str = "audio_temp"):
    """
    Downloads the audio track from a YouTube video.

    Args:
        url (str): The URL of the YouTube video.
        output_path (str): Directory to save the audio file.

    Returns:
        tuple: (audio_filepath, video_title) or (None, None) if download fails.
    """
    logger.info(f"Attempting to download audio from: {url}")
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_path, '%(id)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True, # Only extract info, not download immediately
        'logger': logger,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            video_title = info_dict.get('title', 'Unknown Title')
            video_id = info_dict.get('id')

            if not video_id:
                logger.error(f"Could not get video ID for URL: {url}")
                return None, None

            # Actual download
            ydl_opts['outtmpl'] = os.path.join(output_path, f'{video_id}.%(ext)s')
            ydl.download([url])

            audio_filepath = os.path.join(output_path, f"{video_id}.mp3")
            if os.path.exists(audio_filepath):
                logger.info(f"Successfully downloaded audio for '{video_title}' to {audio_filepath}")
                return audio_filepath, video_title
            else:
                logger.error(f"Downloaded audio file not found at expected path: {audio_filepath}")
                return None, None

    except yt_dlp.DownloadError as e:
        logger.error(f"Failed to download audio from {url}: {e}")
        return None, None
    except Exception as e:
        logger.error(f"An unexpected error occurred during download for {url}: {e}")
        return None, None

def transcribe_audio_with_faster_whisper(audio_filepath: str, language: str = "sr"):
    """
    Transcribes an audio file using faster-whisper.

    Args:
        audio_filepath (str): Path to the audio file.
        language (str): The language of the audio (e.g., "sr" for Serbian).

    Returns:
        tuple: (full_transcript_text, list_of_segments) or (None, None) if transcription fails.
               Each segment is a dict: {'start': float, 'end': float, 'text': str}
    """
    logger.info(f"Starting transcription for {audio_filepath} in {language}...")
    try:
        # Load the model. "large-v3" is recommended for accuracy.
        # device="cuda" uses GPU, compute_type="float16" for faster inference on modern GPUs.
        # If no GPU, use device="cpu" and compute_type="int8" or "float32".
        model = WhisperModel("large-v3", device="cuda", compute_type="float16")

        segments_generator, info = model.transcribe(audio_filepath, language=language, beam_size=5)

        full_transcript_text = ""
        segments_list = []
        for segment in segments_generator:
            segment_data = {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            }
            segments_list.append(segment_data)
            full_transcript_text += segment.text + " " # Add space for concatenation

        logger.info(f"Transcription complete. Detected language: {info.language} ({info.language_probability:.2f})")
        return full_transcript_text.strip(), segments_list

    except Exception as e:
        logger.error(f"Failed to transcribe {audio_filepath}: {e}")
        return None, None

def get_playlist_videos(playlist_url: str):
    """
    Extracts video URLs and metadata from a YouTube playlist.
    Derives the class name from the playlist title.

    Args:
        playlist_url (str): The URL of the YouTube playlist.

    Returns:
        tuple: (class_name, list_of_video_dicts) or (None, []) if extraction fails.
               Each video_dict: {'url': str, 'title': str, 'order': int}
    """
    logger.info(f"Extracting videos from playlist: {playlist_url}")
    ydl_opts = {
        'extract_flat': True,  # Get only top-level info, not download videos
        'force_generic_extractor': True, # Ensure it processes as a generic URL
        'quiet': True,
        'no_warnings': True,
        'logger': logger,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(playlist_url, download=False)

            if 'entries' not in info_dict or not info_dict['entries']:
                logger.warning(f"No videos found in playlist or failed to extract entries for {playlist_url}")
                return None, []

            playlist_title = info_dict.get('title', 'Unknown Playlist').strip()
            # Derive class name: all words from the title except for the last one
            title_parts = playlist_title.split()
            class_name = " ".join(title_parts[:-1]) if len(title_parts) > 1 else playlist_title
            # Sanitize class_name for directory and file naming
            class_name = "".join(c for c in class_name if c.isalnum() or c.isspace()).strip()
            class_name = class_name.replace(' ', '_')


            video_list = []
            for i, entry in enumerate(info_dict['entries']):
                if entry and entry.get('url') and entry.get('title'):
                    video_list.append({
                        'url': entry['url'],
                        'title': entry['title'],
                        'order': i + 1  # 1-based indexing for order
                    })
            logger.info(f"Found {len(video_list)} videos for class '{class_name}' from playlist '{playlist_title}'.")
            return class_name, video_list

    except yt_dlp.DownloadError as e:
        logger.error(f"Failed to extract playlist info from {playlist_url}: {e}")
        return None, []
    except Exception as e:
        logger.error(f"An unexpected error occurred during playlist extraction for {playlist_url}: {e}")
        return None, []

def save_transcript_as_markdown(video_url: str, video_title: str, full_transcript: str, segments: list, class_name: str, video_order: int):
    """
    Saves the transcript and metadata as a Markdown file.

    Args:
        video_url (str): The URL of the original YouTube video.
        video_title (str): The title of the video.
        full_transcript (str): The complete transcribed text.
        segments (list): A list of timestamped segments.
        class_name (str): The name of the university class.
        video_order (int): The order of the video in its playlist.
    """
    logger.info(f"Attempting to save transcript for '{video_title}' (Class: {class_name}, Order: {video_order}) as Markdown...")
    
    # Sanitize class_name for directory and file naming
    sanitized_class_name = "".join(c for c in class_name if c.isalnum() or c.isspace()).strip()
    sanitized_class_name = sanitized_class_name.replace(' ', '_')

    # Create class-specific directory
    class_output_dir = os.path.join(TRANSCRIPTS_OUTPUT_DIR, sanitized_class_name)
    os.makedirs(class_output_dir, exist_ok=True)

    # Sanitize video title for filename
    sanitized_video_title = "".join(c for c in video_title if c.isalnum() or c.isspace()).strip()
    sanitized_video_title = sanitized_video_title.replace(' ', '_')

    # Construct filename
    filename = f"{sanitized_class_name}_video_{video_order}_{sanitized_video_title}.md"
    filepath = os.path.join(class_output_dir, filename)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# Class: {class_name}\n\n")
            f.write(f"## Video {video_order}: {video_title}\n\n")
            f.write(f"**Source URL:** {video_url}\n\n")
            f.write(f"**Transcription Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            f.write("### Transcript with Timestamps:\n\n")
            for segment in segments:
                # Format timestamp as HH:MM:SS
                start_time = time.strftime('%H:%M:%S', time.gmtime(segment['start']))
                end_time = time.strftime('%H:%M:%S', time.gmtime(segment['end']))
                f.write(f"**[{start_time}-{end_time}]** {segment['text'].strip()}\n\n")
            
            f.write("---\n\n")
            f.write("### Full Transcript:\n\n")
            f.write(full_transcript)

        logger.info(f"Successfully saved transcript to: {filepath}")
    except Exception as e:
        logger.error(f"Failed to save transcript to {filepath}: {e}")

async def process_video_url(url: str, video_title: str, class_name: str, video_order: int):
    """
    Orchestrates the download, transcription, and saving as Markdown for a single video URL.
    Includes class name and video order for Markdown file naming.
    """
    logger.info(f"Processing video {video_order} for class '{class_name}': {url} - '{video_title}'")
    audio_dir = "audio_temp"
    os.makedirs(audio_dir, exist_ok=True)

    downloaded_audio_filepath, actual_video_title = download_youtube_audio(url, audio_dir)

    if downloaded_audio_filepath and actual_video_title:
        full_transcript, segments = transcribe_audio_with_faster_whisper(downloaded_audio_filepath, language="sr")

        if full_transcript and segments:
            save_transcript_as_markdown(url, actual_video_title, full_transcript, segments, class_name, video_order)
        else:
            logger.error(f"Transcription failed for {url}. Skipping saving.")
        
        # Clean up the downloaded audio file
        try:
            os.remove(downloaded_audio_filepath)
            logger.info(f"Cleaned up temporary audio file: {downloaded_audio_filepath}")
        except OSError as e:
            logger.warning(f"Error cleaning up audio file {downloaded_audio_filepath}: {e}")
    else:
        logger.error(f"Failed to download audio for {url}. Skipping transcription and saving.")

# --- Main Execution ---

async def main():
    # Replace this with your actual list of YouTube playlist URLs
    playlist_urls = [
        "https://www.youtube.com/playlist?list=PL_EXAMPLE_PLAYLIST_ID_1", # Example: "Introduction to Programming Lectures Fall 2025"
        "https://www.youtube.com/playlist?list=PL_EXAMPLE_PLAYLIST_ID_2", # Example: "Advanced Algorithms Spring 2025"
        # Add more playlist URLs here
    ]

    # Example: If you have a file with playlist URLs, you can load them
    # try:
    #     with open("youtube_playlist_urls.txt", "r") as f:
    #         playlist_urls = [line.strip() for line in f if line.strip()]
    #     logger.info(f"Loaded {len(playlist_urls)} playlist URLs from youtube_playlist_urls.txt")
    # except FileNotFoundError:
    #     logger.warning("youtube_playlist_urls.txt not found. Using hardcoded example playlist URLs.")

    if not playlist_urls:
        logger.warning("No YouTube playlist URLs provided. Exiting.")
        return

    all_video_tasks = []
    for playlist_url in playlist_urls:
        class_name, videos_in_playlist = get_playlist_videos(playlist_url)
        if class_name and videos_in_playlist:
            for video_info in videos_in_playlist:
                all_video_tasks.append(
                    process_video_url(
                        video_info['url'],
                        video_info['title'],
                        class_name,
                        video_info['order']
                    )
                )
        else:
            logger.error(f"Could not retrieve videos for playlist: {playlist_url}. Skipping.")

    if all_video_tasks:
        logger.info(f"Starting transcription for a total of {len(all_video_tasks)} videos across all playlists.")
        await asyncio.gather(*all_video_tasks)
        logger.info(f"All video processing tasks completed. Transcripts saved to '{TRANSCRIPTS_OUTPUT_DIR}' directory.")
    else:
        logger.info("No videos found to process from the provided playlists.")

if __name__ == "__main__":
    # Run the asynchronous main function
    asyncio.run(main())


"""
Audio generation for chess game videos.

Generates move and capture sound effects for each position.
"""

import io
import logging
import struct
import tempfile
import math
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def generate_tone(frequency: float, duration_ms: int, sample_rate: int = 44100, volume: float = 0.3) -> bytes:
    """
    Generate a simple sine wave tone as raw PCM audio.

    Args:
        frequency: Tone frequency in Hz
        duration_ms: Duration in milliseconds
        sample_rate: Audio sample rate
        volume: Volume level (0.0 to 1.0)

    Returns:
        Raw PCM audio bytes (16-bit signed, mono)
    """
    num_samples = int(sample_rate * duration_ms / 1000)
    samples = []

    for i in range(num_samples):
        # Generate sine wave with fade out
        t = i / sample_rate
        fade = 1.0 - (i / num_samples) ** 0.5  # Square root fade out
        value = math.sin(2 * math.pi * frequency * t) * volume * fade
        # Convert to 16-bit signed integer
        sample = int(value * 32767)
        samples.append(struct.pack('<h', sample))

    return b''.join(samples)


def generate_move_sound(sample_rate: int = 44100) -> bytes:
    """Generate a soft 'click' sound for piece moves."""
    # Short, soft click sound - two quick tones
    tone1 = generate_tone(800, 30, sample_rate, 0.2)
    tone2 = generate_tone(600, 20, sample_rate, 0.15)
    silence = b'\x00\x00' * int(sample_rate * 0.01)  # 10ms silence
    return tone1 + silence + tone2


def generate_capture_sound(sample_rate: int = 44100) -> bytes:
    """Generate a sharper 'thud' sound for captures."""
    # Sharper, more impactful sound
    tone1 = generate_tone(400, 40, sample_rate, 0.4)
    tone2 = generate_tone(200, 60, sample_rate, 0.3)
    return tone1 + tone2


def generate_check_sound(sample_rate: int = 44100) -> bytes:
    """Generate a sharp, rising sound for checks."""
    # Sharp ascending tones to indicate danger
    tone1 = generate_tone(600, 30, sample_rate, 0.35)
    tone2 = generate_tone(900, 40, sample_rate, 0.4)
    tone3 = generate_tone(1100, 30, sample_rate, 0.3)
    return tone1 + tone2 + tone3


def generate_checkmate_sound(sample_rate: int = 44100) -> bytes:
    """Generate a dramatic, final sound for checkmate."""
    # Deep, dramatic tones followed by a triumphant flourish
    tone1 = generate_tone(200, 60, sample_rate, 0.5)
    tone2 = generate_tone(300, 50, sample_rate, 0.45)
    tone3 = generate_tone(400, 40, sample_rate, 0.4)
    tone4 = generate_tone(600, 80, sample_rate, 0.5)
    return tone1 + tone2 + tone3 + tone4


def generate_audio_track(
    moves_data: list[tuple[str, int]],  # List of (move_type, position_index)
    frame_duration_ms: int = 1000,
    sample_rate: int = 44100,
    audio_offset_ms: int = -50,  # Negative = sound plays earlier
) -> bytes:
    """
    Generate a complete audio track for a chess game.

    Args:
        moves_data: List of tuples (move_type, position_index)
                   where move_type is "move", "capture", "check", or "checkmate"
        frame_duration_ms: Duration of each frame in milliseconds
        sample_rate: Audio sample rate
        audio_offset_ms: Offset for audio timing (negative = earlier, positive = later)

    Returns:
        Raw PCM audio bytes for the entire track
    """
    # Pre-generate sounds
    sounds = {
        "move": generate_move_sound(sample_rate),
        "capture": generate_capture_sound(sample_rate),
        "check": generate_check_sound(sample_rate),
        "checkmate": generate_checkmate_sound(sample_rate),
    }

    # Calculate total duration
    total_frames = len(moves_data) + 1  # +1 for initial position
    total_samples = int(sample_rate * total_frames * frame_duration_ms / 1000)

    # Create silent audio buffer
    audio_buffer = bytearray(total_samples * 2)  # 2 bytes per sample (16-bit)

    # Place sounds at appropriate positions
    for move_type, position_idx in moves_data:
        # Sound should play slightly before the frame appears for better perceived sync
        frame_time_ms = position_idx * frame_duration_ms + audio_offset_ms
        start_sample = max(0, int(sample_rate * frame_time_ms / 1000))

        sound = sounds.get(move_type, sounds["move"])
        sound_samples = len(sound) // 2

        # Mix sound into buffer
        for i in range(min(sound_samples, total_samples - start_sample)):
            buffer_idx = (start_sample + i) * 2
            if buffer_idx + 1 < len(audio_buffer):
                # Read existing sample
                existing = struct.unpack('<h', audio_buffer[buffer_idx:buffer_idx+2])[0]
                # Read new sample
                new = struct.unpack('<h', sound[i*2:i*2+2])[0]
                # Mix (simple addition with clipping)
                mixed = max(-32768, min(32767, existing + new))
                # Write back
                audio_buffer[buffer_idx:buffer_idx+2] = struct.pack('<h', mixed)

    return bytes(audio_buffer)


def create_wav_file(pcm_data: bytes, sample_rate: int = 44100) -> bytes:
    """
    Create a WAV file from raw PCM data.

    Args:
        pcm_data: Raw PCM audio bytes (16-bit signed, mono)
        sample_rate: Audio sample rate

    Returns:
        Complete WAV file as bytes
    """
    # WAV file header
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm_data)
    file_size = 36 + data_size

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',           # ChunkID
        file_size,         # ChunkSize
        b'WAVE',           # Format
        b'fmt ',           # Subchunk1ID
        16,                # Subchunk1Size (PCM)
        1,                 # AudioFormat (PCM)
        num_channels,      # NumChannels
        sample_rate,       # SampleRate
        byte_rate,         # ByteRate
        block_align,       # BlockAlign
        bits_per_sample,   # BitsPerSample
        b'data',           # Subchunk2ID
        data_size,         # Subchunk2Size
    )

    return header + pcm_data


def add_audio_to_video(
    video_bytes: bytes,
    moves_data: list[tuple[str, int]],
    frame_duration_ms: int = 1000,
) -> Optional[bytes]:
    """
    Add move/capture/check/checkmate sounds to a video.

    Args:
        video_bytes: Original video as bytes
        moves_data: List of tuples (move_type, position_index)
                   where move_type is "move", "capture", "check", or "checkmate"
        frame_duration_ms: Duration of each frame in milliseconds

    Returns:
        Video with audio as bytes, or None if failed
    """
    import subprocess

    if not moves_data:
        return video_bytes

    try:
        # Generate audio track
        sample_rate = 44100
        pcm_data = generate_audio_track(moves_data, frame_duration_ms, sample_rate)
        wav_data = create_wav_file(pcm_data, sample_rate)

        # Create temp files
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as video_file:
            video_file.write(video_bytes)
            video_path = Path(video_file.name)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as audio_file:
            audio_file.write(wav_data)
            audio_path = Path(audio_file.name)

        output_path = video_path.with_suffix('.out.mp4')

        try:
            # Use ffmpeg to merge video and audio
            result = subprocess.run(
                [
                    'ffmpeg', '-y',
                    '-i', str(video_path),
                    '-i', str(audio_path),
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-b:a', '128k',
                    '-shortest',
                    str(output_path),
                ],
                capture_output=True,
                timeout=60,
            )

            if result.returncode != 0:
                logger.error(f"ffmpeg error: {result.stderr.decode()}")
                return video_bytes  # Return original on failure

            # Read output
            return output_path.read_bytes()

        finally:
            # Cleanup temp files
            video_path.unlink(missing_ok=True)
            audio_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    except FileNotFoundError:
        logger.warning("ffmpeg not found, returning video without audio")
        return video_bytes
    except Exception as e:
        logger.error(f"Error adding audio to video: {e}")
        return video_bytes

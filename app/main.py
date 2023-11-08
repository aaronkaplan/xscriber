from fastapi import FastAPI, UploadFile, HTTPException, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from typing import Optional, List
import os
import shutil
from pydub import AudioSegment
from pydantic import BaseModel
from math import ceil

import openai


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


MODEL="gpt-3.5-turbo-16k"
# MODEL="gpt-4-32k"         # in case you already have access to it
MODEL="gpt-4-1106-preview"  # available since Nov 6th 2023. For super large context windows.


AUDIO_EXTENSIONS = ["mp3", "wav", "flac", "m4a"]
MAX_FILE_SIZE = 125 * 1024 * 1024  # 125 MB
PART_SIZE = 25 * 1024 * 1024  # 25 MB
TEMP_DIR = "/tmp"


class FilePart(BaseModel):
    """Class to capture file parts."""
    part: str
    size: int


@app.post("/upload/")
async def upload_audio(file: UploadFile):
    """Upload and process audio files."""

    print("Receiving uploaded file")
    if not file:
        raise HTTPException(status_code=400, detail="No file provided.")

    fileext = file.filename.split(".")[-1]
    if fileext not in AUDIO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid audio file format.")

    temp_file_path = os.path.join(TEMP_DIR, file.filename)
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"File saved as {temp_file_path}")
    file_size = os.path.getsize(temp_file_path)
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 125MB.")
    print(f"File size: {file_size}")

    # now split
    extension = file.filename.split(".")[-1]
    audio = AudioSegment.from_file(temp_file_path, format=extension)

    num_parts = ceil(file_size / PART_SIZE)
    part_duration = len(audio) / num_parts
    print(f"splitting audio file of {file_size} into {num_parts}, each with {part_duration} seconds")

    parts = []

    for i in range(num_parts):
        start_time = i * part_duration
        end_time = (i + 1) * part_duration

        part_audio = audio[start_time:end_time]

        part_filename = f"{file.filename}.part{i}.{extension}"
        part_path = os.path.join(TEMP_DIR, part_filename)

        part_audio.export(part_path, format="mp3")  # or use the format you want
        parts.append(FilePart(part=part_path, size=os.path.getsize(part_path)))

    print("Done")
    # return parts
    # Next transcribe each part via OpenAI's whisper (online)
    return await transcribe_and_summarize_all(parts)

def transcribe_audio(audio_file_path) -> str:
    """Transcribe audio into text"""
    print(f"Transcribing {audio_file_path}")
    with open(audio_file_path, 'rb') as audio_file:
        transcription = openai.Audio.transcribe("whisper-1", audio_file)
    print("Done")
    return transcription['text']


def meeting_minutes(transcription) -> dict:
    """The main function which combines all sub-summaries into a python dict/JSON"""
    print(f"Summarizing transcript starting with [{transcription[:50]}...]")
    abstract_summary = abstract_summary_extraction(transcription)
    key_points = key_points_extraction(transcription)
    action_items = action_item_extraction(transcription)
    sentiment = sentiment_analysis(transcription)
    print("Done")
    return {
        'abstract_summary': abstract_summary,
        'key_points': key_points,
        'action_items': action_items,
        'sentiment': sentiment
    }


def abstract_summary_extraction(transcription):
    """Use OpenAI to extract a summary"""
    response = openai.ChatCompletion.create(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a highly skilled AI trained in language comprehension and summarization. I would like you to read the following text and summarize it into a concise abstract paragraph. Aim to retain the most important points, providing a coherent and readable summary that could help a person understand the main points of the discussion without needing to read the entire text. Please avoid unnecessary details or tangential points."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response['choices'][0]['message']['content']


def key_points_extraction(transcription):
    """Use OpenAI to extract key points"""
    response = openai.ChatCompletion.create(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a proficient AI with a specialty in distilling information into key points. Based on the following text, identify and list the main points that were discussed or brought up. These should be the most important ideas, findings, or topics that are crucial to the essence of the discussion. Your goal is to provide a list that someone could read to quickly understand what was talked about."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response['choices'][0]['message']['content']


def action_item_extraction(transcription):
    """Use OpenAI to extract action items from the meeting"""
    response = openai.ChatCompletion.create(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are an AI expert in analyzing conversations and extracting action items. Please review the text and identify any tasks, assignments, or actions that were agreed upon or mentioned as needing to be done. These could be tasks assigned to specific individuals, or general actions that the group has decided to take. Please list these action items clearly and concisely."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response['choices'][0]['message']['content']


def sentiment_analysis(transcription):
    """Do some sentiment analysis"""
    response = openai.ChatCompletion.create(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "As an AI with expertise in language and emotion analysis, your task is to analyze the sentiment of the following text. Please consider the overall tone of the discussion, the emotion conveyed by the language used, and the context in which words and phrases are used. Indicate whether the sentiment is generally positive, negative, or neutral, and provide brief explanations for your analysis where possible."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response['choices'][0]['message']['content']


async def transcribe_and_summarize_all(parts: List[FilePart]) -> dict:
    """Here we collect all 25MB chunks (parts) and transcribe them and make one big 
    meeting minutes summary out of it."""
    transcription = ""
    minutes = ""

    print("Transcribing...")
    num_parts = len(parts)
    i = 0
    for part in parts:
        print(f"Part {i} of {num_parts}: {part.part}")
        audio_file = part.part
        partial_transcription =  transcribe_audio(audio_file)
        transcription = transcription + partial_transcription
        print("Done")
        i += 1
    minutes = meeting_minutes(transcription)
    return  minutes


@app.get("/")
async def read_root(request: Request, error_message: Optional[str] = None):
    """Render the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request, "error_message": error_message})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9977)

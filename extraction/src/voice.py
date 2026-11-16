"""Voice memos to text with Amazon Transcribe, for the VOICE input (free_text reads a transcript like an email).

Each memo is uploaded to S3, transcribed as a batch job, and the transcript saved as JSON with the other saved answers.
The audio, the transcript in S3, and the job are deleted after, so nothing stays in AWS. A saved transcript is reused,
so each memo is paid for once.

A custom vocabulary of the firm's client names (vocabulary_phrases, ensure_vocabulary) can be passed so a name like
"Halvorsen" comes out spelled as in Matter Master. Its name carries a fingerprint of its phrases, so a changed client
list makes a new vocabulary instead of reusing an old one.

Only the formats Transcribe takes directly are read (AUDIO_TYPES): that covers Voice Memos and most Android recorders
(.m4a), other recorders (.mp3, .wav), WhatsApp voice notes (Ogg Opus), and recording in the browser (.webm in
Chrome, .mp4 in Safari). Apps that hand over text instead (a
transcript or summary from an AI notetaker, or Voice Memos' own transcript) go in as typed text.

The work date of a memo is the day it was recorded, read from the file (recorded_on), not the day it was uploaded.
Who sent it comes from how it arrived (the signed-in user, or the sender of the email or text), not from the file:
memo files don't name their speaker.
"""
import hashlib
import json
import re
import struct
import time
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

AUDIO_TYPES = {".m4a": "m4a", ".mp3": "mp3", ".mp4": "mp4", ".wav": "wav", ".ogg": "ogg", ".opus": "ogg",
               ".webm": "webm"}   # .webm and .mp4: what a browser records in (Chrome and Safari)
MP4_TYPES = {".m4a", ".mp4"}                     # files with a recording time inside
PRICE_PER_MINUTE = 0.024     # batch transcription, US English
LANGUAGE = "en-US"
FIRM_TIME_ZONE = "America/Los_Angeles"
MP4_EPOCH = datetime(1904, 1, 1, tzinfo=timezone.utc)


def is_audio(path):
    return Path(path).suffix.lower() in AUDIO_TYPES


def mp4_boxes(data, start=0, end=None):
    """(type, body start, body end) for each box between start and end."""
    end = len(data) if end is None else end
    while start + 8 <= end:
        size, kind = struct.unpack(">I4s", data[start:start + 8])
        header = 8
        if size == 1:
            size, header = struct.unpack(">Q", data[start + 8:start + 16])[0], 16
        elif size == 0:
            size = end - start
        if size < header:
            return
        yield kind.decode("latin-1"), start + header, start + size
        start += size


def recorded_on(path, time_zone=FIRM_TIME_ZONE):
    """The day a memo was recorded, in the firm's time zone, from the creation time inside an .m4a or .mp4 file (the
    movie header, written by the phone when it records). None for other files, or when it's missing,
    so the caller falls back to when the memo was sent."""
    path = Path(path)
    if path.suffix.lower() not in MP4_TYPES:
        return None
    data = path.read_bytes()
    for kind, body, end in mp4_boxes(data):
        if kind != "moov":
            continue
        for inner, start, _ in mp4_boxes(data, body, end):
            if inner != "mvhd":
                continue
            version = data[start]
            seconds = struct.unpack(">Q" if version == 1 else ">I",
                                    data[start + 4:start + (12 if version == 1 else 8)])[0]
            if not seconds:
                return None
            made = MP4_EPOCH + timedelta(seconds=seconds)
            return made.astimezone(ZoneInfo(time_zone)).date().isoformat()
    return None


def vocabulary_phrases(names):
    """Names as Transcribe vocabulary phrases: letters and apostrophes only, words joined by hyphens. Digits, symbols,
    and one-letter words are dropped (a phrase with anything else fails the whole vocabulary)."""
    phrases = set()
    for name in names:
        words = [w.strip("'") for w in re.sub(r"[^A-Za-z']+", " ", str(name)).split()]
        words = [w for w in words if len(w) > 1]
        phrase = "-".join(words)
        if phrase and len(phrase) < 256:
            phrases.add(phrase)
    return sorted(phrases)


def vocabulary_name(phrases, prefix="tenth-clients"):
    return f"{prefix}-{hashlib.md5('|'.join(phrases).encode()).hexdigest()[:10]}"


def ensure_vocabulary(client, phrases, wait=5, prefix="tenth-clients"):
    """The name of a READY custom vocabulary with these phrases, created if it doesn't exist (a minute or two)."""
    name = vocabulary_name(phrases, prefix)
    try:
        client.get_vocabulary(VocabularyName=name)
    except (client.exceptions.NotFoundException, client.exceptions.BadRequestException):
        client.create_vocabulary(VocabularyName=name, LanguageCode=LANGUAGE, Phrases=phrases)
    while True:
        info = client.get_vocabulary(VocabularyName=name)
        if info["VocabularyState"] == "READY":
            return name
        if info["VocabularyState"] == "FAILED":
            raise RuntimeError(f"Vocabulary {name} failed: {info.get('FailureReason')}")
        time.sleep(wait)


def transcript_answer(result, vocabulary):
    """The parts of Transcribe's result worth keeping: the text, each word with its confidence, and the length."""
    words = [[item["alternatives"][0]["content"], float(item["alternatives"][0].get("confidence", 0))]
             for item in result["results"]["items"] if item["type"] == "pronunciation"]
    ends = [float(item["end_time"]) for item in result["results"]["items"] if "end_time" in item]
    return {"text": result["results"]["transcripts"][0]["transcript"], "words": words,
            "seconds": max(ends) if ends else 0.0, "vocabulary": vocabulary}


def fetch_json(uri):
    with urllib.request.urlopen(uri) as response:
        return json.loads(response.read())


def transcribe(path, out_path, s3, client, bucket, vocabulary=None, wait=5, fetch=fetch_json):
    """The saved transcript at out_path, or a new one from Transcribe (saved there). A failed job returns an answer
    with an "error" and no text, and isn't saved, so a rerun tries again.

    The memo goes to our bucket only for the job; the transcript stays in Transcribe's own bucket and is read once
    through the link it gives (fetch), so Transcribe never needs to write to ours."""
    out_path = Path(out_path)
    if out_path.exists():
        return json.loads(out_path.read_text())
    path = Path(path)
    if not is_audio(path):
        raise ValueError(f"Transcribe doesn't take {path.suffix} files: {path.name}")
    job = f"tenth-{uuid.uuid4().hex[:16]}"
    media_key = f"transcribe/{job}{path.suffix.lower()}"
    s3.upload_file(str(path), bucket, media_key)
    try:
        request = {"TranscriptionJobName": job, "LanguageCode": LANGUAGE, "MediaFormat": AUDIO_TYPES[path.suffix.lower()],
                   "Media": {"MediaFileUri": f"s3://{bucket}/{media_key}"}}
        if vocabulary:
            request["Settings"] = {"VocabularyName": vocabulary}
        client.start_transcription_job(**request)
        while True:
            info = client.get_transcription_job(TranscriptionJobName=job)["TranscriptionJob"]
            if info["TranscriptionJobStatus"] in ("COMPLETED", "FAILED"):
                break
            time.sleep(wait)
        if info["TranscriptionJobStatus"] == "FAILED":
            return {"text": "", "error": info.get("FailureReason", "failed"), "vocabulary": vocabulary}
        result = fetch(info["Transcript"]["TranscriptFileUri"])
    finally:
        s3.delete_object(Bucket=bucket, Key=media_key)
        try:
            client.delete_transcription_job(TranscriptionJobName=job)   # also deletes Transcribe's copy
        except Exception:
            pass
    answer = transcript_answer(result, vocabulary)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(answer, indent=1))
    return answer

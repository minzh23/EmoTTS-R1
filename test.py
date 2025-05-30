from openai import OpenAI
import os
import base64

os.environ["OPENAI_API_KEY"] = "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"

audio_file_path = "/root/EmoVoice/checkpoint/tts_decode_test_rp_seed_greedy_kaiyuan/pred_audio/neutral_prompt_speech/gpt4o_13_angry_verse.wav"
with open(audio_file_path, "rb") as audio_file:
    audio_data = audio_file.read()
    base64_audio = base64.b64encode(audio_data).decode('utf-8')

client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-4o-audio-preview",
  messages=[
    {"role": "developer", "content": "You are a strict assistant who tries to provide precise evaluation. You are given a text message and an audio input. In the text message, there is a target text and a emotional prompt. You have to decide how the input audio conveyed the emotion described in the emotional prompt. Rate the audio from 0 to 10, where 0 is the worst and 10 is the best. Output only the rating number between <answer> and </answer>, with no further explanation. "},
    {"role": "user", 
     "content": [
        {"type": "text", "text": "The text message is \"When exactly did I lose my right to speak?\". The emotional prompt is : \"Echoing suppressed anger and bitter resentment.\""},
        {
            "type": "input_audio", 
            "input_audio":
            {
                "data": base64_audio,
                "format": "wav"
            }
        }
     ]
     }
  ]
)

print(completion.choices[0].message.content)

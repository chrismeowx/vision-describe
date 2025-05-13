import cv2
import openai
import os
import time
import torch
import pygame
import speech_recognition as sr
from ultralytics import YOLO
from gtts import gTTS

openai.api_key = "xxxxx" # enter your own api key :)

def chatgpt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful and talkative assistant. Always answers in 30 words."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=50,
    )
    return response['choices'][0]['message']['content'].strip()

def play_sound(texts):
    tts = gTTS(text=texts, lang='en')
    tts.save("response.mp3")

    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load("response.mp3")
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pass

def conversation():
    recognizer = sr.Recognizer()

    while True:
        try:
            with sr.Microphone() as mic:
                recognizer.adjust_for_ambient_noise(mic, duration=3)
                audio = recognizer.listen(mic)

                texts = recognizer.recognize_google(audio)
                texts = texts.lower() 
                print(f"User: {texts}")
                
        except sr.UnknownValueError:
            recognizer = sr.Recognizer()
            continue

        if texts == "thank you":
                    print("Exiting conversation.")
                    break
        else:
            chat_response = chatgpt(texts)
            print("Isabella: ", chat_response)
            play_sound(chat_response)



def main():
    model = YOLO("yolov8s.pt")

    cap = cv2.VideoCapture(0)
    start = time.time()
    durations = 30

    latest_object = set()  

    while True:
        ret, frame  = cap.read()

        if not ret:
            break

        results = model.predict(source=frame, stream=True, show=True, conf=0.3)

        for result in results:
            for obj in result.boxes.data:
                class_id = int(obj[5])
                class_name = model.names[class_id] 
                latest_object.add(class_name)

        after = time.time() - start
        if after > durations:
            break

    cap.release()
    cv2.destroyAllWindows()

    if latest_object:
        objects_text = f"There is a {latest_object}."
    else:
        objects_text = "There's nothing."

    play_sound(objects_text)
    describe = f"Describe what is a {latest_object}."
    describe_chatgpt = chatgpt(describe)
    play_sound(describe_chatgpt)
    conversation()

if __name__ == "__main__":
    main()

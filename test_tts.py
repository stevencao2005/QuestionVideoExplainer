import pyttsx3

print("Initializing TTS engine...")
try:
    engine = pyttsx3.init()
except Exception as e:
    print(f"Error initializing pyttsx3: {e}")
    exit()

print("Engine initialized.")

# Optional: Print available voices
# voices = engine.getProperty('voices')
# print("Available voices:")
# for voice in voices:
#     print(f" - ID: {voice.id}")
#     print(f"   Name: {voice.name}")
#     print(f"   Lang: {voice.languages}")

text_to_speak = "Hello, this is a test of the text to speech engine."
print(f"Attempting to speak: '{text_to_speak}'")

try:
    engine.say(text_to_speak)
    # Wait for speech to complete
    engine.runAndWait()
    print("Speech finished.")

except Exception as e:
    print(f"Error during speech: {e}")

print("Script finished.") 
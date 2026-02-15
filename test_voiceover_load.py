from manim import Scene # Keep basic Scene import
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.pyttsx3 import PyTTSX3Service

# Define a scene that inherits from VoiceoverScene but does nothing
class VoiceoverLoadTest(VoiceoverScene):
    def construct(self):
        # We don't even initialize the service here yet
        # Just testing if Manim can load a class inheriting from VoiceoverScene
        pass 
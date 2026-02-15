import numpy as np
from manim import *

class BasicTest(Scene):
    def construct(self):
        circle = Circle()
        self.play(Create(circle))
        self.wait()
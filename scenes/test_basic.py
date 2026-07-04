from manim import *


class TestBasic(Scene):
    def construct(self):
        circle = Circle()
        self.play(Create(circle))
        self.wait()
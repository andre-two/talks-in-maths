from manim import *


class TestLatex(Scene):
    def construct(self):
        formula = MathTex(r"\int_0^1 x^2\,dx = \frac{1}{3}")
        self.play(Write(formula))
        self.wait()
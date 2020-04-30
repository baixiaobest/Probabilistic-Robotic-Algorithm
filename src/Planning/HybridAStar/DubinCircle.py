from enum import Enum

class CircleType(Enum):
    CLOCKWISE = 1
    COUNTER_CLOCKWISE = 2

class DubinCircle:
    def __init__(self, position, circle_type):
        self.position = position
        self.circle_type = circle_type
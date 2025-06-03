# backend/objects/sample_class.py

import random

from PySide6.QtGui import QColor

from backend.objects.sample import Sample


class SampleClass:
    def __init__(self, id, name, color="#FFFFFF"):
        self.id = id
        self.name = name
        self.color = color
        self.samples = set()  # Set of Sample objects
        self.children = set()  # Set of SampleClass objects

    def add_image(self, sample):
        self.samples.add(sample)
        sample.class_id = self.id

    def remove_image(self, sample: Sample):
        self.samples.discard(sample)
        if sample.class_id == self.id:
            sample.class_id = None

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "color": self.color,
            "samples": [sample.id for sample in self.samples],
        }

    @staticmethod
    def from_dict(data):
        return SampleClass(
            id=data["id"], name=data["name"], color=data.get("color", "#FFFFFF")
        )

    def _generate_random_color(self):
        """Generates a random QColor."""
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return QColor(r, g, b).name()

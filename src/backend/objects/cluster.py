# backend/objects/cluster.py


class Cluster:
    def __init__(self, id, color="#FFFFFF"):
        self.id = id
        self.color = color
        self.samples = set()  # Set of Image objects

    def add_image(self, sample):
        self.samples.add(sample)
        sample.cluster_ids.add(self.id)

    def remove_image(self, image):
        self.samples.discard(image)
        image.cluster_ids.discard(self.id)

    def to_dict(self):
        return {
            "id": self.id,
            "color": self.color,
            "samples": [sample.id for sample in self.samples],
        }

    @staticmethod
    def from_dict(data):
        return Cluster(id=data["id"], color=data.get("color", "#FFFFFF"))

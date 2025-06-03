import pickle


class SaveFile:
    def __init__(self):
        self.sessions = {}  # {session_id: session_data}
        self.data_managers = {}  # {session_id: data_manager_data}

    def add_session(self, session):
        """Adds a session to the SaveFile."""
        self.sessions[session.id] = {
            "id": session.id,
            "name": session.name,
            "creation_date": session.creation_date,
            "last_opening_date": session.last_opening_date,
            "images_directory": session.images_directory,
            "processor_model_name": session.processor_model_name,
        }

    def add_data_manager(self, session_id, data_manager):
        """Adds the data from a DataManager to the SaveFile."""
        self.data_managers[session_id] = {
            "images": {
                image_id: image.to_dict()
                for image_id, image in data_manager.samples.items()
            },
            "clusters": {
                cluster_id: cluster.to_dict()
                for cluster_id, cluster in data_manager.clusters.items()
            },
            "classes": {
                class_id: class_object.to_dict()
                for class_id, class_object in data_manager.classes.items()
            },
            "masks": {
                mask_id: mask.to_dict() for mask_id, mask in data_manager.masks.items()
            },
        }

    def save_to_file(self, file_path):
        """Saves the SaveFile object to a pickle file."""
        with open(file_path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

# backend/session_manager.py

import logging
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QTimer

from backend.config import DEFAULT_SETTINGS, SESSIONS_INDEX_FILE, SRC_ROOT
from backend.objects.session import Session
from backend.utils.file_utils import atomic_write, read_json

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class SessionManager:
    def __init__(self, sessions_dir: Optional[Path] = None):
        if sessions_dir is None:
            self.sessions_dir: Path = SRC_ROOT.parent / "sessions"
        else:
            self.sessions_dir: Path = Path(sessions_dir)
        os.makedirs(self.sessions_dir, exist_ok=True)
        self.index_file: Path = SESSIONS_INDEX_FILE
        self.sessions: Dict[str, Session] = {}
        self._load_sessions_index()

    def _load_sessions_index(self):
        if not self.index_file.exists():
            try:
                atomic_write(self.index_file, {"sessions": []})
            except Exception as e:
                logging.error(f"Failed to create sessions index file: {e}")
                return
        try:
            data = read_json(self.index_file)
            if data is None:
                data = {"sessions": []}
            for session_entry_data in data.get("sessions", []):
                session_folder_path = session_entry_data.get("path")
                if session_folder_path and os.path.isdir(session_folder_path):
                    session = self._load_session_metadata_from_folder(
                        Path(session_folder_path)
                    )
                    if session:
                        self.sessions[session.id] = session
                    else:
                        logging.warning(
                            f"Failed to load session metadata from: {session_folder_path}"
                        )
                else:
                    logging.warning(
                        f"Invalid session path in index: {session_folder_path}"
                    )
        except Exception as e:
            logging.error(
                f"Error loading sessions index {self.index_file}: {e}", exc_info=True
            )

    def _save_sessions_index(self):
        data = {"sessions": []}
        for session in self.sessions.values():
            data["sessions"].append(
                {
                    "id": session.id,
                    "name": session.name,
                    "creation_date": session.creation_date,
                    "last_opened": session.last_opening_date,  # Ensure Session object updates this on open
                    "path": str(session.session_folder),
                }
            )
        try:
            atomic_write(self.index_file, data)
        except Exception as e:
            logging.error(f"Failed to save sessions index: {e}")

    def _load_session_metadata_from_folder(
        self, session_folder: Path
    ) -> Optional[Session]:
        session_info_path = session_folder / "session_info.json"
        if not session_info_path.is_file():
            logging.warning(f"Session info file not found: {session_info_path}")
            return None
        try:
            session_data_dict = read_json(session_info_path)
            if not session_data_dict:
                raise ValueError("Session data empty.")
            if os.path.abspath(
                session_data_dict.get("session_folder", "")
            ) != os.path.abspath(str(session_folder)):
                session_data_dict["session_folder"] = str(
                    session_folder
                )  # Correct path if moved
            return Session.from_dict(session_data_dict)
        except Exception as e:
            logging.error(
                f"Error loading session metadata from {session_info_path}: {e}",
                exc_info=True,
            )
            return None

    def create_session(self, session_details: Dict[str, Any]) -> Optional[Session]:
        """
        Creates a new analysis session from a dictionary of details.
        Expected keys in session_details: 'name', 'data_folder' (for images_directory).
        Optional keys: 'processor_model_name', 'source_type', 'preprocessing_script', etc.
        """
        session_name = session_details.get("name", "").strip()
        images_directory_str = session_details.get(
            "data_folder", ""
        ).strip()  # From SessionFormWidget

        if not session_name:
            logging.error("SessionManager: Session name cannot be empty for creation.")
            return None
        if not images_directory_str or not os.path.isdir(images_directory_str):
            logging.error(
                f"SessionManager: Images directory '{images_directory_str}' does not exist or is invalid."
            )
            return None

        # Check for duplicate session names (case-insensitive for robustness)
        for existing_session in self.sessions.values():
            if existing_session.name.lower() == session_name.lower():
                logging.error(
                    f"SessionManager: A session with the name '{session_name}' already exists."
                )
                # Consider raising an error or returning a specific status
                return None

        session_id = str(uuid.uuid4())
        # Sanitize session_folder_name if session_name can have problematic chars
        session_folder_name_sanitized = "".join(
            c if c.isalnum() or c in " _-" else "_" for c in session_name
        )
        session_folder_path = (
            self.sessions_dir
            / f"{session_folder_name_sanitized}_{session_id[:8]} [BioSort]"
        )  # Add part of ID for uniqueness

        try:
            os.makedirs(
                session_folder_path, exist_ok=False
            )  # exist_ok=False to prevent overwriting
            for sub_dir_name in ["features", "masks", "metadata", "masked_images"]:
                os.makedirs(session_folder_path / sub_dir_name, exist_ok=True)

            now_iso = datetime.now().isoformat()

            # Prepare data for Session constructor, using defaults from Session object if not in details
            session_constructor_data = {
                "id": session_id,
                "name": session_name,
                "creation_date": now_iso,
                "last_opening_date": now_iso,
                "images_directory": images_directory_str,  # Will be abspath'd by Session constructor
                "session_folder": str(session_folder_path),  # Will be abspath'd
                "processor_model_name": session_details.get(
                    "processor_model_name", DEFAULT_SETTINGS["selected_model"]
                ),
                "source_type": session_details.get("source_type", "images"),
                "preprocessing_script": session_details.get("preprocessing_script"),
                "preprocessing_args": session_details.get("preprocessing_args"),
            }

            session = Session(**session_constructor_data)

            atomic_write(session.session_info_path, session.to_dict())

            self.sessions[session.id] = session
            self._save_sessions_index()

            logging.info(
                f"SessionManager: Session '{session.name}' (ID: {session.id}) created successfully at {session.session_folder}."
            )
            return session

        except FileExistsError:
            logging.error(
                f"SessionManager: Failed to create session '{session_name}'. Folder already exists: {session_folder_path}"
            )
            return None
        except Exception as e:
            logging.error(
                f"SessionManager: Failed to create session '{session_name}': {e}",
                exc_info=True,
            )
            if os.path.exists(session_folder_path):  # Attempt cleanup
                try:
                    shutil.rmtree(session_folder_path)
                except Exception as e_clean:
                    logging.error(
                        f"Error cleaning up session folder {session_folder_path}: {e_clean}"
                    )
            return None

    def update_session_metadata(
        self, session_id: str, new_details: Dict[str, Any]
    ) -> Optional[Session]:
        session = self.sessions.get(session_id)
        if not session:
            logging.error(
                f"SessionManager: Cannot update session. ID {session_id} not found."
            )
            return None

        logging.info(
            f"SessionManager: Updating metadata for session {session_id} (current name: {session.name})."
        )

        changes_made = False

        #  Primarily allow NAME change
        new_name = new_details.get("name", session.name).strip()
        if new_name and session.name != new_name:
            if any(
                s.name.lower() == new_name.lower() and s.id != session_id
                for s in self.sessions.values()
            ):
                logging.error(
                    f"SessionManager: Cannot rename session to '{new_name}'. Name already exists."
                )
                return None

            logging.info(
                f"SessionManager: Renaming session '{session.name}' to '{new_name}'."
            )
            session.name = new_name
            changes_made = True

        # Example: images_directory (data_folder from UI)
        new_images_dir_ui = new_details.get("data_folder", "").strip()
        if new_images_dir_ui and os.path.abspath(
            session.images_directory
        ) != os.path.abspath(new_images_dir_ui):
            logging.warning(
                f"SessionManager: Attempt to change images_directory for session {session_id} "
                f"from '{session.images_directory}' to '{new_images_dir_ui}'. "
                f"This change is currently NOT APPLIED for existing sessions to prevent data inconsistency. "
                f"Name change (if any) will still be applied."
            )

        new_proc_model = new_details.get(
            "processor_model_name", session.processor_model_name
        )
        if session.processor_model_name != new_proc_model:
            session.processor_model_name = new_proc_model
            changes_made = True
            logging.info(
                f"SessionManager: Processor model for session {session_id} changed to {new_proc_model}."
            )

        if session.source_type != new_details.get("source_type", session.source_type):
            session.source_type = new_details.get("source_type", session.source_type)
            changes_made = True

        new_script = new_details.get("preprocessing_script")
        new_script_abs = os.path.abspath(new_script) if new_script else None
        if session.preprocessing_script != new_script_abs:
            session.preprocessing_script = new_script_abs
            changes_made = True

        if session.preprocessing_args != new_details.get(
            "preprocessing_args", session.preprocessing_args
        ):
            session.preprocessing_args = new_details.get(
                "preprocessing_args", session.preprocessing_args
            )
            changes_made = True

        if changes_made:
            session.last_opening_date = (
                datetime.now().isoformat()
            )  # Consider this a modification time
            try:
                atomic_write(session.session_info_path, session.to_dict())
                self._save_sessions_index()  # Name or last_opened might affect index
                logging.info(
                    f"SessionManager: Session {session_id} metadata updated and saved."
                )
                return session
            except Exception as e:
                logging.error(
                    f"SessionManager: Failed to save updated session metadata for {session_id}: {e}",
                    exc_info=True,
                )
                return None
        else:
            logging.info(
                f"SessionManager: No actual changes applied for session {session_id} metadata update."
            )
            return session

    def save_session_metadata(self, session: Session):  # This is called by open_session
        """Saves session's metadata, primarily to update last_opening_date."""
        if not isinstance(session, Session):
            return
        session.last_opening_date = datetime.now().isoformat()
        try:
            atomic_write(session.session_info_path, session.to_dict())
            self._save_sessions_index()  # Index needs update due to last_opening_date
        except Exception as e:
            logging.error(
                f"Failed to save session metadata (on open) for {session.id}: {e}"
            )

    def open_session(self, session_id: str) -> Optional[Session]:
        session = self.sessions.get(session_id)
        if session:
            self.save_session_metadata(session)
            logging.info(f"Session '{session.name}' (ID: {session_id}) opened.")
            return session
        else:
            for item in self.sessions_dir.iterdir():
                if item.is_dir():
                    potential_session_info = item / "session_info.json"
                    if potential_session_info.exists():
                        try:
                            data = read_json(potential_session_info)
                            if data and data.get("id") == session_id:
                                logging.info(
                                    f"Found session {session_id} on disk, not in memory. Loading."
                                )
                                loaded_session = (
                                    self._load_session_metadata_from_folder(item)
                                )
                                if loaded_session:
                                    self.sessions[session_id] = (
                                        loaded_session  # Add to cache
                                    )
                                    self.save_session_metadata(
                                        loaded_session
                                    )  # Update last opened
                                    return loaded_session
                        except Exception as e:
                            logging.warning(
                                f"Error trying to load session {session_id} from disk path {item}: {e}"
                            )
            logging.warning(
                f"SessionManager: Session with ID {session_id} not found in manager or on disk."
            )
            return None

    def delete_session(self, session_id: str, delay_ms: int = 1000) -> bool:
        # ... (existing delete_session code is fine) ...
        session = self.sessions.pop(session_id, None)
        if not session:
            logging.warning(
                f"Session with ID {session_id} not found for deletion in manager."
            )
            self._save_sessions_index()
            return False

        session_folder_to_delete = str(session.session_folder)

        def perform_deletion():
            try:
                if os.path.exists(session_folder_to_delete):
                    shutil.rmtree(session_folder_to_delete)
                logging.info(f"Session folder {session_folder_to_delete} deleted.")
            except Exception as e:
                logging.error(
                    f"Error deleting session folder {session_folder_to_delete}: {e}"
                )
            finally:
                self._save_sessions_index()  # Update index AFTER deletion attempt

        if delay_ms > 0:
            QTimer.singleShot(delay_ms, perform_deletion)
        else:
            perform_deletion()
        return True

    def get_session(self, session_id: str) -> Optional[Session]:
        return self.sessions.get(session_id)

    def list_sessions(self) -> List[Session]:
        return sorted(
            list(self.sessions.values()),
            key=lambda s: s.last_opening_date,
            reverse=True,
        )

    def list_sessions_for_ui(self) -> List[Dict[str, Any]]:  # NEW METHOD
        """
        Lists all sessions formatted as dictionaries suitable for UI display
        (e.g., for SessionManagementDialog), sorted by last opening date.
        Each dictionary is the result of Session.to_dict().
        """
        sorted_sessions = self.list_sessions()  # Get sorted List[Session]
        return [session.to_dict() for session in sorted_sessions]

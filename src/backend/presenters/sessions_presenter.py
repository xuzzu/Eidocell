### backend\presenters\sessions_presenter.py
import logging
from typing import Any, List

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtWidgets import QMessageBox

from backend.objects.session import Session
from backend.session_manager import SessionManager
from UI.navigation_interface.sessions.session_management_dialog import (
    SessionManagementDialog,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class SessionPresenter(QObject):
    """Presenter for handling session related actions via SessionManagementDialog."""

    session_chosen = Signal(str)  # Emitted when a session is chosen to be opened

    def __init__(
        self,
        session_manager: SessionManager,
        session_management_dialog: SessionManagementDialog,  # New dialog type
        backend_initializer_ref: Any,
    ):  # Actual type: BackendInitializer
        super().__init__()
        self.session_manager = session_manager
        self.session_management_dialog = (
            session_management_dialog  # Store reference to the new dialog
        )
        self.backend_initializer = backend_initializer_ref

        # Connect signals from the new SessionManagementDialog
        self.session_management_dialog.create_session_requested.connect(
            self.handle_create_session_request
        )
        self.session_management_dialog.update_session_requested.connect(
            self.handle_update_session_request
        )
        self.session_management_dialog.delete_session_requested.connect(
            self.handle_delete_session_request
        )
        self.session_management_dialog.open_session_requested.connect(
            self.handle_open_session_request
        )

        # The old FolderListDialog signals are no longer relevant here
        # self.session_view_widget.session_created.connect(self.create_session_from_folder_card) # Old
        # self.session_view_widget.session_chosen.connect(self.choose_session_from_folder_card) # Old
        # self.session_view_widget.delete_request.connect(self.delete_session_from_folder_card) # Old

    @Slot(str)
    def handle_open_session_request(self, session_id: str):
        """Handles the request to open a session from SessionManagementDialog."""
        logging.info(
            f"SessionPresenter: Received request to open session ID: {session_id}"
        )
        session = self.session_manager.open_session(
            session_id
        )  # open_session updates last_opened
        if session:
            logging.info(
                f"SessionPresenter: Session '{session.name}' opened successfully."
            )
            self.session_chosen.emit(session_id)  # Signal to BackendInitializer
            # Dialog will be accepted/closed by its own 'Open Session' button logic
        else:
            logging.error(
                f"SessionPresenter: Error opening session with ID {session_id}."
            )
            # SessionManagementDialog might show its own error message or this presenter could.

    @Slot(dict)
    def handle_create_session_request(self, session_details: dict):
        """Handles the request to create a new session from SessionManagementDialog."""
        logging.info(
            f"SessionPresenter: Received request to create new session with details: {session_details.get('name')}"
        )

        # SessionManager.create_session now expects a dict and handles name collision
        session = self.session_manager.create_session(session_details)

        if session:
            logging.info(
                f"SessionPresenter: Session '{session.name}' created successfully by SessionManager."
            )
            QMessageBox.information(
                self.session_management_dialog,
                "Session Created",
                f"Session '{session.name}' created successfully.",
            )

            # Inform the dialog to refresh its list and select the new session
            self._refresh_dialog_session_list()
            self.session_management_dialog.select_session_in_list(session.id)
            # select_session_in_list will trigger _on_session_selection_changed in the dialog,
            # which will load the new session's data and disable the form (as is_edit_mode/is_new_mode become False).
            # No need to call _on_new_session_clicked from here.
        else:
            logging.error(
                f"SessionPresenter: Session creation failed for '{session_details.get('name')}'. SessionManager returned None."
            )
            # If SessionManager.create_session returned None (e.g. due to name conflict it handled and logged),
            # we might show a generic failure message here or rely on dialog's own validation for empty fields.
            # For now, assume SessionManager logs specifics for internal failures.
            QMessageBox.critical(
                self.session_management_dialog,
                "Creation Failed",
                f"Could not create session '{session_details.get('name')}'. "
                "This might be due to an existing session with the same name or an invalid data folder. "
                "Please check the logs for more details.",
            )

    @Slot(str, dict)
    def handle_update_session_request(self, session_id: str, session_details: dict):
        """Handles the request to update an existing session's metadata."""
        logging.info(
            f"SessionPresenter: Received request to update session ID: {session_id} with new name: {session_details.get('name')}"
        )

        updated_session = self.session_manager.update_session_metadata(
            session_id, session_details
        )

        if updated_session:
            logging.info(
                f"SessionPresenter: Session '{updated_session.name}' (ID: {session_id}) updated successfully."
            )
            QMessageBox.information(
                self.session_management_dialog,
                "Update Successful",
                f"Session '{updated_session.name}' updated.",
            )
            self._refresh_dialog_session_list()
            # Re-select the updated session in the dialog's list
            self.session_management_dialog.select_session_in_list(session_id)
            # This will trigger _on_session_selection_changed, disabling the form.
        else:
            logging.error(
                f"SessionPresenter: Session update failed for ID {session_id}."
            )
            QMessageBox.critical(
                self.session_management_dialog,
                "Update Failed",
                f"Could not update session. This might be due to a name conflict. "
                "Please check the logs.",
            )

    @Slot(str)
    def handle_delete_session_request(self, session_id: str):
        """
        Handles the confirmed request to delete a session from SessionManagementDialog.
        """
        logging.info(
            f"SessionPresenter: Received confirmed delete request for session ID: {session_id}"
        )

        is_active_session = False
        if (
            self.backend_initializer.active_session
            and self.backend_initializer.active_session.id == session_id
        ):
            is_active_session = True
            logging.info(
                f"SessionPresenter: Session {session_id} is the currently active session. Triggering active session cleanup."
            )
            self.backend_initializer.on_current_session_deleted_by_user()  # Cleans up DM, presenters, UI

        # Proceed with deletion via SessionManager (this handles file system and index)
        success = self.session_manager.delete_session(
            session_id
        )  # delay_ms is handled by SessionManager

        if success:
            logging.info(
                f"SessionPresenter: SessionManager processed deletion for session: {session_id}"
            )
            # Inform the dialog to refresh its list (it will show one less item)
            self._refresh_dialog_session_list()
            # If it was not the active session, the dialog is still open.
            # If it *was* the active session, the dialog might have been closed if main window got disabled.
            # Dialog should clear its form if the deleted session was selected.
            if (
                self.session_management_dialog and not is_active_session
            ):  # Check if dialog still relevant
                self.session_management_dialog._on_new_session_clicked()  # Reset form to a neutral state
        else:
            logging.error(
                f"SessionPresenter: SessionManager reported failure to delete session: {session_id}"
            )
            # SessionManagementDialog might show an error if deletion fails at SessionManager level.
            QMessageBox.critical(
                self.session_management_dialog,
                "Deletion Failed",
                f"Could not delete session. Please check file permissions and logs.",
            )

    def _refresh_dialog_session_list(self):
        if self.session_management_dialog and self.session_manager:
            # Get List[Session] directly from SessionManager
            updated_sessions_list: List[Session] = self.session_manager.list_sessions()

            # Update the dialog's internal list with List[Session]
            self.session_management_dialog.existing_sessions = updated_sessions_list
            self.session_management_dialog._populate_session_list()  # Tell dialog to refresh UI list
            self.session_management_dialog._update_button_states()
            logging.debug(
                "SessionPresenter: Requested SessionManagementDialog to refresh its session list using Session objects."
            )

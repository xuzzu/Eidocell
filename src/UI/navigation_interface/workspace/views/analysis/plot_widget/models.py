# models.py

from PySide6.QtCore import QAbstractListModel, Qt
from PySide6.QtGui import QPixmap


class StringListModel(QAbstractListModel):
    def __init__(self, strings=None, parent=None):
        super().__init__(parent)
        self._stringList = strings or []
        self.image_data = {}  # Store image data associated with IDs

    def setStringList(self, strings):
        self.beginResetModel()
        self._stringList = strings
        self.endResetModel()

    def setImageData(self, id_, pixmap):
        self.image_data[id_] = pixmap

    def data(self, index, role):
        if not index.isValid():
            return None

        if role == Qt.DisplayRole:
            return self._stringList[index.row()]
        elif role == Qt.DecorationRole:
            return self.image_data.get(self._stringList[index.row()], None)
        return None

    def rowCount(self, parent=None):
        return len(self._stringList)

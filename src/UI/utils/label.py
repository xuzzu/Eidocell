# coding:utf-8
# COPIED FROM THE SRC OF FLUENT WIDGETS

import logging
import os
from typing import Union

import numpy as np
import tifffile
from PySide6.QtCore import Property, QRectF, QSize, Qt, QUrl, Signal
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QFont,
    QImage,
    QImageReader,
    QMovie,
    QPainter,
    QPainterPath,
    QPixmap,
)
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QWidget
from qfluentwidgets import (
    FluentStyleSheet,
    exceptionHandler,
    getFont,
    qconfig,
    setCustomStyleSheet,
    setFont,
)
from qfluentwidgets.common.overload import singledispatchmethod
from qfluentwidgets.components.widgets.menu import LabelContextMenu


class PixmapLabel(QLabel):
    """Label for high dpi pixmap"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.__pixmap = QPixmap()

    def setPixmap(self, pixmap: QPixmap):
        self.__pixmap = pixmap
        self.setFixedSize(pixmap.size())
        self.update()

    def pixmap(self):
        return self.__pixmap

    def paintEvent(self, e):
        if self.__pixmap.isNull():
            return super().paintEvent(e)

        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        painter.setPen(Qt.NoPen)
        painter.drawPixmap(self.rect(), self.__pixmap)


class FluentLabelBase(QLabel):
    """Fluent label base class

    Constructors
    ------------
    * FluentLabelBase(`parent`: QWidget = None)
    * FluentLabelBase(`text`: str, `parent`: QWidget = None)
    """

    @singledispatchmethod
    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self._init()

    @__init__.register
    def _(self, text: str, parent: QWidget = None):
        self.__init__(parent)
        self.setText(text)

    def _init(self):
        # FluentStyleSheet.LABEL.apply(self)
        self.setFont(self.getFont())
        self.setTextColor()
        qconfig.themeChanged.connect(
            lambda: self.setTextColor(self.lightColor, self.darkColor)
        )

        self.customContextMenuRequested.connect(self._onContextMenuRequested)
        return self

    def getFont(self):
        raise NotImplementedError

    @exceptionHandler()
    def setTextColor(self, light=QColor(0, 0, 0), dark=QColor(255, 255, 255)):
        """set the text color of label

        Parameters
        ----------
        light, dark: QColor | Qt.GlobalColor | str
            text color in light/dark mode
        """
        self._lightColor = QColor(light)
        self._darkColor = QColor(dark)

        setCustomStyleSheet(
            self,
            f"FluentLabelBase{{color:{self.lightColor.name(QColor.NameFormat.HexArgb)}}}",
            f"FluentLabelBase{{color:{self.darkColor.name(QColor.NameFormat.HexArgb)}}}",
        )

    @Property(QColor)
    def lightColor(self):
        return self._lightColor

    @lightColor.setter
    def lightColor(self, color: QColor):
        self.setTextColor(color, self.darkColor)

    @Property(QColor)
    def darkColor(self):
        return self._darkColor

    @darkColor.setter
    def darkColor(self, color: QColor):
        self.setTextColor(self.lightColor, color)

    @Property(int)
    def pixelFontSize(self):
        return self.font().pixelSize()

    @pixelFontSize.setter
    def pixelFontSize(self, size: int):
        font = self.font()
        font.setPixelSize(size)
        self.setFont(font)

    @Property(bool)
    def strikeOut(self):
        return self.font().strikeOut()

    @strikeOut.setter
    def strikeOut(self, isStrikeOut: bool):
        font = self.font()
        font.setStrikeOut(isStrikeOut)
        self.setFont(font)

    @Property(bool)
    def underline(self):
        return self.font().underline()

    @underline.setter
    def underline(self, isUnderline: bool):
        font = self.font()
        font.setStyle()
        font.setUnderline(isUnderline)
        self.setFont(font)

    def _onContextMenuRequested(self, pos):
        menu = LabelContextMenu(parent=self)
        menu.exec(self.mapToGlobal(pos))


class CaptionLabel(FluentLabelBase):
    """Caption text label

    Constructors
    ------------
    * CaptionLabel(`parent`: QWidget = None)
    * CaptionLabel(`text`: str, `parent`: QWidget = None)
    """

    def getFont(self):
        return getFont(12)


class BodyLabel(FluentLabelBase):
    """Body text label

    Constructors
    ------------
    * BodyLabel(`parent`: QWidget = None)
    * BodyLabel(`text`: str, `parent`: QWidget = None)
    """

    def getFont(self):
        return getFont(14)


class StrongBodyLabel(FluentLabelBase):
    """Strong body text label

    Constructors
    ------------
    * StrongBodyLabel(`parent`: QWidget = None)
    * StrongBodyLabel(`text`: str, `parent`: QWidget = None)
    """

    def getFont(self):
        return getFont(14, QFont.DemiBold)


class SubtitleLabel(FluentLabelBase):
    """Subtitle text label

    Constructors
    ------------
    * SubtitleLabel(`parent`: QWidget = None)
    * SubtitleLabel(`text`: str, `parent`: QWidget = None)
    """

    def getFont(self):
        return getFont(20, QFont.DemiBold)


class TitleLabel(FluentLabelBase):
    """Sub title text label

    Constructors
    ------------
    * TitleLabel(`parent`: QWidget = None)
    * TitleLabel(`text`: str, `parent`: QWidget = None)
    """

    def getFont(self):
        return getFont(28, QFont.DemiBold)


class LargeTitleLabel(FluentLabelBase):
    """Large title text label

    Constructors
    ------------
    * LargeTitleLabel(`parent`: QWidget = None)
    * LargeTitleLabel(`text`: str, `parent`: QWidget = None)
    """

    def getFont(self):
        return getFont(40, QFont.DemiBold)


class DisplayLabel(FluentLabelBase):
    """Display text label

    Constructors
    ------------
    * DisplayLabel(`parent`: QWidget = None)
    * DisplayLabel(`text`: str, `parent`: QWidget = None)
    """

    def getFont(self):
        return getFont(68, QFont.DemiBold)


class ImageLabel(QLabel):
    """Image label

    Constructors
    ------------
    * ImageLabel(`parent`: QWidget = None)
    * ImageLabel(`image`: str | QImage | QPixmap, `parent`: QWidget = None)
    """

    clicked = Signal()

    @singledispatchmethod
    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.image = QImage()
        self.setBorderRadius(0, 0, 0, 0)
        self._postInit()

    @__init__.register
    def _(self, image: str, parent=None):
        self.__init__(parent)
        self.setImage(image)
        self._postInit()

    @__init__.register
    def _(self, image: QImage, parent=None):
        self.__init__(parent)
        self.setImage(image)
        self._postInit()

    @__init__.register
    def _(self, image: QPixmap, parent=None):
        self.__init__(parent)
        self.setImage(image)
        self._postInit()

    def _postInit(self):
        pass

    def _onFrameChanged(self, index: int):
        self.image = self.movie().currentImage()
        self.update()

    def setBorderRadius(
        self, topLeft: int, topRight: int, bottomLeft: int, bottomRight: int
    ):
        """set the border radius of image"""
        self._topLeftRadius = topLeft
        self._topRightRadius = topRight
        self._bottomLeftRadius = bottomLeft
        self._bottomRightRadius = bottomRight
        self.update()

    def setImage(self, image_source: Union[str, QPixmap, QImage, None] = None):
        loaded_qimage = QImage()

        if isinstance(image_source, str):
            path_str = image_source
            if not os.path.exists(path_str):
                logging.error(f"ImageLabel: Image path does not exist: {path_str}")
            elif path_str.lower().endswith((".tif", ".tiff")):
                try:
                    img_data_tiff = tifffile.imread(path_str)
                    if (
                        img_data_tiff.ndim > 2
                        and img_data_tiff.shape[0] > 1
                        and img_data_tiff.ndim == 3
                    ):
                        if img_data_tiff.shape[0] <= 4:
                            img_data_tiff = np.moveaxis(img_data_tiff, 0, -1)
                        else:
                            img_data_tiff = img_data_tiff[0]
                    elif img_data_tiff.ndim > 3 and img_data_tiff.shape[0] > 1:
                        img_data_tiff = img_data_tiff[0]
                    if img_data_tiff.dtype != np.uint8:
                        if np.issubdtype(img_data_tiff.dtype, np.floating):
                            min_v, max_v = np.min(img_data_tiff), np.max(img_data_tiff)
                            if max_v > min_v:
                                img_data_tiff = (
                                    (img_data_tiff - min_v) / (max_v - min_v + 1e-9)
                                ) * 255
                            else:
                                img_data_tiff = (
                                    np.zeros_like(img_data_tiff)
                                    if min_v == 0
                                    else np.full_like(img_data_tiff, 128)
                                )
                        elif img_data_tiff.max() > 255:
                            img_data_tiff = img_data_tiff / (
                                img_data_tiff.max() / 255.0
                            )
                        img_data_tiff = img_data_tiff.astype(np.uint8)
                    height, width = img_data_tiff.shape[:2]
                    bytes_per_line = (
                        width * img_data_tiff.shape[2]
                        if img_data_tiff.ndim == 3
                        else width
                    )
                    fmt = QImage.Format_Grayscale8
                    if img_data_tiff.ndim == 3:
                        if img_data_tiff.shape[-1] == 3:
                            fmt = QImage.Format_RGB888
                        elif img_data_tiff.shape[-1] == 4:
                            fmt = QImage.Format_RGBA8888
                        elif img_data_tiff.shape[-1] == 1:
                            img_data_tiff = img_data_tiff.squeeze(axis=-1)
                            bytes_per_line = width
                            fmt = QImage.Format_Grayscale8
                    elif img_data_tiff.ndim == 2:
                        fmt = QImage.Format_Grayscale8
                    else:
                        logging.error(
                            f"ImageLabel: Unsupported TIFF shape: {img_data_tiff.shape} for {path_str}"
                        )
                    if not img_data_tiff.flags["C_CONTIGUOUS"]:
                        img_data_tiff = np.ascontiguousarray(img_data_tiff)
                    temp_q_img = QImage(
                        img_data_tiff.data, width, height, bytes_per_line, fmt
                    )
                    if not temp_q_img.isNull():
                        loaded_qimage = temp_q_img.copy()
                    else:
                        logging.error(
                            f"ImageLabel: QImage conversion failed for TIFF {path_str}"
                        )
                except Exception as e:
                    logging.error(
                        f"ImageLabel: Error loading TIFF {path_str}: {e}", exc_info=True
                    )
            else:
                reader = QImageReader(path_str)
                if reader.supportsAnimation() and reader.canRead():
                    movie = QMovie(path_str)
                    if movie.isValid():
                        self.setMovie(movie)
                        return
                    else:
                        logging.error(f"ImageLabel: QMovie invalid for {path_str}")
                elif reader.canRead():
                    temp_q_img = reader.read()
                    if not temp_q_img.isNull():
                        loaded_qimage = temp_q_img
                    else:
                        logging.error(
                            f"ImageLabel: QImageReader failed for {path_str}. Error: {reader.errorString()}"
                        )
                else:
                    logging.error(
                        f"ImageLabel: Cannot read image format for {path_str}. Error: {reader.errorString()}"
                    )
        elif isinstance(image_source, QPixmap):
            if not image_source.isNull():
                loaded_qimage = image_source.toImage()
        elif isinstance(image_source, QImage):
            if not image_source.isNull():
                loaded_qimage = image_source
        elif image_source is None:
            pass
        else:
            logging.warning(
                f"ImageLabel: Unsupported image source type: {type(image_source)}"
            )

        self.image = loaded_qimage if not loaded_qimage.isNull() else QImage()

        self.setFixedSize(self.image.size() if not self.image.isNull() else QSize(0, 0))
        self.update()

    def scaledToWidth(self, width: int):
        if self.isNull():
            return

        h = int(width / self.image.width() * self.image.height())
        self.setFixedSize(width, h)

        if self.movie():
            self.movie().setScaledSize(QSize(width, h))

    def scaledToHeight(self, height: int):
        if self.isNull():
            return

        w = int(height / self.image.height() * self.image.width())
        self.setFixedSize(w, height)

        if self.movie():
            self.movie().setScaledSize(QSize(w, height))

    def isNull(self):
        return self.image.isNull()

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        self.clicked.emit()

    def setPixmap(self, pixmap: QPixmap):
        self.setImage(pixmap)

    def pixmap(self) -> QPixmap:
        return QPixmap.fromImage(self.image)

    def setMovie(self, movie: QMovie):
        super().setMovie(movie)
        self.movie().start()
        self.image = self.movie().currentImage()
        self.movie().frameChanged.connect(self._onFrameChanged)

    def paintEvent(self, e):
        if self.isNull():
            return

        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)

        path = QPainterPath()
        w, h = self.width(), self.height()

        # top line
        path.moveTo(self.topLeftRadius, 0)
        path.lineTo(w - self.topRightRadius, 0)

        # top right arc
        d = self.topRightRadius * 2
        path.arcTo(w - d, 0, d, d, 90, -90)

        # right line
        path.lineTo(w, h - self.bottomRightRadius)

        # bottom right arc
        d = self.bottomRightRadius * 2
        path.arcTo(w - d, h - d, d, d, 0, -90)

        # bottom line
        path.lineTo(self.bottomLeftRadius, h)

        # bottom left arc
        d = self.bottomLeftRadius * 2
        path.arcTo(0, h - d, d, d, -90, -90)

        # left line
        path.lineTo(0, self.topLeftRadius)

        # top left arc
        d = self.topLeftRadius * 2
        path.arcTo(0, 0, d, d, -180, -90)

        # draw image
        image = self.image.scaled(
            self.size() * self.devicePixelRatioF(),
            Qt.IgnoreAspectRatio,
            Qt.SmoothTransformation,
        )

        painter.setPen(Qt.NoPen)
        painter.setClipPath(path)
        painter.drawImage(self.rect(), image)

    @Property(int)
    def topLeftRadius(self):
        return self._topLeftRadius

    @topLeftRadius.setter
    def topLeftRadius(self, radius: int):
        self.setBorderRadius(
            radius, self.topRightRadius, self.bottomLeftRadius, self.bottomRightRadius
        )

    @Property(int)
    def topRightRadius(self):
        return self._topRightRadius

    @topRightRadius.setter
    def topRightRadius(self, radius: int):
        self.setBorderRadius(
            self.topLeftRadius, radius, self.bottomLeftRadius, self.bottomRightRadius
        )

    @Property(int)
    def bottomLeftRadius(self):
        return self._bottomLeftRadius

    @bottomLeftRadius.setter
    def bottomLeftRadius(self, radius: int):
        self.setBorderRadius(
            self.topLeftRadius, self.topRightRadius, radius, self.bottomRightRadius
        )

    @Property(int)
    def bottomRightRadius(self):
        return self._bottomRightRadius

    @bottomRightRadius.setter
    def bottomRightRadius(self, radius: int):
        self.setBorderRadius(
            self.topLeftRadius, self.topRightRadius, self.bottomLeftRadius, radius
        )


class AvatarWidget(ImageLabel):
    """Avatar widget

    Constructors
    ------------
    * AvatarWidget(`parent`: QWidget = None)
    * AvatarWidget(`image`: str | QImage | QPixmap, `parent`: QWidget = None)
    """

    def _postInit(self):
        self.setRadius(48)

    def getRadius(self):
        return self._radius

    def setRadius(self, radius: int):
        self._radius = radius
        self.setFixedSize(2 * radius, 2 * radius)
        self.update()

    def paintEvent(self, e):
        if self.isNull():
            return

        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)

        # center crop image
        image = self.image.scaled(
            self.size() * self.devicePixelRatioF(),
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation,
        )  # type: QImage

        iw, ih = image.width(), image.height()
        d = self.getRadius() * 2 * self.devicePixelRatioF()
        x, y = (iw - d) / 2, (ih - d) / 2
        image = image.copy(int(x), int(y), int(d), int(d))

        # draw image
        path = QPainterPath()
        path.addEllipse(QRectF(self.rect()))

        painter.setPen(Qt.NoPen)
        painter.setClipPath(path)
        painter.drawImage(self.rect(), image)

    radius = Property(int, getRadius, setRadius)


class HyperlinkLabel(QPushButton):
    """Hyperlink label

    Constructors
    ------------
    * HyperlinkLabel(`parent`: QWidget = None)
    * HyperlinkLabel(`text`: str, `parent`: QWidget = None)
    * HyperlinkLabel(`url`: QUrl, `parent`: QWidget = None)
    """

    @singledispatchmethod
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._url = QUrl()

        setFont(self, 14)
        self.setUnderlineVisible(False)
        FluentStyleSheet.LABEL.apply(self)
        self.setCursor(Qt.PointingHandCursor)
        self.clicked.connect(self._onClicked)

    @__init__.register
    def _(self, text: str, parent=None):
        self.__init__(parent)
        self.setText(text)

    @__init__.register
    def _(self, url: QUrl, text: str, parent=None):
        self.__init__(parent)
        self.setText(text)
        self._url = url

    def getUrl(self) -> QUrl:
        return self._url

    def setUrl(self, url: Union[QUrl, str]):
        self._url = QUrl(url)

    def isUnderlineVisible(self):
        return self._isUnderlineVisible

    def setUnderlineVisible(self, isVisible: bool):
        self._isUnderlineVisible = isVisible
        self.setProperty("underline", isVisible)
        self.setStyle(QApplication.style())

    def _onClicked(self):
        if self.getUrl().isValid():
            QDesktopServices.openUrl(self.getUrl())

    url = Property(QUrl, getUrl, setUrl)
    underlineVisible = Property(bool, isUnderlineVisible, setUnderlineVisible)

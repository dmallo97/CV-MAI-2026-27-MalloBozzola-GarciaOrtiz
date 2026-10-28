import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.face_detection import (
    BOX_COLOR,
    Face,
    FaceDetectionTool,
    bgr_to_pil,
    create_detector,
    detect_faces,
    draw_faces,
    parse_faces,
    pil_to_bgr,
)

# One YuNet output row: box (x, y, w, h), 5 landmarks (x, y), score
RAW_FACE = np.array([[10, 12, 20, 24, 15, 20, 25, 20, 20, 25, 16, 30, 24, 30, 0.9]], dtype=np.float32)


def _document(image: Image.Image) -> ImageDocument:
    document = ImageDocument()
    document.current = image
    return document


class ConversionTests(unittest.TestCase):
    def test_pil_to_bgr_swaps_red_and_blue(self) -> None:
        frame = pil_to_bgr(Image.new("RGB", (2, 2), (255, 0, 0)))
        self.assertEqual(tuple(frame[0, 0]), (0, 0, 255))

    def test_round_trip_keeps_pixels(self) -> None:
        image = Image.new("RGB", (3, 2), (10, 20, 30))
        self.assertEqual(bgr_to_pil(pil_to_bgr(image)).getpixel((0, 0)), (10, 20, 30))


class ParseFacesTests(unittest.TestCase):
    def test_none_means_no_faces(self) -> None:
        self.assertEqual(parse_faces(None), [])

    def test_splits_box_landmarks_and_score(self) -> None:
        face = parse_faces(RAW_FACE)[0]
        self.assertEqual(face.box, (10, 12, 20, 24))
        self.assertEqual(face.landmarks.shape, (5, 2))
        self.assertEqual(tuple(face.landmarks[2]), (20, 25))
        self.assertAlmostEqual(face.score, 0.9, places=5)


class DetectFacesTests(unittest.TestCase):
    def test_sets_input_size_to_frame_size(self) -> None:
        detector = MagicMock()
        detector.detect.return_value = (1, RAW_FACE)
        faces = detect_faces(detector, np.zeros((40, 60, 3), dtype=np.uint8))
        detector.setInputSize.assert_called_once_with((60, 40))
        self.assertEqual(len(faces), 1)

    def test_rejects_invalid_thresholds(self) -> None:
        with self.assertRaises(ValueError):
            create_detector(score_threshold=1.5)


class DrawFacesTests(unittest.TestCase):
    def test_draws_box_without_mutating_input(self) -> None:
        frame = np.zeros((60, 60, 3), dtype=np.uint8)
        output = draw_faces(frame, parse_faces(RAW_FACE))
        self.assertEqual(tuple(output[24, 10]), BOX_COLOR)  # left edge of the box
        self.assertEqual(frame.sum(), 0)


class FaceDetectionToolTests(unittest.TestCase):
    @patch("forensics_app.tools.face_detection.detect_faces")
    @patch("forensics_app.tools.face_detection.create_detector")
    @patch("forensics_app.tools.face_detection._ask_options", return_value=(0.6, 0.3))
    def test_returns_overlay_and_details(self, _ask_options, _create_detector, detect) -> None:
        detect.return_value = [Face((5, 5, 10, 10), np.full((5, 2), 8), 0.95)]
        document = _document(Image.new("RGB", (30, 30), "white"))

        result = FaceDetectionTool().run(None, document)

        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.details["Faces"], "1")
        self.assertEqual(result.details["Face 1 score"], "0.950")
        self.assertEqual(document.current.getpixel((5, 5)), (255, 255, 255))

    @patch("forensics_app.tools.face_detection._ask_options", return_value=None)
    def test_cancel_returns_none(self, _ask_options) -> None:
        self.assertIsNone(FaceDetectionTool().run(None, _document(Image.new("RGB", (8, 8)))))


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from skimage import data

from forensics_app.core import ImageDocument
from forensics_app.tools.feature_matching import (
    FeatureParams,
    KeypointsTool,
    extract_features,
    to_gray_float,
)


def _textured() -> Image.Image:
    return Image.fromarray(data.camera()).resize((200, 200))


def _document(image: Image.Image) -> ImageDocument:
    document = ImageDocument()
    document.current = image
    return document


class FeatureParamsTests(unittest.TestCase):
    def test_metric_depends_on_method(self) -> None:
        self.assertEqual(FeatureParams("orb").metric, "hamming")
        self.assertEqual(FeatureParams("sift").metric, "euclidean")

    def test_rejects_invalid_parameters(self) -> None:
        for kwargs in ({"method": "surf"}, {"n_keypoints": 0}, {"n_octaves": 0}, {"sigma_min": 0}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                FeatureParams(**kwargs)


class ExtractFeaturesTests(unittest.TestCase):
    def test_orb_respects_requested_keypoints(self) -> None:
        keypoints, descriptors = extract_features(to_gray_float(_textured()), FeatureParams("orb", n_keypoints=50))
        self.assertEqual(len(keypoints), 50)
        self.assertEqual(keypoints.shape[1], 2)
        self.assertEqual(descriptors.shape, (50, 256))

    def test_sift_descriptors_have_128_values(self) -> None:
        keypoints, descriptors = extract_features(to_gray_float(_textured()), FeatureParams("sift"))
        self.assertGreater(len(keypoints), 0)
        self.assertEqual(descriptors.shape, (len(keypoints), 128))

    def test_flat_image_has_no_features(self) -> None:
        flat = to_gray_float(Image.new("L", (100, 100), 128))
        for method in ("orb", "sift"):
            with self.subTest(method=method):
                keypoints, descriptors = extract_features(flat, FeatureParams(method))
                self.assertEqual(len(keypoints), 0)
                self.assertEqual(len(descriptors), 0)

    def test_rejects_color_array(self) -> None:
        with self.assertRaises(ValueError):
            extract_features(np.zeros((10, 10, 3)), FeatureParams())


class KeypointsToolTests(unittest.TestCase):
    @patch("forensics_app.tools.feature_matching.ask_feature_params", return_value=FeatureParams("orb", n_keypoints=30))
    def test_draws_keypoints_and_reports_details(self, _ask_params) -> None:
        document = _document(_textured())
        original = document.current.copy()

        result = KeypointsTool().run(None, document)

        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.image.size, original.size)
        self.assertEqual(result.details["Descriptor"], "ORB")
        self.assertEqual(result.details["Keypoints"], "30")
        self.assertEqual(result.details["Descriptor length"], "256")
        self.assertEqual(list(document.current.getdata()), list(original.getdata()))

    @patch("forensics_app.tools.feature_matching.ask_feature_params", return_value=None)
    def test_cancel_returns_none(self, _ask_params) -> None:
        self.assertIsNone(KeypointsTool().run(None, _document(_textured())))


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import data, img_as_ubyte, transform

from forensics_app.tools.feature_matching import FeatureParams
from forensics_app.tools.geometric_verification import (
    RansacParams,
    VerificationResult,
    is_plausible_homography,
    verify_matches,
)
from forensics_app.tools.image_lookup import (
    Candidate,
    LookupSettings,
    list_image_files,
    limit_size,
    prepare_reference,
    rank_candidates,
    search_folder,
)

# A mild perspective change, like photographing the same scene from another spot.
TRUE_HOMOGRAPHY = np.array([[0.9, -0.1, 30.0], [0.08, 0.95, 15.0], [1e-4, 0.0, 1.0]])


def synthetic_correspondences(n_points: int, n_outliers: int, seed: int = 0):
    """Keypoints (row, col) related by TRUE_HOMOGRAPHY, the first `n_outliers` corrupted."""
    rng = np.random.default_rng(seed)
    xy1 = rng.uniform(0, 400, size=(n_points, 2))
    xy2 = transform.ProjectiveTransform(TRUE_HOMOGRAPHY)(xy1)
    xy2[:n_outliers] = rng.uniform(0, 400, size=(n_outliers, 2))  # random wrong matches
    matches = np.column_stack([np.arange(n_points), np.arange(n_points)])
    return xy1[:, ::-1], xy2[:, ::-1], matches  # back to scikit-image (row, col)


class VerifyMatchesTests(unittest.TestCase):
    def test_ransac_separates_inliers_from_outliers(self) -> None:
        keypoints1, keypoints2, matches = synthetic_correspondences(60, n_outliers=20)
        result = verify_matches(keypoints1, keypoints2, matches, RansacParams())
        self.assertTrue(result.reliable)
        self.assertTrue(result.inliers[20:].all())  # every true match kept
        self.assertFalse(result.inliers[:20].any())  # every corrupted match rejected
        np.testing.assert_allclose(
            result.model.params / result.model.params[2, 2], TRUE_HOMOGRAPHY, atol=1e-3
        )

    def test_too_few_matches_is_not_reliable(self) -> None:
        keypoints1, keypoints2, matches = synthetic_correspondences(3, n_outliers=0)
        result = verify_matches(keypoints1, keypoints2, matches, RansacParams())
        self.assertIsNone(result.model)
        self.assertFalse(result.reliable)
        self.assertEqual(result.n_inliers, 0)

    def test_mirrored_homography_is_rejected(self) -> None:
        mirror = transform.ProjectiveTransform(np.diag([-1.0, 1.0, 1.0]))
        self.assertFalse(is_plausible_homography(mirror))
        self.assertTrue(is_plausible_homography(transform.ProjectiveTransform(TRUE_HOMOGRAPHY)))

    def test_min_inliers_cannot_be_below_min_samples(self) -> None:
        with self.assertRaises(ValueError):
            RansacParams("fundamental", min_inliers=5)


class RankingTests(unittest.TestCase):
    @staticmethod
    def make_candidate(name: str, n_matches: int, n_inliers: int | None, reliable: bool = True):
        matches = np.zeros((n_matches, 2), dtype=np.intp)
        verification = None
        if n_inliers is not None:
            inliers = np.arange(n_matches) < n_inliers
            verification = VerificationResult(inliers, None, reliable)
        return Candidate(Path(name), np.empty((0, 2)), matches, verification)

    def test_baseline_ranks_by_number_of_matches(self) -> None:
        ranking = rank_candidates([
            self.make_candidate("a.png", 10, None),
            self.make_candidate("b.png", 30, None),
        ])
        self.assertEqual([c.path.name for c in ranking], ["b.png", "a.png"])

    def test_ransac_prefers_consistent_matches_over_many_matches(self) -> None:
        noisy = self.make_candidate("noisy.png", 80, 6, reliable=False)  # many random matches
        good = self.make_candidate("good.png", 40, 35)
        ranking = rank_candidates([noisy, good])
        self.assertEqual(ranking[0].path.name, "good.png")
        self.assertEqual(noisy.score, 0)


class FolderTests(unittest.TestCase):
    def test_list_image_files_filters_and_excludes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name in ("b.JPG", "a.png", "notes.txt", "reference.png"):
                (folder / name).write_bytes(b"")
            files = list_image_files(folder, exclude=folder / "reference.png")
            self.assertEqual([path.name for path in files], ["a.png", "b.JPG"])

    def test_limit_size_never_enlarges(self) -> None:
        self.assertEqual(limit_size(np.zeros((100, 50)), 800).shape, (100, 50))
        self.assertEqual(limit_size(np.zeros((1600, 800)), 800).shape, (800, 400))

    def test_warped_copy_is_found_among_other_images(self) -> None:
        reference = data.camera() / 255.0
        warped = transform.warp(reference, transform.ProjectiveTransform(TRUE_HOMOGRAPHY).inverse)
        settings = LookupSettings(FeatureParams("orb", n_keypoints=300), 0.8, True, RansacParams())
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            Image.fromarray(img_as_ubyte(warped)).save(folder / "same_scene.png")
            Image.fromarray(data.coins()).save(folder / "coins.png")
            Image.fromarray(img_as_ubyte(data.astronaut())).save(folder / "astronaut.png")
            (folder / "broken.png").write_bytes(b"not an image")

            ranking, skipped = search_folder(
                list_image_files(folder), prepare_reference(reference, settings), settings
            )
        self.assertEqual(ranking[0].path.name, "same_scene.png")
        self.assertTrue(ranking[0].verification.reliable)
        self.assertEqual([path.name for path in skipped], ["broken.png"])


if __name__ == "__main__":
    unittest.main()
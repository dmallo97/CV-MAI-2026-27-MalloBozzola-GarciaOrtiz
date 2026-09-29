import unittest

from PIL import Image

from forensics_app.tools.channel_split import (
    _parse_selection,
    extract_channels,
)


def _rgb_sample() -> Image.Image:
    image = Image.new("RGB", (2, 2))
    image.putpixel((0, 0), (10, 20, 30))
    image.putpixel((1, 0), (40, 50, 60))
    image.putpixel((0, 1), (70, 80, 90))
    image.putpixel((1, 1), (100, 110, 120))
    return image


class ExtractChannelsTests(unittest.TestCase):
    def test_single_channel_returns_grayscale(self) -> None:
        result = extract_channels(_rgb_sample(), ["R"])
        self.assertEqual(result.mode, "L")
        self.assertEqual(result.size, (2, 2))
        self.assertEqual(result.getpixel((0, 0)), 10)
        self.assertEqual(result.getpixel((1, 1)), 100)

    def test_two_channels_zero_the_others(self) -> None:
        result = extract_channels(_rgb_sample(), ["R", "G"])
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.getpixel((0, 0)), (10, 20, 0))
        self.assertEqual(result.getpixel((1, 1)), (100, 110, 0))

    def test_all_channels_reproduce_input(self) -> None:
        source = _rgb_sample()
        result = extract_channels(source, ["R", "G", "B"])
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(list(result.getdata()), list(source.getdata()))

    def test_non_rgb_input_is_converted(self) -> None:
        gray = Image.new("L", (2, 2), 128)
        result = extract_channels(gray, ["R"])
        self.assertEqual(result.mode, "L")
        self.assertEqual(result.getpixel((0, 0)), 128)
        self.assertEqual(gray.mode, "L")

    def test_input_image_is_not_mutated(self) -> None:
        source = _rgb_sample()
        pixels_before = list(source.getdata())
        extract_channels(source, ["G"])
        self.assertEqual(list(source.getdata()), pixels_before)

    def test_empty_selection_raises(self) -> None:
        with self.assertRaises(ValueError):
            extract_channels(_rgb_sample(), [])

    def test_unknown_channel_raises(self) -> None:
        with self.assertRaises(ValueError):
            extract_channels(_rgb_sample(), ["X"])


class ParseSelectionTests(unittest.TestCase):
    def test_normalises_case_and_whitespace(self) -> None:
        self.assertEqual(_parse_selection(" rG b "), ["R", "G", "B"])

    def test_accepts_commas(self) -> None:
        self.assertEqual(_parse_selection("R,G"), ["R", "G"])

    def test_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            _parse_selection("   ")

    def test_rejects_duplicates(self) -> None:
        with self.assertRaises(ValueError):
            _parse_selection("RR")

    def test_rejects_unknown_letters(self) -> None:
        with self.assertRaises(ValueError):
            _parse_selection("RGX")


if __name__ == "__main__":
    unittest.main()

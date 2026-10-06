import unittest

from PIL import Image

from forensics_app.tools.channel_swap import (
    _parse_order,
    swap_channels,
)


def _rgb_sample() -> Image.Image:
    image = Image.new("RGB", (2, 2))
    image.putpixel((0, 0), (10, 20, 30))
    image.putpixel((1, 0), (40, 50, 60))
    image.putpixel((0, 1), (70, 80, 90))
    image.putpixel((1, 1), (100, 110, 120))
    return image


class SwapChannelsTests(unittest.TestCase):
    def test_grb_swaps_red_and_green(self) -> None:
        result = swap_channels(_rgb_sample(), ["G", "R", "B"])
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.getpixel((0, 0)), (20, 10, 30))
        self.assertEqual(result.getpixel((1, 1)), (110, 100, 120))

    def test_bgr_reverses_channels(self) -> None:
        result = swap_channels(_rgb_sample(), ["B", "G", "R"])
        self.assertEqual(result.getpixel((1, 0)), (60, 50, 40))

    def test_identity_order_reproduces_input(self) -> None:
        source = _rgb_sample()
        result = swap_channels(source, ["R", "G", "B"])
        self.assertEqual(list(result.getdata()), list(source.getdata()))

    def test_non_rgb_input_is_converted(self) -> None:
        gray = Image.new("L", (2, 2), 128)
        result = swap_channels(gray, ["G", "R", "B"])
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.getpixel((0, 0)), (128, 128, 128))
        self.assertEqual(gray.mode, "L")

    def test_input_image_is_not_mutated(self) -> None:
        source = _rgb_sample()
        pixels_before = list(source.getdata())
        swap_channels(source, ["B", "R", "G"])
        self.assertEqual(list(source.getdata()), pixels_before)

    def test_invalid_orders_raise(self) -> None:
        for order in (["R", "R", "B"], ["R", "G"], ["X", "G", "B"], []):
            with self.subTest(order=order), self.assertRaises(ValueError):
                swap_channels(_rgb_sample(), order)


class ParseOrderTests(unittest.TestCase):
    def test_normalises_case_and_whitespace(self) -> None:
        self.assertEqual(_parse_order(" g r b "), ["G", "R", "B"])

    def test_accepts_commas(self) -> None:
        self.assertEqual(_parse_order("B,G,R"), ["B", "G", "R"])

    def test_two_letters_swap_that_pair(self) -> None:
        self.assertEqual(_parse_order("RG"), ["G", "R", "B"])
        self.assertEqual(_parse_order("bg"), ["R", "B", "G"])

    def test_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            _parse_order("   ")

    def test_rejects_duplicates(self) -> None:
        with self.assertRaises(ValueError):
            _parse_order("RRB")

    def test_rejects_unknown_letters(self) -> None:
        with self.assertRaises(ValueError):
            _parse_order("RGX")

    def test_rejects_wrong_length(self) -> None:
        for raw in ("R", "RGBR"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                _parse_order(raw)


if __name__ == "__main__":
    unittest.main()

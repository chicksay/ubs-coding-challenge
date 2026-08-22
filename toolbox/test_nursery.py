import base64
import io
import unittest

from PIL import Image, ImageDraw

from services import nursery


def _encode(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _canvas(size=100):
    return Image.new("RGB", (size, size), "white")


class NameTests(unittest.TestCase):
    def test_get_name_meets_criteria(self):
        name = nursery.get_name({})
        self.assertTrue(3 <= len(name) <= 30)
        self.assertRegex(name, r"^[A-Za-z0-9 _\-']+$")


class CalculateTests(unittest.TestCase):
    def test_symbolic_expression(self):
        self.assertEqual(nursery.calculate({"expression": "2 + 2 + 5"}), 9)

    def test_precedence(self):
        self.assertEqual(nursery.calculate({"expression": "2 + 3 * 4"}), 14)

    def test_natural_language(self):
        self.assertEqual(
            nursery.calculate({"expression": "What is 2 plus 3 times 4?"}), 14
        )

    def test_negative_operand(self):
        self.assertEqual(nursery.calculate({"expression": "-5 + 10"}), 5)

    def test_division(self):
        self.assertEqual(nursery.calculate({"expression": "10 / 2"}), 5)

    def test_a_op_b_fallback(self):
        self.assertEqual(nursery.calculate({"a": 3, "op": "*", "b": 4}), 12)

    def test_rejects_non_arithmetic(self):
        with self.assertRaises(ValueError):
            nursery.calculate({"expression": "__import__('os')"})


class ShapeTests(unittest.TestCase):
    def test_rectangle(self):
        image = _canvas()
        ImageDraw.Draw(image).rectangle([20, 20, 80, 80], fill="black")
        self.assertEqual(nursery.identify_shape({"image": _encode(image)}), "rectangle")

    def test_circle(self):
        image = _canvas()
        ImageDraw.Draw(image).ellipse([20, 20, 80, 80], fill="black")
        self.assertEqual(nursery.identify_shape({"image": _encode(image)}), "circle")

    def test_triangle(self):
        image = _canvas()
        ImageDraw.Draw(image).polygon([(50, 15), (15, 85), (85, 85)], fill="black")
        self.assertEqual(nursery.identify_shape({"image": _encode(image)}), "triangle")

    def test_count_sides_from_shape_name(self):
        self.assertEqual(nursery.count_sides({"shape": "Triangle"}), 3)
        self.assertEqual(nursery.count_sides({"shape": "rectangle"}), 4)
        self.assertEqual(nursery.count_sides({"shape": "circle"}), 0)

    def test_count_sides_from_image(self):
        image = _canvas()
        ImageDraw.Draw(image).rectangle([20, 20, 80, 80], fill="black")
        self.assertEqual(nursery.count_sides({"image": _encode(image)}), 4)


class OrderOfOperationsTests(unittest.TestCase):
    def test_returns_a_string(self):
        self.assertIsInstance(nursery.order_of_operations({}), str)


if __name__ == "__main__":
    unittest.main()

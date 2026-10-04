import unittest

from make_long_probe import cut_probe


class LongProbeTests(unittest.TestCase):
    def test_cut_preserves_source_and_last_word_boundary(self):
        text = "one\n\ntwo\tthree four five six"
        probe, count = cut_probe(text, lambda value: len(value.split()) + 2, 4, 6)
        self.assertEqual(probe, "one\n\ntwo\tthree four")
        self.assertEqual(count, 6)
        self.assertTrue(text.startswith(probe))

    def test_source_below_minimum_is_refused(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            cut_probe("one two", lambda value: len(value.split()) + 2, 6, 8)

    def test_empty_source_is_refused(self):
        with self.assertRaisesRegex(ValueError, "no word boundary"):
            cut_probe(" \n", lambda value: len(value.split()) + 2, 6, 8)


if __name__ == "__main__":
    unittest.main()

import unittest

from benchmark_suite import table


class SuiteTableTests(unittest.TestCase):
    def test_markdown_table(self) -> None:
        text = table([{"name": "lenet on mnist", "nanograd": 12.0, "torch": 6.0, "difference": 3e-17}])
        lines = text.splitlines()
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[2], "| lenet on mnist | 12.0 | 6.0 | 2.0x | 3.0e-17 |")


if __name__ == "__main__":
    unittest.main()

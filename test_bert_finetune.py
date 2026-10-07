import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from bert_finetune import parse_args, run


SPEECHES = "".join(
    f"{name}:\n{line}\n\n"
    for _ in range(12)
    for name, line in (("ALPHA", "Good morrow, my lord, good morrow."), ("BETA", "Nay, nay, I will not go; away!"))
)


class BertFinetuneTests(unittest.TestCase):
    def test_finetune_and_probe_run_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "text.txt")
            path.write_text(SPEECHES, encoding="utf-8")
            for extra in ([], ["--probe"], ["--lora", "2"]):
                args = parse_args(["--text-file", str(path), "--speakers", "2", "--pretrain-steps", "2",
                                   "--finetune-steps", "2", "--batch-size", "2", "--context-size", "16",
                                   "--embedding-dim", "8", "--heads", "2", "--layers", "1",
                                   "--eval-windows", "10", "--report-every", "1", *extra])
                with contextlib.redirect_stdout(io.StringIO()):
                    results = run(args)
                self.assertEqual(set(results), {"pretrained", "untrained"})
                self.assertTrue(all(0.0 <= value <= 1.0 for value in results.values()))


if __name__ == "__main__":
    unittest.main()

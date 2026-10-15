import json
import struct
import tempfile
import unittest
from array import array
from pathlib import Path

import gpt2

GPT2_DIR = Path(__file__).parent / "data" / "gpt2"

try:
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
except ImportError:
    torch = None


def write_safetensors(path: Path, tensors: dict) -> None:
    """{name: (shape, values)} -> a float32 safetensors file."""

    header, blob = {}, b""
    for name, (shape, values) in tensors.items():
        data = array("f", values).tobytes()
        header[name] = {"dtype": "F32", "shape": list(shape), "data_offsets": [len(blob), len(blob) + len(data)]}
        blob += data
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + blob)


class SafetensorsTest(unittest.TestCase):
    def test_reads_names_shapes_and_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            write_safetensors(path, {"a": ((2, 3), [1, 2, 3, 4, 5, 6]), "b": ((1,), [0.5])})
            tensors = gpt2.read_safetensors(path)
        self.assertEqual(tensors["a"], ((2, 3), array("f", [1, 2, 3, 4, 5, 6])))
        self.assertEqual(tensors["b"], ((1,), array("f", [0.5])))

    def test_transposed_rows_are_the_columns(self):
        rows = gpt2.transposed_rows((2, 3), array("f", [1, 2, 3, 4, 5, 6]))
        self.assertEqual([list(row) for row in rows], [[1, 4], [2, 5], [3, 6]])


class TokenizerTest(unittest.TestCase):
    def test_byte_table_is_a_bijection_onto_printable_characters(self):
        table = gpt2.bytes_to_unicode()
        self.assertEqual(len(set(table.values())), 256)
        self.assertEqual(table[ord("A")], "A")
        self.assertEqual(table[ord(" ")], "Ġ")                     # the space GPT-2's vocabulary shows

    @unittest.skipUnless((GPT2_DIR / "vocab.json").exists(), "GPT-2 files not downloaded (gpt2.py --download)")
    def test_known_ids_and_round_trip(self):
        tokenizer = gpt2.GPT2Tokenizer(GPT2_DIR)
        self.assertEqual(tokenizer.encode("Hello world"), [15496, 995])
        for text in ["naïve café, it's 3:14!\n\n  spaces\ttabs 日本語 🙂", "x_y __init__"]:
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)


@unittest.skipIf(torch is None, "torch and transformers are not installed")
class AgainstHuggingFaceTest(unittest.TestCase):
    def test_tiny_random_gpt2_gives_the_same_logits(self):
        config = GPT2Config(vocab_size=50, n_positions=16, n_embd=32, n_head=4, n_layer=2)
        torch.manual_seed(0)
        reference = GPT2LMHeadModel(config).double().eval()
        with tempfile.TemporaryDirectory() as directory:
            reference.float().save_pretrained(directory)            # float32 on disk, as OpenAI's
            reference.double()
            model = gpt2.load_gpt2(Path(directory))
        ids = [3, 1, 4, 1, 5, 9, 2, 6]
        expected = reference(torch.tensor([ids])).logits.reshape(-1).tolist()
        model.eval()
        got = list(model.sequence_logits([ids]).data)
        self.assertEqual(len(got), len(expected))
        self.assertLess(max(abs(a - b) for a, b in zip(got, expected)), 1e-6)


if __name__ == "__main__":
    unittest.main()

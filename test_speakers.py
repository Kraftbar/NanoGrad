import random
import unittest

from bert import BERTClassifier, TinyBERT
from speakers import speaker_data, speech_windows

TEXT = "".join(
    f"{name}:\n{line} number {index}.\n\n"
    for index in range(20)
    for name, line in (("ROMEO", "But soft, what light"), ("JULIET", "O Romeo, Romeo"), ("NURSE", "Faith, I can tell"))
)


class SpeakerTests(unittest.TestCase):
    def test_split_per_speaker_and_held_out_text_is_removed(self) -> None:
        data = speaker_data(TEXT, num_speakers=2, validation_fraction=0.25)
        self.assertEqual(len(data.speakers), 2)
        self.assertEqual(len(data.train), 30)
        self.assertEqual(len(data.validation), 10)
        self.assertEqual({label for _, label in data.validation}, {0, 1})
        for body, _ in data.validation:
            self.assertNotIn(body, data.pretraining_text)
        for body, _ in data.train:
            self.assertIn(body, data.pretraining_text)

    def test_windows_have_the_requested_length(self) -> None:
        data = speaker_data(TEXT, num_speakers=3)
        texts, labels = speech_windows(data.train, length=50, count=7, rng=random.Random(1))
        self.assertEqual([len(text) for text in texts], [50] * 7)
        self.assertTrue(all(0 <= label < 3 for label in labels))

    def test_classifier_shapes_and_parameters(self) -> None:
        encoder = TinyBERT(10, context_size=8, embedding_dim=8, num_heads=2, num_layers=1)
        model = BERTClassifier(encoder, 3)
        self.assertEqual(model([[1, 2, 3, 4], [5, 6, 7, 8]]).shape, (2, 3))
        head = {id(parameter) for parameter in encoder.head.parameters()}
        self.assertFalse(any(id(parameter) in head for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()

import copy
import math
import unittest

import dpo
from language import CharTransformerModel
from tensor import Tensor, no_grad


class DPOTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = CharTransformerModel(7, context_size=8, embedding_dim=8, num_heads=2, seed=1)

    def test_word_score_counts_only_complete_words(self) -> None:
        known = {"the", "king"}
        self.assertEqual(dpo.word_score("xthe king zorp th", known), 0)    # "king" +1, "zorp" -1; ends cut off
        self.assertEqual(dpo.word_score(" the king ", known), 2)

    def test_sequence_log_probs_sum_the_continuation(self) -> None:
        sequences = [[1, 2, 3, 4, 5], [0, 6, 2, 2, 1]]
        got = dpo.sequence_log_probs(self.model, sequences, prompt_length=2).data
        with no_grad():
            for b, sequence in enumerate(sequences):
                expected = 0.0
                for t in range(2, len(sequence)):
                    logits = self.model.sequence_logits(Tensor.from_list([sequence[:t]])).data[-7:]
                    top = max(logits)
                    expected += logits[sequence[t]] - top - math.log(sum(math.exp(v - top) for v in logits))
                self.assertAlmostEqual(got[b], expected, places=10)

    def test_loss_starts_at_log_two_and_moves_toward_the_winners(self) -> None:
        reference = copy.deepcopy(self.model)
        winners, losers = [[1, 2, 3, 3, 3]], [[1, 2, 5, 6, 0]]
        loss, _ = dpo.dpo_loss(self.model, reference, winners, losers, 2, beta=0.5)
        self.assertAlmostEqual(loss.item(), math.log(2), places=12)        # policy = reference: margin 0
        from optim import TensorAdam

        optimizer = TensorAdam(self.model.parameters(), lr=1e-2)
        for _ in range(20):
            optimizer.zero_grad()
            loss, _ = dpo.dpo_loss(self.model, reference, winners, losers, 2, beta=0.5)
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), 0.3)


if __name__ == "__main__":
    unittest.main()

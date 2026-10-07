import importlib.util
import random
import unittest

from bert import TinyBERT, mask_tokens, masked_loss
from losses import softmax_cross_entropy
from mnist_compare import IMAGE_SIZE, NanoGradRunner, build_nanograd
from resnet import SmallResNet
from tensor import Tensor, conv2d, no_grad, pad2d
from tensor_nn import TensorBatchNorm2d
from train import evaluation_mode
from vit import TinyViT


def random_batch(count, seed):
    rng = random.Random(seed)
    return [rng.random() for _ in range(count * IMAGE_SIZE)], [rng.randrange(10) for _ in range(count)]


class ShapeTests(unittest.TestCase):
    def test_padding_and_stride_shapes(self) -> None:
        image = Tensor([1.0] * (2 * 3 * 7 * 7), (2, 3, 7, 7))
        kernel = Tensor([0.1] * (4 * 3 * 3 * 3), (4, 3, 3, 3))
        self.assertEqual(pad2d(image, 1).shape, (2, 3, 9, 9))
        self.assertEqual(conv2d(image, kernel, padding=1).shape, (2, 4, 7, 7))
        self.assertEqual(conv2d(image, kernel, stride=2, padding=1).shape, (2, 4, 4, 4))

    def test_pad2d_puts_zeros_around_the_image(self) -> None:
        image = Tensor([1.0, 2.0, 3.0, 4.0], (2, 2))
        self.assertEqual(pad2d(image, 1).tolist(), [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 0.0],
            [0.0, 3.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])

    def test_models_classify_a_batch(self) -> None:
        pixels, _ = random_batch(3, seed=1)
        images = Tensor(pixels, (3, 1, 28, 28))
        for model in (SmallResNet(), TinyViT()):
            with self.subTest(model=type(model).__name__):
                self.assertEqual(model(images).shape, (3, 10))


class BatchNormTests(unittest.TestCase):
    def test_training_mode_normalizes_each_channel(self) -> None:
        rng = random.Random(2)
        image = Tensor([rng.uniform(-3.0, 5.0) for _ in range(4 * 2 * 3 * 3)], (4, 2, 3, 3))
        out = TensorBatchNorm2d(2)(image)
        for channel in range(2):
            values = [out[b, channel, r, c] for b in range(4) for r in range(3) for c in range(3)]
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            self.assertAlmostEqual(mean, 0.0, places=12)
            self.assertAlmostEqual(variance, 1.0, places=4)

    def test_running_statistics_round_trip(self) -> None:
        model = SmallResNet(seed=3)
        pixels, labels = random_batch(4, seed=4)
        softmax_cross_entropy(model(Tensor(pixels, (4, 1, 28, 28))), labels).backward()
        copy = SmallResNet(seed=99)
        copy.load_state_dict(model.state_dict())
        images = Tensor(pixels, (4, 1, 28, 28))
        with no_grad(), evaluation_mode(model), evaluation_mode(copy):
            self.assertEqual(list(model(images).data), list(copy(images).data))

    def test_vit_state_round_trip(self) -> None:
        model, copy = TinyViT(seed=5), TinyViT(seed=6)
        copy.load_state_dict(model.state_dict())
        pixels, _ = random_batch(2, seed=7)
        images = Tensor(pixels, (2, 1, 28, 28))
        self.assertEqual(list(model(images).data), list(copy(images).data))


class BertTests(unittest.TestCase):
    def test_mask_tokens_follows_the_recipe(self) -> None:
        rng = random.Random(8)
        sequences = [[rng.randrange(10) for _ in range(50)] for _ in range(40)]
        inputs, positions, targets = mask_tokens(sequences, vocab_size=10, mask_id=10, rng=rng)
        flat_inputs = [token for row in inputs for token in row]
        flat_originals = [token for row in sequences for token in row]
        self.assertEqual(targets, [flat_originals[p] for p in positions])
        masked = sum(flat_inputs[p] == 10 for p in positions)
        self.assertAlmostEqual(len(positions) / len(flat_inputs), 0.15, delta=0.03)
        self.assertAlmostEqual(masked / len(positions), 0.8, delta=0.06)
        untouched = set(range(len(flat_inputs))) - set(positions)
        self.assertTrue(all(flat_inputs[i] == flat_originals[i] for i in untouched))

    def test_masked_loss_trains(self) -> None:
        model = TinyBERT(6, context_size=8, embedding_dim=8, num_heads=2, num_layers=1)
        inputs, positions, targets = mask_tokens([[1, 2, 3, 4, 5, 0, 1, 2]] * 3, vocab_size=6, mask_id=6, rng=random.Random(9))
        loss = masked_loss(model, inputs, positions, targets)
        loss.backward()
        self.assertTrue(any(value != 0.0 for value in model.head.weight.grad))


class LlamaPartsTests(unittest.TestCase):
    def test_kv_cache_matches_full_forward_with_rope_rmsnorm_swiglu(self) -> None:
        from language import CharTransformerModel

        model = CharTransformerModel(11, context_size=8, embedding_dim=16, num_heads=2, num_layers=2,
                                     feed_forward_activation="swiglu", norm="rms", positions="rope")
        rng = random.Random(11)
        tokens = [rng.randrange(11) for _ in range(8)]
        with no_grad():
            cache = model.new_cache()
            for position, token in enumerate(tokens):
                step = list(model.step_logits(token, position, cache).data)
                full = list(model(Tensor.from_list([tokens[: position + 1]])).data)
                for a, b in zip(step, full):
                    self.assertAlmostEqual(a, b, places=12)

    def test_grouped_query_attention_kv_cache_matches_full_forward(self) -> None:
        from language import CharTransformerModel

        for kv_heads in (2, 1):
            model = CharTransformerModel(11, context_size=8, embedding_dim=16, num_heads=4, num_kv_heads=kv_heads,
                                         num_layers=2, feed_forward_activation="swiglu", norm="rms", positions="rope")
            self.assertEqual(len(model.new_cache()[0]["keys"]), kv_heads)
            rng = random.Random(12)
            tokens = [rng.randrange(11) for _ in range(8)]
            with no_grad():
                cache = model.new_cache()
                for position, token in enumerate(tokens):
                    step = list(model.step_logits(token, position, cache).data)
                    full = list(model(Tensor.from_list([tokens[: position + 1]])).data)
                    for a, b in zip(step, full):
                        self.assertAlmostEqual(a, b, places=12)

    def test_grouped_query_attention_equals_heads_with_copied_keys(self) -> None:
        """Sharing a key/value head is the same as giving every query head
        its own copy of it: GQA is a constraint on ordinary attention."""

        from language import CausalSelfAttention

        grouped = CausalSelfAttention(8, 5, num_heads=4, num_kv_heads=2, rotary=True, seed=3)
        full = CausalSelfAttention(8, 5, num_heads=4, rotary=True, seed=9)
        for head in range(4):
            full.queries[head].load_state_dict(grouped.queries[head].state_dict())
            full.keys[head].load_state_dict(grouped.keys[head // 2].state_dict())
            full.values[head].load_state_dict(grouped.values[head // 2].state_dict())
        full.projection.load_state_dict(grouped.projection.state_dict())
        inputs = Tensor([random.Random(4).uniform(-1, 1) for _ in range(2 * 5 * 8)], (2, 5, 8))
        for a, b in zip(grouped(inputs).data, full(inputs).data):
            self.assertAlmostEqual(a, b, places=12)
        self.assertEqual(grouped.num_parameters(), full.num_parameters() - 2 * 2 * (8 * 2 + 2))

    def test_grouped_query_attention_rejects_uneven_groups(self) -> None:
        from language import CausalSelfAttention

        with self.assertRaises(ValueError):
            CausalSelfAttention(8, 4, num_heads=4, num_kv_heads=3)

    def test_llama_style_state_round_trip(self) -> None:
        from language import CharTransformerModel

        settings = dict(context_size=8, embedding_dim=16, num_heads=2, num_layers=1,
                        feed_forward_activation="swiglu", norm="rms", positions="rope")
        model, copy = CharTransformerModel(9, seed=1, **settings), CharTransformerModel(9, seed=2, **settings)
        copy.load_state_dict(model.state_dict())
        inputs = Tensor.from_list([[1, 2, 3, 4]])
        self.assertEqual(list(model(inputs).data), list(copy(inputs).data))
        gpt2_style = CharTransformerModel(9, context_size=8, embedding_dim=16, num_heads=2, num_layers=1)
        with self.assertRaises(ValueError):
            gpt2_style.load_state_dict(model.state_dict())


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class TorchAgreementTests(unittest.TestCase):
    def test_gpt2_and_llama_parts_match_pytorch(self) -> None:
        import torch

        from language import CharTransformerModel
        from torch_twin import TorchTwin

        for norm, positions, activation, kv_heads in (
            ("layer", "learned", "gelu", 4), ("rms", "rope", "swiglu", 4), ("rms", "rope", "swiglu", 2),
            ("layer", "learned", "gelu", 1),
        ):
            with self.subTest(norm=norm, positions=positions, activation=activation, kv_heads=kv_heads):
                model = CharTransformerModel(
                    13, context_size=10, embedding_dim=16, num_heads=4, num_kv_heads=kv_heads, num_layers=2,
                    feed_forward_activation=activation, tie_embeddings=True, init="gpt2",
                    norm=norm, positions=positions, seed=1,
                )
                twin = TorchTwin(model)
                rng = random.Random(2)
                indices = [[rng.randrange(13) for _ in range(10)] for _ in range(3)]
                ours = model.sequence_logits(Tensor.from_list(indices))
                theirs = twin(torch.tensor(indices))
                weights = [rng.uniform(-1.0, 1.0) for _ in range(ours.numel)]
                (ours * Tensor(weights, ours.shape)).sum().backward()
                (theirs * torch.tensor(weights, dtype=torch.float64).reshape(theirs.shape)).sum().backward()
                for a, b in zip(ours.data, theirs.detach().reshape(-1).tolist()):
                    self.assertAlmostEqual(a, b, places=12)
                for parameter, tensor in zip(model.parameters(), twin.parameters):
                    for a, b in zip(parameter.grad, tensor.grad.reshape(-1).tolist()):
                        self.assertAlmostEqual(a, b, places=11)

    def test_bert_trains_like_pytorch(self) -> None:
        from bert_compare import NanoGradRunner as BertRunner
        from bert_compare import TorchRunner as BertTorchRunner

        model = TinyBERT(7, context_size=10, embedding_dim=8, num_heads=2, num_layers=2, seed=4)
        theirs = BertTorchRunner(model, lr=1e-3, threads=1)
        ours = BertRunner(model, lr=1e-3)
        rng = random.Random(10)
        for _ in range(3):
            windows = [[rng.randrange(7) for _ in range(10)] for _ in range(3)]
            batch = mask_tokens(windows, vocab_size=7, mask_id=7, rng=rng)
            self.assertAlmostEqual(ours.train_step(*batch), theirs.train_step(*batch), places=10)
        self.assertEqual(ours.predict(*batch[:2]), theirs.predict(*batch[:2]))

    def test_every_model_trains_like_pytorch(self) -> None:
        from mnist_compare import TorchRunner

        for name in ("lenet", "resnet", "vit"):
            with self.subTest(model=name):
                model = build_nanograd(3, name)
                torch_runner = TorchRunner(model, lr=1e-3, dtype="float64", threads=1)
                ours = NanoGradRunner(model, lr=1e-3)
                for step in range(3):
                    pixels, labels = random_batch(4, seed=10 + step)
                    self.assertAlmostEqual(ours.train_step(pixels, labels), torch_runner.train_step(pixels, labels), places=10)
                for left, right in zip(ours.weights(), torch_runner.weights()):
                    for a, b in zip(left, right):
                        self.assertAlmostEqual(a, b, places=10)
                # Evaluation mode, including batch-norm running statistics.
                pixels, _ = random_batch(5, seed=20)
                self.assertEqual(ours.predict(pixels, 5), torch_runner.predict(pixels, 5))
                ours_running = [
                    values
                    for module in model.modules() if isinstance(module, TensorBatchNorm2d)
                    for values in (module.running_mean, module.running_var)
                ]
                for left, right in zip(ours_running, torch_runner.twin.running_statistics()):
                    for a, b in zip(left, right):
                        self.assertAlmostEqual(a, b, places=10)


if __name__ == "__main__":
    unittest.main()

# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Loss regressions from PyTorch e2c2673d00500d268fcb1cc63fa7cc3173876505.

Removed in 6dca9f63bc4975ac323044a440c2c7e5f8512408. Preserve the
original rank/reduction/weight/ignored-target matrices with modern capture.
"""

import torch
from test_runtime_regressions import RecoveredRuntimeHarness


class _SpatialNLL(torch.nn.Module):
    def __init__(self, reduction, weighted, ignore_index):
        super().__init__()
        self.loss = torch.nn.NLLLoss(
            reduction=reduction,
            weight=torch.randn(4) if weighted else None,
            ignore_index=ignore_index,
        )
        self.conv = torch.nn.Conv2d(16, 4, (3, 3))
        self.m = torch.nn.LogSoftmax(dim=1)

    def forward(self, input, target):
        return self.loss(self.m(self.conv(input)), target)


class _BinaryCrossEntropy(torch.nn.Module):
    def __init__(self, reduction):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target, weight=None, pos_weight=None):
        return torch.nn.functional.binary_cross_entropy_with_logits(
            input,
            target,
            weight=weight,
            pos_weight=pos_weight,
            reduction=self.reduction,
        )


class TestRecoveredLosses(RecoveredRuntimeHarness):
    def test_crossentropyloss(self):
        for ignore_index in [-100, 1]:
            for shape in [(3, 5), (3, 5, 2), (3, 5, 2, 7)]:
                x = torch.randn(shape)
                y = torch.empty((shape[0], *shape[2:]), dtype=torch.long).random_(5)
                y[y == 1] = ignore_index
                for reduction in ["none", "sum", "mean"]:
                    for weighted in [False, True]:
                        with self.subTest(
                            shape=shape,
                            ignore_index=ignore_index,
                            reduction=reduction,
                            weighted=weighted,
                        ):
                            loss = torch.nn.CrossEntropyLoss(
                                reduction=reduction,
                                ignore_index=ignore_index,
                                weight=torch.randn(5) if weighted else None,
                            )
                            self.run_test(loss, (x, y))

    def test_MSELoss(self):
        class MSELoss(torch.nn.Module):
            def forward(self, input, target):
                return tuple(
                    torch.nn.functional.mse_loss(input, target, reduction=reduction)
                    for reduction in ["none", "sum", "mean"]
                )

        self.run_test(MSELoss(), (torch.randn(2, 3, 5), torch.randn(2, 3, 5)))

    def test_kldiv_loss(self):
        class KLDivLoss(torch.nn.Module):
            def __init__(self, reduction, log_target, size_average=None):
                super().__init__()
                self.log_target = log_target
                self.loss = torch.nn.KLDivLoss(
                    reduction=reduction,
                    log_target=log_target,
                    size_average=size_average,
                )

            def forward(self, input, target):
                return self.loss(input, target.log() if self.log_target else target)

        for shape in [(5,), (2, 3, 5), (2, 3, 5, 7)]:
            x, y = torch.rand(shape).log(), torch.rand(shape)
            for reduction, log_target, size_average in [
                ("none", True, None),
                ("mean", False, None),
                ("sum", True, None),
                ("batchmean", False, None),
                ("batchmean", True, False),
            ]:
                self.run_test(KLDivLoss(reduction, log_target, size_average), (x, y))

    def test_nllloss(self):
        class NLLModel(torch.nn.Module):
            def forward(self, input, target):
                return torch.nn.functional.nll_loss(
                    torch.nn.functional.log_softmax(2 * input, dim=1),
                    target,
                    reduction="none",
                )

        x = torch.randn(5, 16)
        target = torch.empty(5, dtype=torch.long).random_(0, 4)
        # Use test data containing default ignore_index=-100.
        target[target == 1] = -100
        self.run_test(NLLModel(), (x, target))

    def _spatial_nll(self, reduction, weighted=False, ignore_index=-100):
        x = torch.randn(5, 16, 10, 10)
        target = torch.empty(5, 8, 8, dtype=torch.long).random_(0, 4)
        if ignore_index == -100:
            # Use test data containing default ignore_index=-100.
            target[target == 1] = -100
        self.run_test(_SpatialNLL(reduction, weighted, ignore_index), (x, target))

    def test_nllloss_2d_none(self):
        self._spatial_nll("none")

    def test_nllloss_2d_mean(self):
        self._spatial_nll("mean")

    def test_nllloss_2d_sum(self):
        self._spatial_nll("sum")

    def test_nllloss_2d_mean_weights(self):
        self._spatial_nll("mean", weighted=True)

    def test_nllloss_2d_mean_ignore_index(self):
        self._spatial_nll("mean", ignore_index=1)

    def test_nllloss_2d_mean_ignore_index_weights(self):
        self._spatial_nll("mean", weighted=True, ignore_index=1)

    def test_nllloss_dynamic_ignore_index(self):
        class LabelSmoothingCrossEntropy(torch.nn.Module):
            def forward(self, preds, target, start_position):
                log_preds = torch.nn.functional.log_softmax(preds, dim=-1)
                ignore_index = start_position.size(1)
                nll = torch.nn.functional.nll_loss(
                    log_preds, target, ignore_index=ignore_index
                )
                return nll + start_position.float()

        # The source derives ignore_index from a statically captured dimension.
        self.run_test(
            LabelSmoothingCrossEntropy(),
            (torch.randn(5, 16), torch.randint(5, (5,)), torch.randint(10, (5, 5))),
        )

    def test_binary_cross_entropy_with_logits(self):
        cases = [
            (torch.randn(5), torch.empty(5).random_(2), None, None),
            (
                torch.randn(3, 4),
                torch.empty(3, 4).random_(2),
                torch.tensor([3]),
                None,
            ),
            (
                torch.randn(3, 2, 4),
                torch.empty(3, 2, 4).random_(2),
                None,
                torch.empty(2, 4).random_(2),
            ),
            (
                torch.randn(3, 3, 4),
                torch.empty(3, 3, 4).random_(2),
                torch.tensor([3]),
                torch.empty(3, 4).random_(2),
            ),
        ]
        for inputs in cases:
            for reduction in ["none", "mean", "sum"]:
                self.run_test(_BinaryCrossEntropy(reduction), inputs)

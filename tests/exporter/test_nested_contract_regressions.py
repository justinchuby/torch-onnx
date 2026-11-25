# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Nested input/output contracts from PyTorch's deleted ONNX runtime suite.

Parent e2c2673d00500d268fcb1cc63fa7cc3173876505; removal
6dca9f63bc4975ac323044a440c2c7e5f8512408. Use regular modules, not the
retired scripted frontend. Legacy failures for None/default binding and nested
kwargs now require successful numerical execution. Optional choices are separately
exported signatures, not claimed as runtime-switchable ONNX Optional inputs.
Python list outputs retain their original fixed arity.
"""

from collections import OrderedDict

import torch
import torchvision
from _common.runtime import RecoveredRuntimeHarness
from torch import Tensor


def _init_test_generalized_rcnn_transform():
    min_size = 100
    max_size = 200
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(
        min_size, max_size, image_mean, image_std
    )
    return transform


class TestRecoveredNestedContracts(RecoveredRuntimeHarness):
    def test_resize_images(self):
        class TransformModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.transform = _init_test_generalized_rcnn_transform()

            def forward(self, images):
                return self.transform.resize(images, None)[0]

        input = torch.rand(3, 10, 20)
        input_test = torch.rand(3, 100, 150)
        self.run_test(
            TransformModule(),
            (input,),
            input_names=["input1"],
            dynamic_axes={"input1": [0, 1, 2]},
            additional_test_inputs=[(input,), (input_test,)],
        )

    def test_transform_images(self):
        class TransformModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.transform = _init_test_generalized_rcnn_transform()

            def forward(self, images: list[Tensor]):
                return self.transform(images)[0].tensors

        input = torch.rand(3, 100, 200), torch.rand(3, 200, 200)
        input_test = torch.rand(3, 100, 200), torch.rand(3, 200, 200)
        self.run_test(
            TransformModule(),
            (input,),
            additional_test_inputs=[(input,), (input_test,)],
        )

    def test_multi_scale_roi_align(self):
        class TransformModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = torchvision.ops.MultiScaleRoIAlign(
                    ["feat1", "feat2"], 3, 2
                )
                self.image_sizes = [(512, 512)]

            def forward(self, input: dict[str, Tensor], boxes: list[Tensor]) -> Tensor:
                return self.model(input, boxes, self.image_sizes)

        i = OrderedDict()
        i["feat1"] = torch.rand(1, 5, 64, 64)
        i["feat2"] = torch.rand(1, 5, 16, 16)
        boxes = torch.rand(6, 4) * 256
        boxes[:, 2:] += boxes[:, :2]

        i1 = OrderedDict()
        i1["feat1"] = torch.rand(1, 5, 64, 64)
        i1["feat2"] = torch.rand(1, 5, 16, 16)
        boxes1 = torch.rand(6, 4) * 256
        boxes1[:, 2:] += boxes1[:, :2]

        self.run_test(
            TransformModule(),
            (
                i,
                [boxes],
            ),
            additional_test_inputs=[
                (
                    i,
                    [boxes],
                ),
                (
                    i1,
                    [boxes1],
                ),
            ],
        )

    def test_tolist(self):
        class List(torch.nn.Module):
            def forward(self, input):
                res: list[int] = input.tolist()
                return res

        self.run_test(
            List(),
            (torch.randint(100, (1,)),),
            additional_test_inputs=[
                torch.tensor([0]),
                torch.tensor([-100]),
                torch.tensor([99]),
            ],
        )

    def test_inplace_list(self):
        class Arithmetic(torch.nn.Module):
            def forward(self, x, y):
                return torch.cat([x.add_(3), y.fill_(0)])

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        self.run_test(
            Arithmetic(),
            (x, y),
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1], "y": [0, 1]},
            additional_test_inputs=[(torch.randn(4, 5), torch.randn(6, 5))],
        )
        self.run_test(Arithmetic(), (x, y), remained_onnx_input_idx=[0])

    def test_mixed_optional_default_tensor_script(self):
        default_y = torch.ones(2, 3)
        default_z = torch.zeros(2, 3)

        class Model(torch.nn.Module):
            def forward(
                self,
                x,
                y: Tensor | None = default_y,
                z: Tensor | None = default_z,
            ):
                if y is not None:
                    return x + y
                if z is not None:
                    return x + z
                return x

        x, y, z = (torch.randn(2, 3) for _ in range(3))
        model = Model()
        self.run_test(model, (x, y, z), input_names=("x", "y", "z"))
        self.run_test(model, (x,), {"y": y, "z": z}, input_names=("x", "y", "z"))
        self.run_test(model, (x,), {"y": y}, input_names=("x", "y"))
        for args, kwargs in (
            ((x, y, None), {}),
            ((x, None, z), {}),
            ((x,), {"y": y, "z": None}),
            ((x,), {"y": None, "z": z}),
            ((x, None, None), {}),
            ((x,), {}),
        ):
            self.run_test(model, args, kwargs)

    def test_all_optional_default_tensor_script(self):
        default_x = torch.ones(2, 3)
        default_y = torch.zeros(2, 3)

        class Model(torch.nn.Module):
            def forward(
                self,
                x: Tensor | None = default_x,
                y: Tensor | None = default_y,
            ):
                if x is not None:
                    return x
                elif y is not None:
                    return y
                else:
                    return torch.tensor(-1.0)

        x, y = torch.randn(2, 3), torch.randn(2, 3)
        model = Model()
        # Optional supports None inputs.
        # Historical legacy behavior, now checked as successful default binding:
        # NOTE: default value is not supported on ONNX, so torch and ONNX has
        # different behavior
        for args, kwargs in (
            ((x,), {}),
            ((), {"y": y}),
            ((x, y), {}),
            ((), {"x": x, "y": y}),
            ((None, y), {}),
            ((None, None), {}),
            ((), {}),
        ):
            self.run_test(model, args, kwargs)

    def test_tuple_of_optional_default_tensor_script(self):
        class Model(torch.nn.Module):
            def forward(
                self,
                x,
                y: tuple[Tensor | None, Tensor | None] = (
                    torch.zeros(2, 3),
                    torch.zeros(2, 3),
                ),
            ):
                y0, y1 = y
                if y0 is not None:
                    return x + y0
                if y1 is not None:
                    return x + y1
                return x

        x, y0, y1 = (torch.randn(2, 3) for _ in range(3))
        model = Model()
        # Historical harness limitation, now checked as a successful call:
        # export succeeds, but running ORT through run_test would fail because the exported model
        # has the inputs flattened into 3 inputs.
        for args, kwargs in (
            ((x, (None, y1)), {}),
            ((x, (y0, y1)), {}),
            ((x,), {"y": (y0, y1)}),
            ((x, (None, None)), {}),
            ((x,), {}),
        ):
            self.run_test(model, args, kwargs)

    def test_symbolic_shape_inference_box_if(self):
        class BoxIfModel(torch.nn.Module):
            def forward(self, boxes, scores):
                # test If
                inds = torch.where(scores > 0.0)[0]
                boxes_1 = boxes[inds]
                return torch.cond(
                    boxes_1.numel() > 3,
                    lambda value: value.clone(),
                    lambda value: value * 2,
                    (boxes_1,),
                )

        boxes = torch.ones(2, 4)
        scores = torch.ones(1, 4)
        model = BoxIfModel()
        for values, predicates in [
            (boxes, scores),
            (boxes, -scores),
            (boxes + 2, torch.tensor([[-1.0, 1.0, -1.0, -1.0]])),
        ]:
            selected = values[torch.where(predicates > 0.0)[0]]
            expected = selected if selected.numel() > 3 else selected * 2
            self.assertEqual(model(values, predicates), expected)
        program = self.run_test(
            BoxIfModel(),
            (boxes, scores),
            additional_test_inputs=[
                (boxes, -scores),
                (boxes, torch.tensor([[1.0, -1.0, -1.0, -1.0]])),
                (boxes + 2, torch.tensor([[-1.0, 1.0, -1.0, -1.0]])),
            ],
        )
        self.assertIn("If", {node.op_type for node in program.model.graph})

# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""Modern capture of the removed TorchVision postprocessing regressions.

Parent e2c2673d00500d268fcb1cc63fa7cc3173876505; removal
6dca9f63bc4975ac323044a440c2c7e5f8512408. Original fixtures compared
eager and JIT tracing. Batched sampling now exercises actual native ONNX export,
dynamic shapes, ROI-dependent resizing and serialization against untouched eager
TorchVision functions. This does not claim automatic capture of their Python
loops. Heatmap capture uses nonsingleton examples; ONNX execution additionally
checks the original single-keypoint case, outside ExportedProgram's guards.
"""

import torch
import torchvision
from test_runtime_regressions import RecoveredRuntimeHarness
from torchvision.models.detection.roi_heads import (
    heatmaps_to_keypoints,
    paste_masks_in_image,
)


class _ClipBoxes(torch.nn.Module):
    def forward(self, boxes, size):
        height = torch.scalar_tensor(
            size.shape[0], dtype=boxes.dtype, device=boxes.device
        )
        width = torch.scalar_tensor(
            size.shape[1], dtype=boxes.dtype, device=boxes.device
        )
        x = boxes[..., 0::2].clamp_min(0).clamp_max(width)
        y = boxes[..., 1::2].clamp_min(0).clamp_max(height)
        return torch.stack((x, y), dim=-1).flatten(start_dim=-2)


class _BatchedMaskPaste(torch.nn.Module):
    def forward(self, masks, boxes, canvas):
        padded, scale = torchvision.models.detection.roi_heads.expand_masks(
            masks, padding=1
        )
        boxes = torchvision.models.detection.roi_heads.expand_boxes(boxes, scale).to(
            torch.int64
        )
        height, width = canvas.shape
        rows = torch.arange(height, dtype=masks.dtype, device=masks.device)[
            None, :, None
        ]
        columns = torch.arange(width, dtype=masks.dtype, device=masks.device)[
            None, None, :
        ]
        left, top, right, bottom = (boxes[:, i, None, None] for i in range(4))
        box_width = (right - left + 1).clamp(min=1)
        box_height = (bottom - top + 1).clamp(min=1)
        x = (columns - left + 0.5) / box_width * 2 - 1
        y = (rows - top + 0.5) / box_height * 2 - 1
        grid = torch.stack((x.expand(-1, height, -1), y.expand(-1, -1, width)), dim=-1)
        sampled = torch.nn.functional.grid_sample(
            padded, grid, mode="bilinear", padding_mode="border", align_corners=False
        )
        visible = (
            (columns >= left) & (columns <= right) & (rows >= top) & (rows <= bottom)
        )
        return torch.where(visible[:, None], sampled, torch.zeros_like(sampled))


class _BatchedHeatmapKeypoints(torch.nn.Module):
    def forward(self, maps, rois):
        sizes = (rois[:, 2:] - rois[:, :2]).clamp(min=1)
        integer_sizes = sizes.ceil().to(torch.int64)
        width = integer_sizes[:, 0].amax().item()
        height = integer_sizes[:, 1].amax().item()
        torch._check(width > 0)
        torch._check(height > 0)
        # Masked margins keep the data-derived sampling canvas nonsingleton.
        width = width + 2
        height = height + 2
        columns = torch.arange(width, dtype=maps.dtype, device=maps.device)[
            None, None, :
        ]
        rows = torch.arange(height, dtype=maps.dtype, device=maps.device)[None, :, None]
        widths = integer_sizes[:, 0, None, None]
        heights = integer_sizes[:, 1, None, None]
        grid = torch.stack(
            (
                ((columns + 0.5) / widths * 2 - 1).expand(-1, height, -1),
                ((rows + 0.5) / heights * 2 - 1).expand(-1, -1, width),
            ),
            dim=-1,
        )
        sampled = torch.nn.functional.grid_sample(
            maps, grid, mode="bicubic", padding_mode="border", align_corners=False
        )
        valid = (columns < widths) & (rows < heights)
        flat = torch.where(valid[:, None], sampled, -torch.inf).flatten(2)
        indices = flat.argmax(dim=2)
        x = (indices % width).float()
        y = torch.div(indices, width, rounding_mode="floor").float()
        correction = sizes / integer_sizes
        x = (x + 0.5) * correction[:, 0, None] + rois[:, 0, None]
        y = (y + 0.5) * correction[:, 1, None] + rois[:, 1, None]
        predictions = torch.stack((x, y, torch.ones_like(x)), dim=-1)
        scores = torch.gather(flat, 2, indices[:, :, None]).squeeze(2)
        return predictions, scores


class TestRecoveredDetectionPostprocessing(RecoveredRuntimeHarness):
    def _check_postprocessing(
        self,
        model,
        reference,
        arguments,
        dimensions,
        *,
        capture_index=0,
        required_operator="GridSample",
    ):
        for args in arguments:
            torch.testing.assert_close(
                model(*args), reference(*args), rtol=1e-3, atol=1e-7
            )
        program = self.run_test(
            model, arguments[capture_index], dynamic_shapes=dimensions
        )
        self.assertIn(required_operator, {node.op_type for node in program.model.graph})
        self.assert_program_roundtrip(
            program,
            examples=arguments,
            reference=reference,
            rtol=1e-3,
            atol=1e-7,
        )

    def test_clip_boxes_to_image(self):
        class Module(torch.nn.Module):
            def forward(self, boxes, size):
                shape = (size.shape[0], size.shape[1])
                return torchvision.ops.boxes.clip_boxes_to_image(boxes, shape)

        boxes = torch.randn(5, 4) * 500
        boxes[:, 2:] += boxes[:, :2]
        size = torch.randn(200, 300)
        size_2 = torch.randn(300, 400)
        arguments = [
            (boxes, size),
            (boxes, size_2),
            (
                torch.tensor([[-2.0, -3.0, 20.0, 30.0], [1.0, 2.0, 5.0, 6.0]]),
                torch.empty(7, 11),
            ),
            (torch.empty(0, 4), torch.empty(7, 11)),
            (boxes, torch.empty(0, 11)),
        ]
        self._check_postprocessing(
            _ClipBoxes(),
            Module(),
            arguments,
            (
                {0: torch.export.Dim("box_count")},
                {
                    0: torch.export.Dim("clip_height"),
                    1: torch.export.Dim("clip_width"),
                },
            ),
            required_operator="Clip",
        )

    def test_paste_mask_in_image(self):
        arguments = []
        for count, size, scale in ((10, 100, 50), (20, 200, 100)):
            masks = torch.rand(count, 1, 26, 26)
            boxes = torch.rand(count, 4)
            boxes[:, 2:] += torch.rand(count, 2)
            boxes *= scale
            arguments.append((masks, boxes, torch.empty(size, size)))
        arguments.append(
            (
                torch.rand(3, 1, 26, 26),
                torch.tensor(
                    [
                        [-2.0, -3.0, 5.0, 8.0],
                        [3.0, 4.0, 12.0, 10.0],
                        [20.0, 10.0, 27.0, 19.0],
                    ]
                ),
                torch.empty(13, 23),
            )
        )
        count = torch.export.Dim("mask_count", min=2)
        self._check_postprocessing(
            _BatchedMaskPaste(),
            lambda masks, boxes, canvas: paste_masks_in_image(
                masks, boxes, canvas.shape
            ),
            arguments,
            (
                {0: count},
                {0: count},
                {
                    0: torch.export.Dim("canvas_height", min=2),
                    1: torch.export.Dim("canvas_width", min=2),
                },
            ),
        )

    def test_heatmaps_to_keypoints(self):
        arguments = [
            (torch.rand(10, 1, 26, 26), torch.rand(10, 4)),
            (torch.rand(20, 2, 21, 21), torch.rand(20, 4)),
            (
                torch.rand(3, 2, 15, 17),
                torch.tensor(
                    [[2.0, 3.0, 6.2, 8.7], [0.0, 0.0, 9.0, 7.0], [1.0, 4.0, 8.5, 6.5]]
                ),
            ),
        ]
        count = torch.export.Dim("roi_count", min=2)
        self._check_postprocessing(
            _BatchedHeatmapKeypoints(),
            heatmaps_to_keypoints,
            arguments,
            (
                {
                    0: count,
                    1: torch.export.Dim("keypoints", min=2),
                    2: torch.export.Dim("map_height", min=2),
                    3: torch.export.Dim("map_width", min=2),
                },
                {0: count},
            ),
            capture_index=1,
        )

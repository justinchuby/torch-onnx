# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""List construction/mutation regressions from PyTorch's deleted runtime suite.

Parent e2c2673d00500d268fcb1cc63fa7cc3173876505; removal
6dca9f63bc4975ac323044a440c2c7e5f8512408. Preserve source computations
and output containers with regular modules. These fixed-example loops are
captured with fixed container arities, not claimed as variable-length sequences.
"""

import torch
from test_runtime_regressions import RecoveredRuntimeHarness


class TestRecoveredListMutations(RecoveredRuntimeHarness):
    def test_loop_multi_dim(self):
        class LoopMultiDimModel(torch.nn.Module):
            def forward(self, x, y):
                values = torch.flip(x.narrow(0, 0, 7), [0])

                def cond(index, selected):
                    return index < 7

                def body(index, selected):
                    # Tensor indexing avoids specializing a data-derived loop index.
                    row = values.index_select(0, index.reshape(1))[0, 0]
                    return index + 1, row[selected]

                return torch.while_loop(
                    cond, body, (torch.zeros((), dtype=torch.int64), y)
                )[1]

        model = LoopMultiDimModel()
        original = (torch.randint(0, 5, (8, 1, 17)), torch.ones(1, dtype=torch.long))
        alternate = (torch.randint(0, 5, (10, 1, 17)), torch.tensor([4]))
        for values, initial in [original, alternate]:
            expected = initial
            for row in torch.flip(values.narrow(0, 0, 7), [0]):
                expected = row[0][expected]
            self.assertEqual(model(values, initial), expected)
        program = self.run_test(
            model,
            original,
            input_names=["x", "y"],
            dynamic_axes={"x": [0]},
            additional_test_inputs=[alternate],
        )
        self.assertIn("Loop", {node.op_type for node in program.model.graph})

    def test_loop_with_list(self):
        class ListLoopModel(torch.nn.Module):
            def forward(self, x):
                res = []
                res1 = []
                arr = x.split([3, 4, 1, 1, 2, 3, 2], 0)
                res2 = torch.zeros(3, 4, dtype=torch.long)
                res3 = []
                res4 = []
                for i in range(len(arr)):
                    res.append(arr[i].sum(0, False))
                    res1.append(arr[-1 - i].sum(0, False))
                    res2 += 1
                    res3 = res3 + [arr[i].sum(0, False)]  # noqa: RUF005 - test list concatenation separately from +=.
                    res4 += [arr[-1 - i].sum(0, False)]
                return res, res1, res2, torch.stack(res3), torch.stack(res4)

        model = ListLoopModel()
        inputs = torch.randn(16)
        self.run_test(model, inputs)

    def test_list_append(self):
        class ListModel(torch.nn.Module):
            def forward(self, x, y):
                res = []
                for i in range(x.size(0)):
                    res += [torch.matmul(x[i], y)]
                return res

        model = ListModel()
        x = torch.randn(16, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    def test_list_append_nested(self):
        class ListModel(torch.nn.Module):
            def forward(self, x, y):
                res = []
                for i in range(x.size(0)):
                    for j in range(x.size(1)):
                        res += [torch.matmul(x[i][j], y)]
                return res

        model = ListModel()
        x = torch.randn(4, 4, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    def test_list_append_nested_mixed_dtype(self):
        class ListModel(torch.nn.Module):
            def forward(self, x, y):
                res = []
                for i in range(x.size(0)):
                    for j in range(x.size(1)):
                        if i == j:
                            res.append(x == y)
                        else:
                            res.append(x != y)
                return res

        model = ListModel()
        x = torch.randn(4, 4, 3, 4)
        y = torch.randn(3, 4)
        self.run_test(model, (x, y))

    def test_list_pop(self):
        class ListModel(torch.nn.Module):
            def forward(self, x, y):
                res = []
                for i in range(x.size(0)):
                    res += [torch.matmul(x[i], y)]
                res.pop()
                return res

        model = ListModel()
        x = torch.randn(16, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    def test_list_pop_nested(self):
        class ListModel(torch.nn.Module):
            def forward(self, x, y):
                res = []
                for i in range(x.size(0)):
                    for j in range(x.size(1)):
                        res += [torch.matmul(x[i][j], y)]
                        res.pop()
                    res += [torch.matmul(x[i][0], y)]
                return res

        model = ListModel()
        x = torch.randn(4, 4, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    def test_list_del(self):
        class ListModel(torch.nn.Module):
            def forward(self, x, y):
                res = []
                for i in range(x.size(0)):
                    res += [torch.matmul(x[i], y)]
                del res[2]
                return res

        model = ListModel()
        x = torch.randn(16, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    def test_list_del_nested(self):
        class ListModel(torch.nn.Module):
            def forward(self, x, y):
                res = []
                for i in range(x.size(0)):
                    for j in range(x.size(1)):
                        res += [torch.matmul(x[i][j], y)]
                        del res[i]
                    res += [torch.matmul(x[i][0], y)]
                return res

        model = ListModel()
        x = torch.randn(4, 4, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

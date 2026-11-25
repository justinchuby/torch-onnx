# Copyright: PyTorch contributors. See LICENSE-PYTORCH.
"""List construction/mutation regressions from PyTorch's deleted runtime suite.

Parent e2c2673d00500d268fcb1cc63fa7cc3173876505; removal
6dca9f63bc4975ac323044a440c2c7e5f8512408. Preserve source computations
and output containers with regular modules. These fixed-example loops are
captured with fixed container arities, not claimed as variable-length sequences.
"""

import torch
from _common.runtime import RecoveredRuntimeHarness


class TestRecoveredListMutations(RecoveredRuntimeHarness):
    def test_sequance_loopcarried(self):
        class SequanceLoopModel(torch.nn.Module):
            def forward(self, x):
                outputs = []
                for _ in range(3):
                    outputs += [x]
                return torch.stack(outputs).transpose(0, 1)

        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(
            SequanceLoopModel(),
            x,
            additional_test_inputs=[torch.arange(12, dtype=torch.int).reshape(3, 4)],
        )

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

    def test_list_append_in_block(self):
        class ListModel(torch.nn.Module):
            def forward(self, x, y):
                res = []
                for i in range(x.size(0)):
                    res.append(torch.matmul(x[i], y))
                return res

        model = ListModel()
        x = torch.randn(16, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(
            model,
            (x, y),
            additional_test_inputs=[(torch.randn_like(x), torch.randn_like(y))],
        )

    def test_list_append_in_nested_block(self):
        class ListModel(torch.nn.Module):
            def forward(self, x, y):
                res = []
                for i in range(x.size(0)):
                    for j in range(x.size(1)):
                        res.append(torch.matmul(x[i][j], y))
                return res

        model = ListModel()
        x = torch.randn(4, 4, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(
            model,
            (x, y),
            additional_test_inputs=[(torch.randn_like(x), torch.randn_like(y))],
        )

    def test_list_pop_in_block(self):
        class ListModel(torch.nn.Module):
            def forward(self, x, y):
                res = []
                elem = torch.matmul(x[0], y)
                for i in range(x.size(0)):
                    res.append(torch.matmul(x[i], y))
                for _ in range(x.size(0)):
                    elem = res.pop()
                for i in range(x.size(0)):
                    res.append(torch.matmul(x[i], y))
                    elem = res.pop()
                # Scripted append returned the list; Python append returns None.
                res.append(elem)
                return res

        model = ListModel()
        x = torch.randn(16, 3, 4)
        y = torch.randn(4, 5)
        program = self.run_test(
            model,
            (x, y),
            additional_test_inputs=[(torch.randn_like(x), torch.randn_like(y))],
        )
        self.assert_program_outputs(program, (x, y), expected=x[-1] @ y)

    def test_list_del_in_block(self):
        class ListModel(torch.nn.Module):
            def forward(self, x, y):
                res = []
                elem = torch.matmul(x[0], y)
                for i in range(x.size(0)):
                    res.append(torch.matmul(x[i], y))
                for _ in range(x.size(0)):
                    del res[0]
                for i in range(x.size(0)):
                    res.append(torch.matmul(x[i], y))
                    del res[0]
                # Scripted append returned the list; Python append returns None.
                res.append(elem)
                return res

        model = ListModel()
        x = torch.randn(16, 3, 4)
        y = torch.randn(4, 5)
        program = self.run_test(
            model,
            (x, y),
            additional_test_inputs=[(torch.randn_like(x), torch.randn_like(y))],
        )
        self.assert_program_outputs(program, (x, y), expected=x[0] @ y)

    def test_list_unpack(self):
        class ListModel(torch.nn.Module):
            def forward(self, x, y):
                res = []
                _elem = torch.matmul(x[0], y)
                for i in range(x.size(0)):
                    res.append(torch.matmul(x[i], y))
                a, b, _c = res
                return a, b

        model = ListModel()
        x = torch.randn(3, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(
            model,
            (x, y),
            additional_test_inputs=[(torch.randn_like(x), torch.randn_like(y))],
        )

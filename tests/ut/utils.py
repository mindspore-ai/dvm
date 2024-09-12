# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from functools import reduce
import re


class Operation:
    def __init__(self, name, graph, inputs=[]):
        self.name = name
        self.graph = graph
        self.inputs = inputs

    def input(self, i):
        return self.graph[self.inputs[i]]

    def __str__(self) -> str:
        result = self.name + '('
        result = reduce(lambda x, y: x+str(y)+', ', self.inputs, result)
        result += ')'
        return result


class Graph:
    # pat1: find operation name and inputs string
    pat1 = re.compile(r'= (\b\w+\b)\((.*?)\)')
    # pat2: find id of all inputs
    pat2 = re.compile(r'%(\d+)\[')

    def __init__(self, tester=None):
        self.ops = []
        if tester is not None:
            self.construct(tester)

    def construct(self, tester):
        lines = tester.dump().split("\n")
        for line in lines:
            match = self.pat1.search(line)
            if not match:
                continue
            op_name = match.group(1)
            inputs_str = match.group(2)
            input_ids = list(map(int, self.pat2.findall(inputs_str)))
            op = Operation(op_name, self, input_ids)
            self.ops.append(op)

    def __getitem__(self, key):
        return self.ops[key]

    def __len__(self):
        return len(self.ops)

    def __str__(self) -> str:
        result = ""
        for i, op in enumerate(self.ops):
            result += "%" + str(i) + " = "
            result += str(op) + "\n"
        return result

# Third-Party Notices

This repository contains code copied or adapted from the projects below.
Modifications made for MoE Load Balancer are distributed under the repository's
Apache-2.0 license; the original portions remain subject to their original
licenses.

## DeepSeek EPLB

- Source: <https://github.com/deepseek-ai/EPLB>
- License: MIT
- Local code: `src/moe_load_balancer/policies/l1/eplb/algorithms/deepseek.py`
- Copyright: Copyright (c) 2025 DeepSeek

## DeepSeek LPLB

- Source: <https://github.com/deepseek-ai/LPLB>
- License: MIT
- Local code: the Interior Point Method implementation in
  `src/moe_load_balancer/kernels/lplb/csrc/ipm.cuh`
- Copyright: Copyright (c) 2025 DeepSeek

The MIT-licensed components above are distributed under the following terms:

> MIT License
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

## SGLang

- Source: <https://github.com/sgl-project/sglang>
- License: Apache License 2.0
- Copyright: Copyright 2023-2026 SGLang Team
- Local code includes the LPLB policy and CUDA/JIT support under
  `src/moe_load_balancer/policies/l2/lplb.py`,
  `src/moe_load_balancer/kernels/utils.py`, and
  `src/moe_load_balancer/kernels/lplb/`.

The Apache License 2.0 text is provided in the repository's `LICENSE` file.

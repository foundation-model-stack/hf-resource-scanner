# HFResourceScanner

Scan resources consumed during training using HFTrainer and break up consumption by category with **ZERO** overheads.

Works for all training approaches (such as full-fine tuning, Prompt Tuning, LORA) for HFTrainer and other trainers based on it, such as SFTTrainer. Limited support for FSDP (currently).

Measures and reports GPU memory consumption broken up into 4 categories:
1. Model paramters
2. Optimizer
3. Gradients
4. Activations

## Install

```
pip install .
```

## Usage

2 line change to your existing code:

1. Import the Scanner.
```
from HFResourceScanner import Scanner
```

2. Create and add a Scanner object to the list of callbacks:
```
...
callbacks.append(Scanner())
...
```

In the default configuration, prints out data to stdout.

## Configuring

You can further configure the Scanner to:
1. Choose the step to instrument and scan at (we only scan at a single step). There is no reason to change from the default of 5.
2. Output to stdout (the default), file or use a callback function to deal with the output. See examples provided in the `examples/` folder.

## Methodology

Uses a combination of the following items:

1. HFTrainer Callbacks to measure memory and breakup at step boundary.
2. Pytorch hook functions such as `nn.Module` Forward and `optimizer.step` function to measure memory at ideal locations.

![Memory breakup](./imgs/memory.png)

It is important to note that this scanning happens for a single step:
1. At step start, setup hook functions.
2. During the step, run the functions to take single point measurements.
3. At the end of the step, correlate the data and cleaup the hook functions.

## Alternatives

1. Pytorch Profile: *very* heavy weight, can slow down severely for larger models. In comparison, our approach is the most minimalistic approach possible and scales to any model type and size.


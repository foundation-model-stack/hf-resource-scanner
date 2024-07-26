# HFResourceScanner

Scan resources consumed during training using HFTrainer and break up consumption by category with **almost ZERO** overheads.

Works for all training approaches (such as full-fine tuning, Prompt Tuning, LORA) for HFTrainer and other trainers based on it, such as SFTTrainer. Limited support for FSDP (currently).

Measures and reports GPU memory consumption broken up into 4 categories:
1. Model paramters
2. Optimizer
3. Gradients
4. Activations

In addition, it also auto-detects various configurational settings operational during the training phase. 

## Install

```
pip install .
```

## Usage

3 line change to the existing code:

1. Import the scanner.
```
from HFResourceScanner import scanner
```
(import scanner module instead of the class Scanner)

2. Create and add a Scanner object to the list of callbacks:
```
...
callbacks.append(scanner.Scanner())
...
```

3. Call the function from scanner to attach hooks to the objects in the script. This line is supposed to be added just before the training begins. (after trainer object is defined)
```
...
scanner.modelhook(vars().items())
...
```
(vars.items() passes all the objects as an argument from the trainer script to the scanner)

In the default configuration, prints out data to stdout.

## Configuring

You can further configure the Scanner to:
1. Choose the step to instrument and scan at (we only scan at a single step). There is no reason to change from the default of 5.
2. Output to stdout (the default), file or use a callback function to deal with the output. See examples provided in the `examples/` folder.

## Methodology for memory profiler

Uses a combination of the following items:

1. HFTrainer Callbacks to measure memory and breakup at step boundary.
2. Pytorch hook functions such as `nn.Module` Forward and `optimizer.step` function to measure memory at ideal locations.

![Memory breakup](./imgs/memory.png)

It is important to note that this scanning happens for a single step:
1. At step start, setup hook functions.
2. During the step, run the functions to take single point measurements.
3. At the end of the step, correlate the data and cleaup the hook functions.

## Methodology for configuration detector

We leverage the strategic advantage offered by PyTorch Hooks.
By identifying important modules, objects and tensors to attach these hooks, we gain access to the hidden activations which can provide valuable insights about the training framework.
We aim to extract the following configurations:
1. The model ( dtype, attention implementation, number of parameters, etc. )
2. The training loop (Sequence length, Batch Size, Gradient accumulation steps, optimizer related configs, Mixed precision training, etc)
3. The distributed frameworks used. (FSDP/DeepSpeed, Num of processes, Configs specific to the particular framework)

## Alternatives

1. Pytorch Profile: can give a complete breakup of stack traces and [memory consumption](https://pytorch.org/blog/understanding-gpu-memory-1/). While this is much more exhaustive and useful for optimizing implementations, this may be overwhelming for casual users. Also, this approach can take non-trivial amount of time to compute memory allocations and is quite slow for larger models.

## License

Apache 2.0

import torch
from transformers.trainer_callback import TrainerCallback
import accelerate
import peft

import logging
logger = logging.getLogger(__name__)

import sys
import os

TARGET_STEP = 5

class Scanner(TrainerCallback):
    """Scan category-wise resource consumption during training.

    Attributes:
    -----------
    target_step: int
        Scanning is done during a single step of training.
    """
    def __init__(self, target_step: int = TARGET_STEP, output_fmt=None):
        """Construct a Scanner.

        Params:
        -------
        target_step:
          the step number during which to scan. Defaults to 5, if not specified.
          *Important: if this number is larger than the total number of steps,
          the scanner will never fire.*

        output_fmt:
          The preferred output approach. Should be one of:
          - Unspecified (the default): will write to stdout.
          - String: will write in plain text format to a file of given name
          - <filename>.json: will write to file in JSON format
        """
        self.data = {}
        self.metadata = {}

        if not isinstance(target_step, int):
            logger.warning("Non integer target_step requested: switching to default value instead!")
            target_step = TARGET_STEP

        if target_step < 0:
            logger.warning("Negative target_step: switching to default instead!")
            target_step = TARGET_STEP
        elif target_step <= 3:
            logger.warning("Initial steps are prone to be unstable. A target_step higher than 3 is recommended.")

        self.target_step = target_step
        self.output_fmt = output_fmt

    def on_step_begin(self, args, state, control, model, tokenizer, optimizer, **kwargs):
        # only calculate for master process in fsdp, other GPUs will be symmetrical
        if state and not state.is_world_process_zero:
            return

        # note that global_step is number of steps completed, so we need the -1
        if state.global_step != self.target_step - 1:
            return

        # run only for the target step

        ## setup optimizer hook to calc grad

        # in case we use accelerate, the real optimizer is one step removed
        if isinstance(optimizer, accelerate.optimizer.AcceleratedOptimizer):
            optimizer = optimizer.optimizer

        def optim_track_gradmem(optimizer, *args, **kwargs):
            gradmem = 0
            for lay in optimizer.state.items():
                ps = lay[0]
                if ps.grad != None:
                    gradmem += ps.grad.nelement() * ps.grad.element_size()
            self.data["gradients"] = gradmem

        self.optim_hook_handle = optimizer.register_step_pre_hook(optim_track_gradmem)

        ## setup model fwd hook to calc activations
        def model_track_activations(model, args, output):
            self.data["activation"] = torch.cuda.memory_allocated()

        self.model_fwd_hook_handle = model.register_forward_hook(model_track_activations)

    def on_step_end(self, args, state, control, model, tokenizer, optimizer, **kwargs):
        # only calculate for master process in fsdp, other GPUs will be symmetrical
        if state and not state.is_world_process_zero:
            return

        if state.global_step != self.target_step:
            return

        # run only for the target step

        self.data["cudamem"] = torch.cuda.memory_allocated()
        self.data["cuda_max_mem"] = torch.cuda.max_memory_allocated()
        self.data["model"] = model.get_memory_footprint()

        if isinstance(model, peft.PeftModel):
            self.metadata["peft_trainable_params"] = model.get_nb_trainable_parameters()

        optimizer_mem = 0
        for lay in optimizer.state.items():
            lstate = lay[1]
            for v in lstate.values():
                if isinstance(v, torch.Tensor):
                    optimizer_mem += v.nelement() * v.element_size()
        self.data["optimizer"] = optimizer_mem

        # we cannot calculate gradients from here
        # it will happen in the optimizer step hook
        # now, deregister that hook
        self.optim_hook_handle.remove()
        # similary for activations
        self.model_fwd_hook_handle.remove()
        # update activation value to remove out the model params + optimizer
        self.data["activation"] -= self.data["model"] + self.data["optimizer"]

        # optional clean up step in presenting data
        from .utils import fmt_size
        for k, v in self.data.items():
            self.data[k] = fmt_size(v)

        self.handle_output()

    def write_plain(self, fout=sys.stdout):
        print("ResourceScanner: ", self.data, file=fout)
        if self.metadata:
            print("ResourceScanner metadata: ", self.metadata, file=fout)

    def write_json(self, fout=sys.stdout):
        import json
        out = json.dumps({"data": self.data, "metadata": self.metadata})
        print(out, file=fout)

    def handle_output(self):

        # simplest case, no output_fmt specified
        if self.output_fmt == None:
            self.write_plain()
            return

        # writing to a file
        if isinstance(self.output_fmt, str):
            outfile = self.output_fmt

            try:
                fout = open(outfile, "w")
                _, ext = os.path.splitext(outfile)

                if ext == ".json":
                    self.write_json(fout)
                else:
                    self.write_plain(fout)

                fout.close()
            except:
                logger.error("Problem in writing to file")
                logger.info("Switching to stdout instead.")

                self.write_plain()
            finally:
                return

        logger.warning(f"Unrecognized output format requested: {self.output_fmt}")
        logger.info("Switching to default.")
        self.write_plain()

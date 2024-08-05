import torch
from transformers.trainer_callback import TrainerCallback
import accelerate
import peft

import logging
logger = logging.getLogger(__name__)

import sys
import os
from inspect import getfullargspec

import time

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
          - Function which takes in 2 arguments:
              will call back the provided function with the data and metadata dict
        """
        self.mem_data = {}
        self.time_data = {}
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
        
        # the following lines of code run
        # only for the target step on rank 0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.time_data["step_begin"] = time.time_ns()

        ## setup optimizer hook to calc grad
        # in case we use accelerate, the real optimizer is one step removed
        if isinstance(optimizer, accelerate.optimizer.AcceleratedOptimizer):
            optimizer = optimizer.optimizer
        
        # functions to be called when hooks fired
        def fwd_begin(module, *args, **kwargs):
            self.time_data["fwd_begin"] = time.time_ns()
        
        def bwd_begin(module, *args, **kwargs):   
            self.time_data["bwd_begin"] = time.time_ns()

        def opt_step_begin(module, *args, **kwargs):
            self.time_data["opt_begin"] = time.time_ns()
            gradmem = 0
            for lay in optimizer.state.items():
                ps = lay[0]
                if ps.grad != None:
                    gradmem += ps.grad.nelement() * ps.grad.element_size()
            self.mem_data["gradients"] = gradmem
        
        def fwd_end(module, *args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.time_data["fwd_end"] = time.time_ns()
            self.mem_data["activation"] = torch.cuda.memory_allocated()
            
        def bwd_end(module, *args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize() 
            self.time_data["bwd_end"] = time.time_ns()    
         
        def opt_step_end(module, *args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.time_data["opt_end"] = time.time_ns()

        
        for name, param in model.named_parameters():
            first_learnable_param = param
            break        
        
        # registering all hooks
        self.fwd_begin_hook_handle = model.register_forward_pre_hook(fwd_begin)
        self.fwd_end_hook_handle = model.register_forward_hook(fwd_end)
        
        self.bwd_begin_hook_handle = model.lm_head.register_full_backward_pre_hook(bwd_begin)
        self.bwd_end_hook_handle = first_learnable_param.register_hook(bwd_end)
    
        self.opt_step_begin_hook_handle = optimizer.register_step_pre_hook(opt_step_begin)
        self.opt_step_end_hook_handle = optimizer.register_step_post_hook(opt_step_end)

        ##############
        # NOTE: This is not ideal, but is it factually correct as a hack?
        # Although this is working, this is not a good way to use hooks
        # ideal case should have been:
        # model.register_full_backward_pre_hook(bwd_begin)
        # model.register_full_backward_hook(bwd_end)
        # BUT, the above code is not working, because the hooks never fire
        # because the model, although inheriting from nn.Module, the register_hook functions
        # give a warning, that it needs a tensor, not a huggingface model.
        # The next best thing couldve been to use the register a hook to every trainanble paramater,
        # but then there are too many hooks fired for one backpropagation step.
        # The current solution is to use the `lm_head` as the last learnable parameter, and register a hook to it.
        # and named_paramaters() to get the first learnable parameter, and then "register_hook()"
        # register_hook() is not ideal, because it is not a full_backward_hook to an nn.Module, but a hook to a single parameter.
        # but apparently, this is the only way to get a hook to a torch.nn.Parameter
        # and this hook fires on gradient calculation, which is what we need.
        # documentation on this weirdness is here: https://pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html
        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook
        ########

        ### NOTE: The following are the different ways I tried to register hooks, and the results:
        # self.bwd_end_hook_handle = model.register_backward_hook(bwd_end) # FIRES before the full_backward_pre_hook # register_backward_hook is  deprecated in favor of register_full_backward_hook
        # self.bwd_end_hook_handle = model.lm_head.register_backward_hook(bwd_end) # time difference bw this and the backprop start is 4e-5 sec
        # self.bwd_end_hook_handle = first_learnable_param.register_backward_hook(bwd_end) # Throws error because first_learnable_param is not a torch.nn.Parameter, but this needs a tensor [???]
        
        # self.bwd_end_hook_handle = model.register_hook(bwd_end) # Error, because model has no attribute `register_hook`
        # self.bwd_end_hook_handle = model.lm_head.register_hook(bwd_end) # Error: Linear has no attribute `register_hook`
        
        # self.bwd_end_hook_handle = model.register_full_backward_hook(bwd_end) # DOESNT FIRE -> ideal, but doesnt fire
        # self.bwd_end_hook_handle = model.lm_head.register_full_backward_hook(bwd_end) # time difference is 0.00019 sec
        # self.bwd_end_hook_handle = first_learnable_param.register_full_backward_hook(bwd_end) # Throws error because first_learnable_param is a torch.nn.Parameter, but this needs a tensor [???]
        ##############

        
    def on_step_end(self, args, state, control, model, tokenizer, optimizer, **kwargs):
        # only calculate for master process in fsdp, other GPUs will be symmetrical
        if state and not state.is_world_process_zero:
            return

        if state.global_step != self.target_step:
            return

        # the following lines of code run
        # only for the target step on rank 0
        torch.cuda.synchronize()
        self.time_data["step_end"] = time.time_ns()
        self.mem_data["cudamem"] = torch.cuda.memory_allocated()
        self.mem_data["cuda_max_mem"] = torch.cuda.max_memory_allocated()
        self.mem_data["model"] = model.get_memory_footprint()

        if isinstance(model, peft.PeftModel):
            self.metadata["peft_trainable_params"] = model.get_nb_trainable_parameters()

        optimizer_mem = 0
        for lay in optimizer.state.items():
            lstate = lay[1]
            for v in lstate.values():
                if isinstance(v, torch.Tensor):
                    optimizer_mem += v.nelement() * v.element_size()
        self.mem_data["optimizer"] = optimizer_mem

        # we cannot calculate gradients from here
        # it will happen in the optimizer step hook
        # update activation value to remove out the model params + optimizer
        self.mem_data["activation"] -= self.mem_data["model"] + self.mem_data["optimizer"]

        # optional clean up step in presenting data
        from .utils import fmt_size
        for k, v in self.mem_data.items():
            self.mem_data[k] = fmt_size(v)
        
        # deregister all hooks
        self.remove_all_hook_handles()
        self.handle_output()

    def remove_all_hook_handles(self):
        self.fwd_begin_hook_handle.remove()
        self.fwd_end_hook_handle.remove()
        self.bwd_begin_hook_handle.remove()
        self.bwd_end_hook_handle.remove()
        self.opt_step_begin_hook_handle.remove()
        self.opt_step_end_hook_handle.remove()

    def write_plain(self, fout=sys.stdout):
        print("ResourceScanner Data: ", self.mem_data, file=fout)
        print("Time Data: ", self.time_data, file=fout)
        if self.metadata:
            print("Scanner metadata: ", self.metadata, file=fout)

    def write_json(self, fout=sys.stdout):
        import json
        out = json.dumps({"time_data": self.time_data,
                          "mem_data": self.mem_data, 
                          "metadata": self.metadata})
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

        # callback function
        if callable(self.output_fmt):
            try:
                if len(getfullargspec(self.output_fmt).args) == 2:
                    self.output_fmt(self.time_data, self.mem_data, self.metadata)
                    return
            except:
                logger.error("Problem in inspecting callback function.")
                logger.info("Switching to stdout instead.")
                self.write_plain()

        logger.warning(f"Unrecognized output format requested: {self.output_fmt}")
        logger.info("Switching to default.")
        self.write_plain()

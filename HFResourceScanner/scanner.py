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
import torch.nn as nn
from transformers import PreTrainedModel 
import transformers
import deepspeed

TARGET_STEP = 4

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

        ## CONFIGS ##
        self.step=0 
        self.num_fwd_pass=0
        self.grad_accum_steps=0
        self.model_handle = self.layer_handle = self.bwd_step_handle = self.grad_accum_handle = self.amp_handle = self.print_handle = None
        self.configs_dict={}
        self.trainer=None
        self.is_hf_model=False
        self.grad_accum_hook_attached=False
        self.dtypes_for_amp=[]
        self.grad_checkpointing_checked=False
        self.opt_hook_fired = False
        

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
        self.time_data["step_begin_absolute"] = time.time_ns()

        ## setup optimizer hook to calc grad
        # in case we use accelerate, the real optimizer is one step removed
        if isinstance(optimizer, accelerate.optimizer.AcceleratedOptimizer):
            optimizer = optimizer.optimizer
        
        # functions to be called when hooks fired
        def fwd_begin(module, *args, **kwargs):
            self.time_data["fwd_begin_absolute"] = time.time_ns()
            self.time_data["fwd_begin_relative"] = self.time_data["fwd_begin_absolute"] - self.time_data["step_begin_absolute"]
        
        def bwd_begin(module, *args, **kwargs):   
            self.time_data["bwd_begin_absolute"] = time.time_ns()
            self.time_data["bwd_begin_relative"] = self.time_data["bwd_begin_absolute"] - self.time_data["step_begin_absolute"]

        def opt_step_begin(module, *args, **kwargs):
            self.time_data["opt_begin_absolute"] = time.time_ns()
            self.time_data["opt_begin_relative"] = self.time_data["opt_begin_absolute"] - self.time_data["step_begin_absolute"]
            gradmem = 0
            for lay in optimizer.state.items():
                ps = lay[0]
                if ps.grad != None:
                    gradmem += ps.grad.nelement() * ps.grad.element_size()
            self.mem_data["gradients"] = gradmem
        
        def fwd_end(module, *args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.time_data["fwd_end_absolute"] = time.time_ns()
            self.time_data["fwd_end_relative"] = self.time_data["fwd_end_absolute"] - self.time_data["step_begin_absolute"]
            self.mem_data["activation"] = torch.cuda.memory_allocated()
            
        def bwd_end(module, *args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize() 
            self.time_data["bwd_end_absolute"] = time.time_ns()
            self.time_data["bwd_end_relative"] = self.time_data["bwd_end_absolute"] - self.time_data["step_begin_absolute"]    
         
        def opt_step_end(module, *args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.time_data["opt_end_absolute"] = time.time_ns()
            self.time_data["opt_end_relative"] = self.time_data["opt_end_absolute"] - self.time_data["step_begin_absolute"]

        
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
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.time_data["step_end_absolute"] = time.time_ns()
        self.time_data["step_end_relative"] = self.time_data["step_end_absolute"] - self.time_data["step_begin_absolute"]
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



    
    """
    CONFIGURATION DETECTION USING PYTORCH HOOKS
    """


    def attach_hooks(self,objs_dict):

        if torch.cuda.current_device() != 0:
            return  
        
        model=None

        # DETECTING OBJECTS
        for name, obj in objs_dict:
            if isinstance(obj, transformers.trainer.Trainer):
                self.trainer=obj
                model= self.trainer.model_wrapped
                break # for hf trainers, we detect all objects from the trainer object      

        if model==None:
            print("No Model Object Found")
            return

        initial_params_dtype= next(model.parameters()).dtype #Model dtype is checked outside the hook because, once prepared by the trainer, it takes the dtype of the mixed precision
        
        print(f'\nLoaded Model parameters dtype: {initial_params_dtype}')
        self.configs_dict['Model loaded in dtype'] = initial_params_dtype
        

        # Getting the first layer from the innermost submodule
        first_layer = next(model.children())
        has_children = True
        while has_children == True:
            has_children = False
            first_layer=next(first_layer.children())
            for child in first_layer.children():
                has_children = True
        
        

        """ATTACHING HOOKS"""

        # Forward hook for BS and SeqLen
        self.layer_handle = first_layer.register_forward_hook(self.layer_hook_func)


        #STEPS
        self.fwd_step_handle=model.register_forward_hook(self.fwd_steps_hook_func)
        """
        For huggingFace trainers, one global step is one gradient accumulation. So if grad_accum_steps=n, then one global step happens after n forward passes.
        """
       
        # Forward Hook for AMP and Grad Checkpointing
        for name, module in model.named_modules():
            self.amp_handle=module.register_forward_hook(self.amp_grad_checkpoint_forward_hook) #mixed precision handle


         # Forward hook for all the other params
        self.model_handle=model.register_forward_hook(self.model_hook_func)




    def fwd_steps_hook_func(self,module,input,output):
        """
        This function keeps track of steps for HF trainers.
        In HF trainers, one global step is actually one gradient update. So if grad_accum_steps=n, then n forward passes make one global step.
        """
        if torch.cuda.current_device() != 0:
            return
       
        if self.step == self.trainer.state.global_step:
            self.num_fwd_pass=self.num_fwd_pass + 1

        elif self.grad_accum_steps == 0 :
            self.grad_accum_steps = self.num_fwd_pass
            self.step = self.step + 1
            self.num_fwd_pass=self.num_fwd_pass + 1
        else:
            self.step = self.step + 1
            self.num_fwd_pass=self.num_fwd_pass + 1




    def layer_hook_func(self,module, inp, out):
        """
        This function extracts the BS and seqlen from the input dimensions of first layer of the model
        """
        if torch.cuda.current_device() != 0:
            return
        
        if self.num_fwd_pass == self.target_step * self.grad_accum_steps : #check only during the examining step = global step (which is num_fwd_pass/grad_accum_steps).
            """Above conditions: First condition checks if the script is using HF trainers and if so makes sure that the examining step matches the step in progress bar (which is num_fwd_pass / grad_accum_step)"""
            
            inp_shape=inp[0].shape             
            
                
            bs=inp_shape[0]
            seqlen=inp_shape[1]
            print(f"\nBatch Size: {bs}")
            print(f"\nSeq Length: {seqlen}")

            self.configs_dict['Batch Size'] = bs
            self.configs_dict['Sequence Length'] = seqlen

            self.layer_handle.remove()





    def optimizer_hook_func(self,opt,args,kwargs):
        """
        This function calculates the gradient accumulation for non-HF trainer scripts.
        In custom training loops, gradient accumulation steps determine the number of forward passes in which one optimizer hook fires.
        """
        if torch.cuda.current_device() != 0:
            return

        
        print(f"optimizer: {args}")
        self.configs_dict['Optimizer Info'] = args
        self.opt_hook_fired = True
           
        self.grad_accum_handle.remove()





    def amp_grad_checkpoint_forward_hook(self,module, input, output):
        """
        This hook funtion helps check if AMP and gradient checkpointing is enabled.
        Gradient Checkpointing: if gradient activation is enabled only some of the activations are saved during forward pass
                                Hence only some of the outputs will have gradients. So we are checking if the outputs.requires_grad is false for any output. 
        AMP: If AMP is enabled, the output activations are calculated in multple dtypes. (However, Deepspeed's execution doesnot seem to follow this and to be further studied.
                                We have used a different logic for deepspeed which involves checking if the initial model loaded dtype matches the activaions dtype)
        Note: only the dtypes of all layers of the model are checked in this function. AMP is actually checked using these dtypes in the model_hook_func().
        """
        
        if torch.cuda.current_device() != 0:
            return

        if self.num_fwd_pass == self.target_step * self.grad_accum_steps : #check only during the examining step = global step (which is num_fwd_pass/grad_accum_steps).
            
            if not output[0].requires_grad and self.grad_checkpointing_checked==False: # if gradient activation is enabled only some of the activations are saved during forward pass. Hence only some of the outputs will have gradients. So we are checking if the outputs.requires_grad is false for any output. 
                self.grad_checkpointing_checked = True

            if str(output[0].dtype) not in self.dtypes_for_amp:
                self.dtypes_for_amp.append(str(output[0].dtype))
            
            
        if self.step==self.target_step+1:
            self.amp_handle.remove()




    def model_hook_func(self,module, input,output):

        """
        This Hook function does the following:
        1) Extracts Model Configs: num params, attention mechanism
        2) Attaches optimizer hooks.
        3) Extracts FSDP/Deepspeed Configs
        4) Checks for AMP with results from amp_grad_checkpoint_forward_hook
        """
    
        if torch.cuda.current_device() != 0:
            return

        is_ds_enabled=False #deepspeed
  
        module = self.trainer.model_wrapped #the objects are wrapped with fsdp or deepspeed only after the training starts. so we initialize it again. If fsdp/Ds is not used, this is the same as normal model object
        
        """   
        optimizer hook attached within model hook function because, for default trainer optimizrs, the optimizer object is not defined until training starts and is accessible only from here.
        optimizer hook is fired for every grad accum step unless there is gradient overflow.
        """
        if self.grad_accum_hook_attached==False: 
                if isinstance(self.trainer.optimizer,accelerate.optimizer.AcceleratedOptimizer):
                    
                    if hasattr(self.trainer.optimizer.optimizer, 'optimizer'):
                        self.grad_accum_handle=self.trainer.optimizer.optimizer.optimizer.register_step_pre_hook(self.optimizer_hook_func)
                    else:
                        self.grad_accum_handle=self.trainer.optimizer.optimizer.register_step_pre_hook(self.optimizer_hook_func)

                else:
                    self.grad_accum_handle=self.trainer.optimizer.register_step_pre_hook(self.optimizer_hook_func)
                self.grad_accum_hook_attached=True #to ensure that grad accum hook function is called only once and it doesnt interrupt the remaining part of this function. If this condition is not included, we get an error about states modified during iteration.
                
                if self.grad_accum_steps == 0:
                    "Make sure examining step is greater than grad accum steps."
                    return
                    

        if self.num_fwd_pass == self.target_step * self.grad_accum_steps : #check only during the examining step = global step (which is num_fwd_pass/grad_accum_steps).
            
            total_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f'\nModel has {total_params} parameters')
            self.configs_dict['Total Num Params']= total_params
                

            print("Gradient Accumulation steps:", self.grad_accum_steps)
            self.configs_dict['Gradient Accumulation steps'] = self.grad_accum_steps
            if self.opt_hook_fired == False: # Since examining step is greater than grad_accum steps, the optimizer hook should fire atleast once. But if it doesn't, it likely indicates gradient overflow.
                print("Warning: Optimizer hook was not fired. Could be due to Gradient Overflow.")



            # ATTENTION IMPLEMENTATION CHECKING
            attn_implementation = None
            for _,sub_module in module.named_modules(): #works for all hugging face models
                if "FlashAttention2" in type(sub_module).__name__ :
                    attn_implementation= "Flash Attention 2"
                    # print(type(sub_module).__name__)
                if "SdpaAttention" in type(sub_module).__name__ :
                    attn_implementation= "SDPA Attention"

            print("Attention Implementation: ", attn_implementation)
            self.configs_dict["Attention Implementation"]= attn_implementation





            #FSDP CONFIGS
            if isinstance(module, torch.distributed.fsdp.fully_sharded_data_parallel.FullyShardedDataParallel):
                print("\nFSDP CONFIGS:")
                print("\nNum processes: ", torch.cuda.device_count())
                print("\nSharding strategy: ",module.sharding_strategy)
                print("\nBackward prefetch: ",module.backward_prefetch)
                print("\nForward prefetch: ",module.forward_prefetch)
                print("\nMixed precision: ",module.mixed_precision.param_dtype)
                print("\nUse_orig_params: ",module._use_orig_params)
                self.configs_dict['Distributed Type'] = 'FSDP'
                self.configs_dict['Num Processes'] = torch.cuda.device_count()
                self.configs_dict['FSDP Configs']={'Sharding strategy':module.sharding_strategy , 
                                                   'Backward prefetch':module.backward_prefetch , 
                                                   'Forward prefetch': module.forward_prefetch,
                                                   'Mixed precision': module.mixed_precision.param_dtype,
                                                   'Use_orig_params': module._use_orig_params}

            
            #DEEPSPEED CONFIGS
            if isinstance(module, deepspeed.runtime.engine.DeepSpeedEngine):
                print("\nNum processes: ", torch.cuda.device_count())
                print(module._config._param_dict)
                is_ds_enabled=True
                self.configs_dict['Distributed Type'] = 'DeepSpeed'
                self.configs_dict['Num Processes'] = torch.cuda.device_count()
                self.configs_dict['Deepspeed Configs'] = module._config._param_dict

            


            # AMP CHECKING          
            amp = 'Not enabled'
            if (len(self.dtypes_for_amp)>1) and (is_ds_enabled == False): #the second condition added bcz first condition doesnt detect AMP for deepspeed
                print(f"Dtypes while training: {self.dtypes_for_amp}") #from amp hook. Printed here because if printed within the hook func, it is printed for each layers which we dont want.
                if "torch.bfloat16" in self.dtypes_for_amp:
                    amp='BF16'
                    print(f"Automatic Mixed Precision Training enabled with BF16")
                if "torch.float16" in self.dtypes_for_amp:
                    amp='FP16'
                    print(f"Automatic Mixed Precision Training enabled with FP16")


            elif is_ds_enabled == True:             
                if [str(self.configs_dict['Model loaded in dtype'])] != self.dtypes_for_amp:  #checking if the initial model loaded dtype matches the activaions dtype
                    if "torch.bfloat16" in self.dtypes_for_amp:
                        amp='BF16'
                        print(f"Automatic Mixed Precision Training enabled with BF16")
                    if "torch.float16" in self.dtypes_for_amp:
                        amp='FP16'
                        print(f"Automatic Mixed Precision Training enabled with FP16")


            else:
                print(f"Automatic Mixed Precision not enabled")

            self.configs_dict['Automatic Mixed Precision'] = amp



            #Gradient checkpointing checking

            if self.grad_checkpointing_checked == False:
                self.configs_dict['Gradient Checkpointing'] = 'Not Enabled'
                print("Gradient checkpointing not enabled")
            else:
                self.configs_dict['Gradient Checkpointing'] = 'Enabled'
                print("Gradient checkpointing enabled")


            


        if self.num_fwd_pass == self.target_step * self.grad_accum_steps + 1: #print results in target_step + 1
            print(self.configs_dict)
            self.model_handle.remove()



            
            
        
        



def model_hook_func(module, input,output):
    if step == 3:
    
        total_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f'\n\n{module.__class__.__name__} has {total_params} parameters')
        print(f'\n{module.__class__.__name__} parameters dtype: {next(module.parameters()).dtype}\n')
        model_handle.remove()

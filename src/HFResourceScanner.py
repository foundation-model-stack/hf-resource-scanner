import torch
from transformers.trainer_callback import TrainerCallback

class HFResourceScanner(TrainerCallback):
    def on_step_end(self, args, state, control, model, tokenizer, optimizer, **kwargs):
        # only calculate for master process in fsdp, other GPUs will be symmetrical
        if state and not state.is_world_process_zero:
            return

        if state.global_step != 5:
            return

        # run only once - for the 5th step

        data = {}
        data["cudamem"] = torch.cuda.memory_allocated()
        data["cuda_max_mem"] = torch.cuda.max_memory_allocated()
        data["model"] = model.get_memory_footprint()

        optimizer_mem = 0
        for lay in optimizer.state.items():
            lstate = lay[1]
            for v in lstate.values():
                if isinstance(v, torch.Tensor):
                    optimizer_mem += v.nelement() * v.element_size()
        data["optimizer"] = optimizer_mem

        # this may show empty gradients if we end up free'ing them
        gradmem = 0
        for ps in model.parameters():
            if ps.grad != None:
                gradmem += ps.grad.nelement() * ps.grad.element_size()
        data["gradients"] = gradmem

        print(data)

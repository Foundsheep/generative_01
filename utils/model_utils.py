import ..training_part.diffusers.src.diffusers as diffusers

def get_scheduler(scheduler_name):
    scheduler = None
    if scheduler_name == "DDPMScheduler":
        scheduler = diffusers.schedulers.DDPMScheduler()
    elif scheduler_name == "DDIMScheduler":
        scheduler = diffusers.schedulers.DDIMScheduler()
    elif scheduler_name == "DDPMParallelScheduler":
        scheduler = diffusers.schedulers.DDPMParallelScheduler()
    elif scheduler_name == "DDIMParallelScheduler":
        scheduler = diffusers.schedulers.DDIMParallelScheduler()
    elif scheduler_name == "AmusedScheduler":
        scheduler = diffusers.schedulers.AmusedScheduler()
    elif scheduler_name == "DDPMWuerstchenScheduler":
        scheduler = diffusers.schedulers.DDPMWuerstchenScheduler()
    elif scheduler_name == "DDIMInverseScheduler":
        scheduler = diffusers.schedulers.DDIMInverseScheduler()
    elif scheduler_name == "CogVideoXDDIMScheduler":
        scheduler = diffusers.schedulers.CogVideoXDDIMScheduler()
    # else:
    #     raise Exception, f"scheduler name should be given, but [{scheduler_name = }]"
    return scheduler
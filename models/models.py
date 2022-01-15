from .SCGAN import SCGAN


def create_model(opt, dataset):
    model = SCGAN(dataset)
    model.initialize(opt)
    print(f"model [{model.name()}] was created")
    return model

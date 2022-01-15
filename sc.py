from models.models import create_model
from options.test_options import TestOptions
from SCDataset.SCDataset import SCDataLoader


def main():
    opt = TestOptions().parse()
    data_loader = SCDataLoader(opt)
    SCGan = create_model(opt, data_loader)

    if opt.phase == "train":
        SCGan.train()
    elif opt.phase == "test":
        SCGan.test()

    print("Finished!!!")


if __name__ == "__main__":
    main()

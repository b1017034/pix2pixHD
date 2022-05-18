import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch


def convert_to_onnx(model, dummy_input,  output_name):
    torch.onnx.export(
        model,
        dummy_input,
        output_name,
        opset_version=10,
        verbose=False,
        export_params=True,
        input_names=['modelInput'],
        output_names=['modelOutput']
    )


if __name__ == '__main__':
    opt = TestOptions().parse(save=False)
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)

    for i, data in enumerate(dataset):
        input_size = torch.zeros(1,3,1080,1920)
        tensor_in = torch.Tensor(data['label'])
        convert_to_onnx(
            model,
            data['label'],
            opt.export_onnx
        )
        exit(0)

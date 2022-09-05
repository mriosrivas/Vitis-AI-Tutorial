from PIL import Image
import vart
import xir
import numpy as np

def preprocess_fn(image_path):
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((227, 227), Image.NEAREST)

    data = np.asarray(img, dtype="float32" )
    data = data/255.0
    return data

x0 = preprocess_fn('../dataset/dump/cat/cat.10046.jpg')
x1 = preprocess_fn('../dataset/dump/cat/cat.11175.jpg')
x2 = preprocess_fn('../dataset/dump/dog/dog.10048.jpg')
x3 = preprocess_fn('../dataset/dump/dog/dog.11175.jpg')

model = '../compilation/compiled_model/deploy.xmodel'

g = xir.Graph.deserialize(model)

root_subgraph = g.get_root_subgraph()

child_subgraph = root_subgraph.toposort_child_subgraph()

dpu = vart.Runner.create_runner(child_subgraph[1], 'run')

inputTensors = dpu.get_input_tensors()
outputTensors = dpu.get_output_tensors()
input_ndim = tuple(inputTensors[0].dims)
output_ndim = tuple(outputTensors[0].dims)


input_fixpos = inputTensors[0].get_attr("fix_point")
output_fixpos = outputTensors[0].get_attr("fix_point")

outputData = []
inputData = []

inputData = [np.empty(input_ndim, dtype=np.float32, order="C")]
outputData = [np.empty(output_ndim, dtype=np.float32, order="C")]

imageRun = inputData[0]

imageRun[0] = x0*(2**input_fixpos-1)
imageRun[1] = x1*(2**input_fixpos-1)
imageRun[2] = x2*(2**input_fixpos-1)
imageRun[3] = x3*(2**input_fixpos-1)

job_id = dpu.execute_async(inputData, outputData)
dpu.wait(job_id)

result = outputData[0]*(2**(output_fixpos)-1)
print(result.astype('uint8'))



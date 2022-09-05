#  Copyright 2021 Industrial Technology Research Institute
#
#  Copyright 2020 Xilinx Inc.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  NOTICE: This file has been modified by Industrial Technology Research Institute for AIdea "FPGA Edge AI – AOI Defect
#  Classification" competition

from ctypes import *
from typing import List
from PIL import Image

import cv2
import numpy as np
import vart
import os
import pathlib
import xir
import threading
import time
import sys
import argparse

divider = '------------------------------------'

def preprocess_fn_old(image_path):
    '''
    Image pre-processing.
    Rearranges from BGR to RGB then normalizes to range 0:1
    input arg: path of image file
    return: numpy array
    '''
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = image / 255.0
    image = cv2.resize(image, (227, 227), interpolation= cv2.INTER_NEAREST)
    #print('Image \n')
    #print(image)
    return image


def preprocess_fn(image_path, scale=1./255):
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((227, 227), Image.NEAREST)

#    img = np.array([np.array(img)])
    #img = np.array(img)
    #print('Image transformed \n')
    #print(f'image name = {image_path}')    
    #print(f'Image shape = {img.shape}')
    #data = img
    data = np.asarray(img, dtype="float32" )
    data = data/255.0
    #print(f'data {data}')
    #data = data*(2**6)
    #data = np.array(data*((2**6)), dtype='float32')
    #print(data)
    return data

    

def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def runDPU(id, start, dpu, img):
    '''get tensor'''
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    #print(f'inputTensors = {inputTensors}')
    #print(f'outputTensors = {outputTensors}')
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)
    #print(f'input_ndim = {input_ndim}')
    #print(f'output_ndim = {output_ndim}')

    input_fixpos = inputTensors[0].get_attr("fix_point")
    output_fixpos = outputTensors[0].get_attr("fix_point")
    print(f'input_fixpos = {input_fixpos}')
    print(f'output_fixpos = {output_fixpos}')
    
    
    batchSize = input_ndim[0]

    n_of_images = len(img)
    count = 0
    write_index = start
    while count < n_of_images:
        if count + batchSize <= n_of_images:
            runSize = batchSize
        else:
            runSize = n_of_images - count

        '''prepare batch input/output '''
        outputData = []
        inputData = []
        inputData = [np.empty(input_ndim, dtype=np.float32, order='C')]
        #print(f'inputData shape = {inputData[0].shape}')
        outputData = [np.empty(output_ndim, dtype=np.float32, order='C')]

        '''init input image to input buffer '''
        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j, ...] = (2**input_fixpos-1)*img[(count + j) % n_of_images].reshape(input_ndim[1:])
            #imageRun[j, ...] = img[(count + j) % n_of_images]
            #print(f'input_ndim[1:] : {input_ndim}')
            #print(f'imageRun[j, ...] : {imageRun[j, ...]}')


        '''run with batch '''
        job_id = dpu.execute_async(inputData, outputData)
        dpu.wait(job_id)

        '''store output vectors '''
        #print(f'runSize = {runSize}')
        for j in range(runSize):
            #out_q[write_index] = np.argmax((outputData[0][j]))
            out_q[write_index] = (outputData[0][j][0][0][0]*(2**output_fixpos-1)).astype('uint8')
            #print(f'output = {out_q[write_index]}')
            #print(f'out = {outputData}')
            write_index += 1
        count = count + runSize


def app(image_dir, threads, model):
    listimage = sorted(os.listdir(image_dir))
    runTotal = len(listimage)

    global out_q
    out_q = [None] * runTotal

    g = xir.Graph.deserialize(model)
    subgraphs = get_child_subgraph_dpu(g)
    all_dpu_runners = []
    for i in range(threads):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))

    ''' preprocess images '''
    print(divider)
    print('Pre-processing', runTotal, 'images...')
    img = []
    for i in range(runTotal):
        path = os.path.join(image_dir, listimage[i])
        img.append(preprocess_fn(path))

    '''run threads '''
    print('Starting', threads, 'threads...')
    threadAll = []
    start = 0
    for i in range(threads):
        if (i == threads - 1):
            end = len(img)
        else:
            end = start + (len(img) // threads)
        in_q = img[start:end]
        t1 = threading.Thread(target=runDPU, args=(i, start, all_dpu_runners[i], in_q))
        threadAll.append(t1)
        start = end

    time1 = time.time()
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()
    time2 = time.time()
    timetotal = time2 - time1

    fps = float(runTotal / timetotal)
    print(divider)
    print("Throughput=%.2f fps, total frames = %.0f, time=%.4f seconds" % (fps, runTotal, timetotal))

    ''' post-processing '''
    classes = ['0', '1']

    print('Post-processing', len(out_q), 'images..')

    with open('result.csv', 'w') as f:
        f.write(f'ID,Label\n')
        for i in range(len(out_q)):
            #prediction = classes[out_q[i]]
            #prediction = np.binary_repr(int(out_q[i]*2**3))
            #print(f'len_q = {len(out_q)}')
            prediction = out_q[i]
            f.write(f'{listimage[i]},{prediction}\n')
            #print(f'{listimage[i]},{prediction}\n')

    print(divider)

    with open('/cat-dog/time_info.txt', 'w') as f:
        f.write(f'FPS = {fps}\n')


# only used if script is run as 'main' from command line
def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--image_dir', type=str, default='images', help='Path to folder of images. Default is images')
    ap.add_argument('-t', '--threads', type=int, default=1, help='Number of threads. Default is 1')
    ap.add_argument('-m', '--model', type=str, default='customcnn.xmodel',
                    help='Path of xmodel. Default is customcnn.xmodel')

    args = ap.parse_args()

    print(divider)
    print('Command line options:')
    print(' --image_dir : ', args.image_dir)
    print(' --threads   : ', args.threads)
    print(' --model     : ', args.model)

    app(args.image_dir, args.threads, args.model)


def sigmoid(x):
    return 1/(1+np.exp(-x))


if __name__ == '__main__':
    main()


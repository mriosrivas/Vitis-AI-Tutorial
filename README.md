# Vitis-AI-Tutorial

This is a tutorial to develop a CNN on an Alveo U200 card using Vitis AI. In order to run this tutorial you will need an Alveo U200  card installed on your server, you can follow the installation procedure here [Alveo U200 Data Center Accelerator Card](https://www.xilinx.com/products/boards-and-kits/alveo/u200.html#gettingStarted).

## Clone this Repository

To follow this tutorial you will need to download this repository by typing

```bash
git clone https://github.com/mriosrivas/Vitis-AI-Tutorial.git
```

Later you will need to setup a script with the location where you downloaded these files, so keep in mind where you downloaded this repository.

## Vitis AI Installation

Clone the Vitis AI repository

```bash
git clone --recurse-submodules https://github.com/Xilinx/Vitis-AI  

cd Vitis-AI
```

#### Using Pre-built Docker

Download the latest Vitis AI Docker with the following command. This container runs on a CPU.

```
docker pull xilinx/vitis-ai-cpu:latest  
```

Add below line 90 of your `docker_run.sh` script the following code

```shell
-v $CLONED_PATH/Vitis-AI-Tutorial:/Vitis-AI-Tutorial \
```

Where `$CLONED_PATH` is the location where you downloaded this repository.

Then, run docker, using the command:

```
./docker_run.sh xilinx/vitis-ai-cpu:latest
```

## Train the model

To train the model you can start the Jupyter Notebook by running:

```bash
jupyter notebook Classifier.ipynb
```

This notebook will save an `h5` model inside the `checkpoints` folder. Models are saved based on the best binary accuracy performance on the validation data. For reference, a trained model is provided inside the `checkpoints` folder named `cat_dog_classifier20_0.879.h5`. In this case the model reached a `0.879` accuracy on the validation data at epoch `20`.

## Quantize the model

To quantize the model go inside the `quantization` folder. In it, there is a file called `quantize.py` where quantization is performed. You can set a different model, from the provided, model by changing the following line:

```python
model_path = '../checkpoints/cat_dog_classifier20_0.879.h5'
```

The you can run the script as follows:

```bash
python quantize.py
```

Your quantized model will be saved inside the `quantization` folder with the name `quantized_model.h5`.

You can also dump the simulation results by running:

```bash
python dump.py
```

 The dump results are saved inside the `quantization` folder in a new folder named `dump_model`.

## Compile the model

To compile the model go inside the `compilation` folder and run the compilation script as follows:

```bash
./compile.sh
```

This will create an `xmodel` file that will be deployed in the DPU. The `xmodel` is saved in the `compiled_model` folder.

If you want to see which part of the model will be implemented in the DPU and CPU, run the following script:

```bash
./graph_plot.sh
```

This will save a PNG image with the graph model in the `compiled_model` folder.

## Perform Inference

To perform inference on the model you can run 

```bash
./inference_run.sh
```

This will call the `inference.py` script along with the `xmodel` created in compilation and perform inference on the `/dataset/test` data.

There is also a simpler implementation called `simple_example.py` which goes into the process of running the model with just 4 images. Again the `xmodel` is used, but it is easier to understand.

You can run it by typing

```bash
python simple_example.py
```

## Test Accuracy

You can also test the accuracy of the deployed model using the `accuracy_calc.py` script located in the `inference` folder.

Just type

```shell
python accuracy_calc.py
```

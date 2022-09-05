vai_c_tensorflow2 \
    --model ../quantization/quantized_model.h5\
    --arch /opt/vitis_ai/compiler/arch/DPUCADF8H/U200/arch.json \
    --output_dir compiled_model \
    --net_name deploy \
    --options '{"input_shape": "4,227,227,3"}'

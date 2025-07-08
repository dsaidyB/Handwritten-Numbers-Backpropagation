import numpy as np

# scale times 8bit = original

# ---------- Parameters ----------
MAX_INT8 = 127
output_max_val = 40  # from testing across both layers, its actually like 14.68 in layer 1, 19.15 layer 2, 1 for input layer (not an output tho)
# outputLayer max is: 32.68

# Input to all layers already int8: (for input of raw data it has max value of 1 for white 1/127), 
# rest layers multiply by previous scale output value since that is the effective scaling compared to ORIGNIAL value
# layer1_input: 1/127, layer2_input: 0.3149606, outputLayer_input: 0.3149606
scale_input = 0.3149606

# ---------- Load Weights and Biases ----------
layer = "outputLayer"
weights = np.loadtxt(layer+"Weights.txt", delimiter=",")
biases = np.loadtxt(layer+"Biases.txt")

# ---------- Compute Scale Factors ----------
# Scale weights so that max(abs(weight)) maps to int8
max_w = np.max(np.abs(weights))
scale_weights = max_w / MAX_INT8  # max = scale * max_int_8

# Accumulated scale (input × weight)
scale_accum = scale_input * scale_weights # since s_output * output_int_8 = (w_int8 * s_w)*(input_int8 * input_scale) = output max val

# ---------- Quantize ----------
# Quantize weights to int8
weights_q = np.round(weights / scale_weights).astype(np.int8)

# Quantize biases to int32 using accum scale
biases_q = np.round(biases / scale_accum).astype(np.int32)

# ---------- Requantization scale for output ----------
# Output of this layer will be int32, we want to scale back to int8
# So: requant_scale = 127 / max_output_val (in the quant version), but max_output_val_quant = (w*s_w)*(max output int 8?)/s_output
scale_output = output_max_val / MAX_INT8
requant_scale = scale_accum / scale_output  # s_input * s_weight / s_output, requant_scale * max_output_8bit = max output, O WAIT MULTIPLY THIS BY max int 8 vals
# layer1:  7.620948/100000, layer2: 9.091541/1000, outputLayer: 9.426532/1000

# Save these for use in next layer
with open(layer+"_scale_factors.txt", "w") as f:
    f.write(f"scale_input:  {scale_input:.6e}\n")
    f.write(f"scale_weight: {scale_weights:.6e}\n")
    f.write(f"scale_accum:  {scale_accum:.6e}\n")
    f.write(f"scale_output: {scale_output:.6e}\n")
    f.write(f"requant_scale: {requant_scale:.6e}\n")

# ---------- Save HEX ----------
def write_hex_file(filename, array, bits=8):
    flat = array.flatten()
    with open(filename, "w") as f:
        for val in flat:
            if bits == 8:
                f.write(f"{int(val) & 0xFF:02x}\n")
            elif bits == 32:
                f.write(f"{int(val) & 0xFFFFFFFF:08x}\n")
            else:
                raise ValueError("Unsupported bit width")

write_hex_file(layer+"Weights.hex", weights_q, bits=8)
write_hex_file(layer+"Biases.hex", biases_q, bits=32)

# ---------- Save Readable Debug ----------
np.savetxt(layer+"Weights_int8.txt", weights_q, fmt="%d", delimiter=",")
np.savetxt(layer+"Biases_int32.txt", biases_q, fmt="%d")

print("Quantization done.")
print(f"  - weight scale:      {scale_weights:.6e}")
print(f"  - output max value:  {output_max_val:.6f}")
print(f"  - output scale:      {scale_output:.6e}")
print(f"  - requant scale:     {requant_scale:.6e}")
print("Files: "+layer+"Weights.hex, "+layer+"Biases.hex, "+layer+"_scale_factors.txt")

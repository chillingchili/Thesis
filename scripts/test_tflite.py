from ai_edge_litert.interpreter import Interpreter
import numpy as np
import glob

for p in sorted(glob.glob("/mnt/c/Users/hibye/Documents/ACTUAL PROJECTS/Thesis/runs/**/best_int8.tflite", recursive=True)):
    it = Interpreter(model_path=p)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    out = it.get_output_details()[0]
    x = np.zeros(inp["shape"], dtype=inp["dtype"])
    it.set_tensor(inp["index"], x)
    it.invoke()
    y = it.get_tensor(out["index"])
    print(p.split("Thesis/")[1], "in", inp["shape"].tolist(), inp["dtype"], "-> out", y.shape)
print("TFLITE_OK")

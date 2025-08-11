import sys, json
print(json.dumps({"error": "ONNX inference adapter not implemented in template. Please provide your ONNX handler."}))
sys.exit(0)
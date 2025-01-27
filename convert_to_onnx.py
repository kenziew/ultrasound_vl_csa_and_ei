import torch
import segmentation_models_pytorch as smp

def save_to_onnx(model, dummy_input, onnx_output_path):
    torch.onnx.export(
        model,                              # PyTorch model
        dummy_input,                        # Example input
        onnx_output_path,                   # Path to save ONNX model
        export_params=True,                 # Store trained parameters in the model file
        opset_version=11,                   # ONNX opset version (adjust based on your target)
        do_constant_folding=True,           # Fold constant nodes for optimization
        input_names=["input"],              # Name of input layer
        output_names=["output"],            # Name of output layer
        dynamic_axes={                      # Dynamic axes for variable-sized inputs
            "input": {0: "batch_size"}, 
            "output": {0: "batch_size"}
        }
    )

    print(f"ONNX model exported to {onnx_output_path}")

def main():
    model = smp.Unet(
        encoder_name="resnet50",
        classes = 2,
        in_channels=1,
    )
    device = torch.device("cuda:0")
    state_dict = torch.load("./training_logs/gaussian_blur+colorjitter/best_epoch-00.bin", map_location="cpu")
    model.load_state_dict(state_dict)

    dummy_input = torch.randn(1, 1, 512, 512)

    onnx_output_path = "./training_logs/gaussian_blur+colorjitter/model.onnx"
    save_to_onnx(model, dummy_input, onnx_output_path)

    model = model.to(device)
    dummy_input = dummy_input.to(device)
    onnx_output_path = "./training_logs/gaussian_blur+colorjitter/model_gpu.onnx"
    save_to_onnx(model, dummy_input, onnx_output_path)

if __name__ == "__main__":
    main()
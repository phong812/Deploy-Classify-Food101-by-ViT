import torch
import torchvision

def create_vit_best_model(num_classes: int, seed: int):
    weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
    transform = weights.transforms()
    model = torchvision.models.vit_b_16(weights=weights)
    
    for param in model.parameters():
        param.requires_grad = False
    
    torch.manual_seed(seed)    
    model.heads = torch.nn.Sequential(
        torch.nn.Linear(in_features=768, out_features=num_classes),
    )
    return model, transform
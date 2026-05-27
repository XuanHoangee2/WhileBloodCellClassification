import torch
from config.config_loader import load_config

config = load_config()
learning_rate = config["Domain_Adaptation_training"]["LEARNING_RATE"]
weight_decays = config["Domain_Adaptation_training"]["WEIGHT_DECAY"]
task_config = load_config(path = "config_task.yaml")
cls_learning_rate = task_config["Task_training"]["LEARNING_RATE"]
cls_weight_decays = task_config["Task_training"]["WEIGHT_DECAY"]
encoder_lr = task_config["Encoder_lr"]["weight"]

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def check_accuracy(loader,model,device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()
    
def get_optimizer(model, reconstruction, learning_rate=learning_rate, weight_decay=weight_decays):

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(reconstruction.parameters()),
        lr=learning_rate,
        weight_decay=float(weight_decay),
        betas=(0.9, 0.999)
    )
    return optimizer

def get_classification_optimizer(model,learning_rate=cls_learning_rate,weight_decay=cls_weight_decays,encoder_multiplier=encoder_lr):

    encoder_lr = learning_rate * encoder_multiplier

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.pixel_encoder.parameters(),
                "lr": encoder_lr
            },

            {
                "params": model.classifier.parameters(),
                "lr": learning_rate
            }
        ],

        weight_decay=float(weight_decay),
        betas=(0.9, 0.999)
    )

    return optimizer

def get_scaler():

    scaler = torch.cuda.amp.GradScaler()
    return scaler

def save_checkpoint(model, optimizer, epoch, loss,file_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, file_path)


def load_checkpoint(model, optimizer,file_path):
    checkpoint = torch.load(file_path)
    checkpoint_state_dict = checkpoint['model_state_dict']
    # Create a new state_dict with only the matching keys
    checkpoint_state_dict = {k.replace('module.', ''): v for k, v in checkpoint_state_dict.items()}
    filtered_state_dict = {k: v for k, v in checkpoint_state_dict.items() if k in model.state_dict()}
    model.load_state_dict(filtered_state_dict, strict=False)
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss
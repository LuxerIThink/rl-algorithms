import torch


def rename_and_save_checkpoint(old_checkpoint_path, new_checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(old_checkpoint_path)

    # Create a mapping of old keys to new keys
    key_mapping = {
        # Old DQN
        "fc1.weight": "layers.0.weight",
        "fc1.bias": "layers.0.bias",
        "fc2.weight": "layers.2.weight",
        "fc2.bias": "layers.2.bias",
        "fc3.weight": "layers.4.weight",
        "fc3.bias": "layers.4.bias",
        # Old PPO
        "fc.0.weight": "layers.0.weight",
        "fc.0.bias": "layers.0.bias",
        "fc.2.weight": "layers.2.weight",
        "fc.2.bias": "layers.2.bias",
    }

    # Create a new state_dict with renamed keys
    new_state_dict = {}
    for old_key, new_key in key_mapping.items():
        if old_key in checkpoint:
            new_state_dict[new_key] = checkpoint[old_key]

    # Update the state_dict in the checkpoint
    checkpoint = new_state_dict

    # Save the updated checkpoint to a new file
    torch.save(checkpoint, new_checkpoint_path)


if __name__ == "__main__":
    checkpoint_path = "../models/Hopper-v4_PPO.pth"
    output_path = "../models/Hopper-v4_PPO.pth"

    rename_and_save_checkpoint(checkpoint_path, output_path)

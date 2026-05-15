import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

def prepare_mia_data(batch_size=64):
    
    # Standard CIFAR-10 normalization
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download the full training and test datasets
    
    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    full_testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    # CIFAR-10 train has 50k images. We'll split this into Target Train and Shadow Train.
    
    target_train_len = len(full_trainset) // 2
    shadow_train_len = len(full_trainset) - target_train_len
    target_train_set, shadow_train_set = random_split(full_trainset, [target_train_len, shadow_train_len])

    # CIFAR-10 test has 10k images. We'll split this into Target Test and Shadow Test.
    
    target_test_len = len(full_testset) // 2
    shadow_test_len = len(full_testset) - target_test_len
    target_test_set, shadow_test_set = random_split(full_testset, [target_test_len, shadow_test_len])

    print(f"Target Train size: {len(target_train_set)} | Target Test size: {len(target_test_set)}")
    print(f"Shadow Train size: {len(shadow_train_set)} | Shadow Test size: {len(shadow_test_set)}")

    return target_train_set, target_test_set, shadow_train_set, shadow_test_set

if __name__ == "__main__":
    prepare_mia_data()
import matplotlib.pyplot as plt
import torch

from dataset import (
    create_dataloaders,
    CLASS_NAMES
)


train_loader, _, _ = create_dataloaders(
    batch_size=8
)

images, labels = next(
    iter(train_loader)
)


# Ham dua anh tu normalized ve gan anh goc
def denormalize(image):

    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    image = image * std + mean

    image = torch.clamp(
        image,
        0,
        1
    )

    return image


plt.figure(
    figsize=(14, 8)
)

for i in range(
    min(8, len(images))
):

    image = denormalize(
        images[i]
    )

    image = image.permute(
        1,
        2,
        0
    )

    label = labels[i].item()

    plt.subplot(
        2,
        4,
        i + 1
    )

    plt.imshow(image)

    plt.title(
        CLASS_NAMES[label].upper()
    )

    plt.axis("off")


plt.tight_layout()

plt.show()
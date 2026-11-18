import torch
import torch.nn.functional as F

def symmetric_clip_loss(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    temperature: int | float,
) -> dict:
    """
    Returns:
        loss: scalar tensor
        image_to_text_accuracy: scalar tensor
        text_to_image_accuracy: scalar tensor
        similarities: (B, B) tensor
    """

    image_embeddings = F.normalize(image_embeddings, p=2, dim=1, eps=1e-12)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=1, eps=1e-12)

    similarities = (image_embeddings @ text_embeddings.T) / temperature

    B = image_embeddings.shape[0]
    targets = torch.arange(B, device=image_embeddings.device)

    loss_i2t = F.cross_entropy(similarities, targets)

    loss_t2i = F.cross_entropy(similarities.T, targets)

    loss = (loss_i2t + loss_t2i) / 2

    image_to_text_accuracy = (
        (similarities.argmax(dim=1) == targets).to(image_embeddings.dtype).mean()
    )

    text_to_image_accuracy = (
        (similarities.argmax(dim=0) == targets).to(image_embeddings.dtype).mean()
    )

    return {
        "loss": loss,
        "image_to_text_accuracy": image_to_text_accuracy,
        "text_to_image_accuracy": text_to_image_accuracy,
        "similarities": similarities,
    }
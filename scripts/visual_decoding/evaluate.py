import torch
from torch.utils.data import DataLoader

from scripts.utils.results import compute_metrics


class Evaluation:
    def __init__(self, model, criterion, device, test_dataset):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.test_dataset = test_dataset

    def test(self, batch_size):
        self.model.eval()

        loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        total_loss = 0.0
        predictions = []
        labels = []

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                total_loss += self.criterion(logits, y).item()

                predictions.extend(logits.argmax(dim=1).cpu().tolist())
                labels.extend(y.cpu().tolist())

        test_loss = total_loss / len(loader)

        return compute_metrics(predictions, labels, test_loss)
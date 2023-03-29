import torch


def simple_train_torch(model: torch.nn.Module, data: torch.Tensor, labels: torch.Tensor) -> torch.nn.Module:
    """Simple training function for torch"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(10):
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, labels)
        loss.backward()
        optimizer.step()

        print("Epoch: {0}, Loss: {1}".format(epoch, loss.item()))

    return model


def simple_evaluate_torch(model: torch.nn.Module, data: torch.Tensor, labels: torch.Tensor) -> float:
    """Simple evaluation function for torch"""
    model.eval()
    output = model(data)
    loss = torch.nn.functional.nll_loss(output, labels)

    return loss.item()

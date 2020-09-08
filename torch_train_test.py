import numpy as np
from tqdm.autonotebook import tqdm

def train(model, train_loader, optimizer, criterion, device, char2index):
    model.train()
    epoch_loss = 0

    dataset_length = len(train_loader.dataset)
    progress_bar = tqdm(total=len(train_loader.dataset), desc='Epoch')

    train_losses = []

    batch_idx = 0
    for src, tgt in train_loader:
        batch_idx += 1
        if batch_idx == 500:
            break
        batch_size = tgt.shape[0]
        sequence_length = tgt.shape[1]

        src = src.view(sequence_length, batch_size).to(device)
        tgt = tgt.view(sequence_length, batch_size).to(device)

        optimizer.zero_grad()

        src_ohe = torch.nn.functional.one_hot(src, len(char2index)).to(torch.float).to(device)
        tgt_ohe = torch.nn.functional.one_hot(tgt, len(char2index)).to(torch.float).to(device)

        hidden = model.init_hidden(batch_size)  # torch.zeros(1, 1, 2).to(device)
        output, hidden = model(src_ohe, hidden)

        batch_loss = torch.mean(
            torch.stack([criterion(output.view(batch_size,
                                               sequence_length,
                                               len(char2index))[batch_entry_idx],
                                   tgt.view(batch_size, sequence_length)[batch_entry_idx])
                         for batch_entry_idx in range(batch_size)]))

        loss = batch_loss.mean()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        progress_bar.set_postfix(train_loss=np.mean(train_losses))
        progress_bar.update(batch_size)

    progress_bar.close()

    return np.mean(train_losses)


def test(model, train_loader, criterion, device, char2index, return_predictions=False):
    model.eval()

    epoch_loss = 0

    dataset_length = len(train_loader.dataset)
    progress_bar = tqdm(total=len(train_loader.dataset), desc='Epoch')

    test_losses = []

    outputs = []
    with torch.no_grad():
        for src, tgt in train_loader:
            batch_size = tgt.shape[0]
            sequence_length = tgt.shape[1]

            src = src.view(sequence_length, batch_size).to(device)
            tgt = tgt.view(sequence_length, batch_size).to(device)

            src_ohe = torch.nn.functional.one_hot(src, len(char2index)).to(torch.float).to(device)
            tgt_ohe = torch.nn.functional.one_hot(tgt, len(char2index)).to(torch.float).to(device)

            hidden = model.init_hidden(batch_size)  # torch.zeros(1, 1, 2).to(device)
            output, hidden = model(src_ohe, hidden)
            outputs.append(output)

            batch_loss = torch.mean(
                torch.stack([criterion(output.view(batch_size,
                                                   sequence_length,
                                                   len(char2index))[batch_entry_idx],
                                       tgt.view(batch_size, sequence_length)[batch_entry_idx])
                             for batch_entry_idx in range(batch_size)]))

            loss = batch_loss.mean()
            test_losses.append(loss.item())

            progress_bar.set_postfix(train_loss=np.mean(test_losses))
            progress_bar.update(batch_size)

    progress_bar.close()

    # todo: refactor
    if return_predictions:
        return outputs, test_losses
    else:
        return np.mean(test_losses)

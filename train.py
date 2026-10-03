from modules import plot_attention_matrix
import torch
import torch.nn as nn
import polars as pl


def seq2seq_train(model, train_dataloader, valid_dataloader, loss_fn, optimizer, device, epochs, encoder_only=False, visualize=False, id_to_token=None):
    is_decoder = not encoder_only
    model = model.to(device)
    best_val_loss = float('inf')
    patience = 3
    epochs_without_improvement = 0
    results = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for x_batch, y_batch in train_dataloader:
            x_batch = x_batch.to(device)
            x_input = x_batch[:, :-1]
            target = x_batch[:, 1:]
            if is_decoder:
                y_batch = y_batch.to(device)
                y_input = y_batch[:, :-1]
                target = y_batch[:, 1:]

            optimizer.zero_grad()

            if is_decoder:
                logits = model(x_input, y_input)
            else:
                logits, _ = model(x_input)

            logits = logits.transpose(1, 2)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
        train_loss = total_train_loss / len(train_dataloader)

        if visualize:
            # Plot attention matrix every n epochs
            cross_attention_weights = model.decoder.attention.attention_list
            if epoch % 3 == 0:
                # Decode tokens for axis labels on the first batch sample
                src_tokens = [id_to_token[int(idx)] for idx in x_input[0]]
                tgt_tokens = [id_to_token[int(idx)] for idx in y_input[0, :-1]]
                plot_attention_matrix(cross_attention_weights, src_tokens, tgt_tokens, epoch)


        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in valid_dataloader:
                x_batch = x_batch.to(device)
                x_input = x_batch[:, :-1]
                target = x_batch[:, 1:]
                if is_decoder:
                    y_batch = y_batch.to(device)
                    y_input = y_batch[:, :-1]
                    target = y_batch[:, 1:]
                    logits = model(x_input, y_input)
                else:
                    logits, _ = model(x_input)

                logits = logits.transpose(1, 2)
                loss = loss_fn(logits, target)

                total_val_loss += loss.item()
        val_loss = total_val_loss / len(valid_dataloader)
        print(
                f"Epoch {epoch+1:02d}:\n"
                f"train_loss: {train_loss:.4f} | "
                f"validation_loss: {val_loss:.4f}"
                )
        results.append([epoch+1, train_loss, val_loss])
        # save the model
        # torch.save(model.state_dict(), "best_lstm_encoder_model.pkl")

        # Early stopping
        if val_loss < best_val_loss:
            diff = best_val_loss - val_loss
            if diff <= 0.008:
                epochs_without_improvement += 1
            else:
                epochs_without_improvement = 0
            best_val_loss = val_loss
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stop initializing!")
                break

    return pl.DataFrame(results, schema=['epoch', 'train_loss', 'validation_loss'], orient="row")


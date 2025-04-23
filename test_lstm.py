import time
from pathlib import Path
from neural_bytes.lstm import LSTM
from neural_bytes.utils import initialize_empty_matrix, build_ascii_vocab, text_to_indices, build_word_vocab, text_to_word_indices
from neural_bytes import plot_loss

def char_lstm(file_path, params=None):
    if params is None:
        params = {"hidden_size":128, "seq_length":25, "learning_rate":0.1, "epochs":100, "sample_len":100}

    raw = Path(file_path).read_text(encoding="utf-8", errors="ignore").lower()
    char_to_ix, ix_to_char = build_ascii_vocab()
    data_ix = text_to_indices(raw, char_to_ix)
    n_chars, vocab_size = len(data_ix), 256

    hs = params["hidden_size"]
    sl = params["seq_length"]
    lr = params["learning_rate"]
    epochs = params["epochs"]
    samp = params["sample_len"]

    lstm = LSTM(vocab_size, hs, sl, lr)
    hprev = initialize_empty_matrix(hs, 1)
    cprev = initialize_empty_matrix(hs, 1)
    pointer = 0

    loss_hist = []
    samples = {}
    start = time.time()

    for epoch in range(1, epochs+1):
        if pointer + sl + 1 >= n_chars:
            pointer, hprev, cprev = 0, initialize_empty_matrix(hs, 1), initialize_empty_matrix(hs,1)

        inputs = data_ix[pointer: pointer+sl]
        targets = data_ix[pointer+1: pointer+sl+1]
        pointer += sl

        loss, cache, hprev, cprev = lstm.forward(inputs, targets, hprev, cprev)
        grads = lstm.backward(cache)
        lstm.update(grads)

        loss_hist.append(loss)
        if epoch % 10 == 0:
            print(f"[LSTM CHAR] Epoch {epoch:3d}/{epochs} — Loss: {loss:.4f}")

        if epoch in {20,40,60,80,100}:
            seed = inputs[0]
            idxs = lstm.sample(seed, samp, hprev)

            chars = []
            for i in idxs:
                chars.append(ix_to_char[i])
            samples[epoch] = "".join(chars)

    end = f"{time.time()-start:.1f}"
    print(f"Training finished in {end}s")
    plot_loss(loss_hist)

    for ep in sorted(samples):
        print(f"\nLSTM CHAR Sample at epoch {ep} | \n{samples[ep]}\n")


def word_lstm(file_path, params=None):
    if params is None:
        params = {"hidden_size":128, "seq_length":25, "learning_rate":0.1, "epochs":100, "sample_len":50}

    raw = Path(file_path).read_text(encoding="utf-8", errors="ignore").lower()
    word_to_ix, ix_to_word = build_word_vocab(raw, max_words=5000)
    data_w = text_to_word_indices(raw, word_to_ix)
    n_tok, vocab_size = len(data_w), 5000

    hs = params["hidden_size"]
    sl = params["seq_length"]
    lr = params["learning_rate"]
    epochs = params["epochs"]
    samp = params["sample_len"]

    lstm = LSTM(vocab_size, hs, sl, lr)
    hprev = initialize_empty_matrix(hs, 1)
    cprev = initialize_empty_matrix(hs, 1)
    pointer = 0

    loss_hist = []
    samples = {}
    start = time.time()

    for epoch in range(1, epochs+1):
        if pointer + sl + 1 >= n_tok:
            pointer, hprev, cprev = 0, initialize_empty_matrix(hs, 1), initialize_empty_matrix(hs,1)

        inputs = data_w[pointer: pointer+sl]
        targets = data_w[pointer+1: pointer+sl+1]
        pointer += sl

        loss, cache, hprev, cprev = lstm.forward(inputs, targets, hprev, cprev)
        grads = lstm.backward(cache)
        lstm.update(grads)

        loss_hist.append(loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {loss:.4f}")

        if epoch in {20,40,60,80,100}:
            seed = inputs[0]
            idxs = lstm.sample(seed, samp, hprev)

            words = []
            for i in idxs:
                words.append(ix_to_word[i])
            samples[epoch] = " ".join(words)
            
    end = f"{time.time()-start:.1f}"
    print(f"Training finished in {end}s")
    plot_loss(loss_hist)

    for ep in sorted(samples):
        print(f"\nLSTM WORD Sample at epoch {ep} | \n{samples[ep]}\n")


def main():
    base = Path("./JulesVerde")
    mystery = base / "The Mysterious Island.txt"
    earth = base / "From the Earth to the Moon.txt"

    params_char={"hidden_size":128, "seq_length":50, "learning_rate":0.01, "epochs":100, "sample_len":40}
    params_word={"hidden_size":128, "seq_length":50, "learning_rate":0.01, "epochs":100, "sample_len":40}

    char_lstm(mystery, params_char)
    word_lstm(earth, params_word)


if __name__ == "__main__":
    main()

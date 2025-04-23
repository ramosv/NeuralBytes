from pathlib import Path
from neural_bytes import RNN
from neural_bytes.utils import initialize_empty_matrix, build_ascii_vocab, text_to_indices, build_word_vocab, text_to_word_indices
from neural_bytes import plot_loss
import time

def char_level(file_path, params=None):
    if params is None:
        params = {"hidden_size":128, "seq_length":25, "learning_rate":0.1, "epochs":100, "sample_len":100}

    txt_path = file_path
    raw_text = txt_path.read_text(encoding="utf-8", errors="ignore").lower()
    char_to_ix, ix_to_char = build_ascii_vocab()
    data_ix = text_to_indices(raw_text, char_to_ix)
    n_chars, vocab_size = len(data_ix), 256

    # hyperparams
    hidden_size = params["hidden_size"]
    seq_length = params["seq_length"]
    learning_rate = params["learning_rate"]
    epochs = params["epochs"]
    sample_len = params["sample_len"]

    rnn = RNN(vocab_size, hidden_size, seq_length, learning_rate)
    hprev = initialize_empty_matrix(hidden_size, 1)
    pointer = 0

    loss_history = []
    samples = {}
    start = time.time()

    for epoch in range(1, epochs + 1):
        if pointer + seq_length + 1 >= n_chars:
            pointer, hprev = 0, initialize_empty_matrix(hidden_size, 1)

        inputs  = data_ix[pointer : pointer + seq_length]
        targets = data_ix[pointer + 1 : pointer + seq_length + 1]
        pointer += seq_length

        loss, cache, hprev = rnn.forward(inputs, targets, hprev)
        grads = rnn.backward(cache)
        rnn.update(grads)

        loss_history.append(loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {loss:.4f}")

        if epoch in {20, 40, 60, 80, 100}:
            seed = inputs[0]
            idxs = rnn.sample(seed, sample_len, hprev)
            chars = []

            for i in idxs:
                chars.append(ix_to_char[i])
            samples[epoch] = "".join(chars)

    end = f"{time.time()-start:.1f}"
    print(f"Training finished in {end}s")
    plot_loss(loss_history)

    for ep in sorted(samples):
        print(f"\nCHAR Sample at epoch {ep}\n{samples[ep]}\n")


def word_level(file_path, params=None):
    if params is None:
        params = {"hidden_size":128, "seq_length":25, "learning_rate":0.1, "epochs":100, "sample_len":200}

    txt_path = file_path
    raw_text = txt_path.read_text(encoding="utf-8", errors="ignore").lower()

    word_to_ix, ix_to_word = build_word_vocab(raw_text, max_words=5000)
    data_wix = text_to_word_indices(raw_text, word_to_ix)
    n_tokens, vocab_size = len(data_wix), 5000

    # hyperparams
    hidden_size = params["hidden_size"]
    seq_length = params["seq_length"]
    learning_rate = params["learning_rate"]
    epochs = params["epochs"]
    sample_len = params["sample_len"]

    rnn = RNN(vocab_size, hidden_size, seq_length, learning_rate)
    hprev = initialize_empty_matrix(hidden_size, 1)
    pointer = 0

    loss_history = []
    samples = {}
    start = time.time()

    for epoch in range(1, epochs + 1):
        if pointer + seq_length + 1 >= n_tokens:
            pointer, hprev = 0, initialize_empty_matrix(hidden_size, 1)

        inputs  = data_wix[pointer : pointer + seq_length]
        targets = data_wix[pointer + 1 : pointer + seq_length + 1]
        pointer += seq_length

        loss, cache, hprev = rnn.forward(inputs, targets, hprev)
        grads              = rnn.backward(cache)
        rnn.update(grads)

        loss_history.append(loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {loss:.4f}")

        if epoch in {20, 40, 60, 80, 100}:
            seed = inputs[0]
            idxs = rnn.sample(seed, sample_len, hprev)

            # map back to words
            words = []
            for i in idxs:
                words.append(ix_to_word[i])
            samples[epoch] = " ".join(words)

    end = f"{time.time()-start:.1f}"
    print(f"Training finished in {end}s")
    plot_loss(loss_history)

    for ep in sorted(samples):
        print(f"\nWORD Sample at epoch {ep}\n{samples[ep]}\n")


def main():

    all_files = Path("./JulesVerde")
    mystery = all_files / "The Mysterious Island.txt"
    around_ = all_files / "Around the World in Eighty Days.txt"
    earth = all_files / "From the Earth to the Moon.txt"
    off = all_files / "Off on a Comet.txt"
    master = all_files / "The Master of the World.txt"


    params_char={"hidden_size":128, "seq_length":50, "learning_rate":0.01, "epochs":100, "sample_len":40}
    params_word={"hidden_size":128, "seq_length":50, "learning_rate":0.01, "epochs":100, "sample_len":40}

    char_level(around_,params_char)
    word_level(master,params_word)

if __name__ == "__main__":
    main()
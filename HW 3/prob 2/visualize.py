import matplotlib.pyplot as plt
import dill as pickle


def plot_losses_accs(logger_pkl_path):
    with open(logger_pkl_path, 'rb') as f:
        logger = pickle.load(f)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), dpi=160)

    # plot losses
    axs[0].plot(
        range(1, len(logger['train losses']) + 1),
        logger['train losses'],
        label='train',
        color='#5f1b6b'
    )
    axs[0].plot(
        range(1, len(logger['val losses']) + 1),
        logger['val losses'],
        label='val',
        color='xkcd:amber'
    )
    axs[0].legend()
    axs[0].set_title('losses')
    axs[0].set_xlabel('epochs')
    axs[0].set_ylabel('loss')

    # plot accuracies
    axs[1].plot(
        range(1, len(logger['train accs']) + 1),
        logger['train accs'],
        label='train',
        color='#5f1b6b'
    )
    axs[1].plot(
        range(1, len(logger['val accs']) + 1),
        logger['val accs'],
        label='val',
        color='xkcd:amber'
    )
    axs[1].legend()
    axs[1].set_title('accuracies')
    axs[1].set_xlabel('epochs')
    axs[1].set_ylabel('accuracy (%)')

    plt.show()


if __name__ == "__main__":
    logger_pkl_path = 'models/adam_optimizer_dropout0.3/logger.pkl'
    plot_losses_accs(logger_pkl_path)

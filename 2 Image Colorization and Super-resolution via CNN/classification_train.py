from __future__ import print_function
from data_processor_c import *
from load_data import *
import time
from colourization import CNN,UNet

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def train(args, cnn=None):
    # Set the maximum number of threads to prevent crash in Teaching Labs
    torch.set_num_threads(5)
    # Numpy random seed
    npr.seed(args.seed)

    # Save directory
    save_dir = "outputs/" + args.experiment_name

    # LOAD THE COLOURS CATEGORIES
    colours = np.load(args.colours,encoding='bytes')[0]
    num_colours = np.shape(colours)[0]
    # INPUT CHANNEL
    num_in_channels = 1 if not args.downsize_input else 3
    # LOAD THE MODEL
    if cnn is None:
        if args.model == "CNN":
            cnn = CNN(args.kernel, args.num_filters, num_colours, num_in_channels)
        elif args.model == "UNet":
            cnn = UNet(args.kernel, args.num_filters, num_colours, num_in_channels)

    # LOSS FUNCTION
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=args.learn_rate)

    # DATA
    print("Loading data...")
    (x_train, y_train), (x_test, y_test) = load_cifar10()

    print("Transforming data...")
    train_rgb, train_grey = process(x_train, y_train, downsize_input=args.downsize_input)
    train_rgb_cat = get_rgb_cat(train_rgb, colours)
    test_rgb, test_grey = process(x_test, y_test, downsize_input=args.downsize_input)
    test_rgb_cat = get_rgb_cat(test_rgb, colours)

    # Create the outputs folder if not created already
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Beginning training ...")
    if args.gpu:
        cnn.cuda()
    start = time.time()

    train_losses = []
    valid_losses = []
    valid_accs = []
    for epoch in range(args.epochs):
        # Train the Model
        cnn.train()  # change model to 'train' mode
        losses = []
        for i, (xs, ys) in enumerate(get_batch(train_grey,
                                               train_rgb_cat,
                                               args.batch_size)):
            images, labels = get_torch_vars(xs, ys, args.gpu)
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn(images)

            loss = compute_loss(criterion,
                                outputs,
                                labels,
                                batch_size=args.batch_size,
                                num_colours=num_colours)
            loss.backward()
            optimizer.step()
            losses.append(loss.data.item())

        # plot training images
        if args.plot:
            _, predicted = torch.max(outputs.data, 1, keepdim=True)
            plot(xs, ys, predicted.cpu().numpy(), colours,
                 save_dir + '/train_%d.png' % epoch,
                 args.visualize,
                 args.downsize_input)

        # plot training images
        avg_loss = np.mean(losses)
        train_losses.append(avg_loss)
        time_elapsed = time.time() - start
        print('Epoch [%d/%d], Loss: %.4f, Time (s): %d' % (
            epoch + 1, args.epochs, avg_loss, time_elapsed))

        # Evaluate the model
        cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
        val_loss, val_acc = run_validation_step(cnn,
                                                criterion,
                                                test_grey,
                                                test_rgb_cat,
                                                args.batch_size,
                                                colours,
                                                save_dir + '/test_%d.png' % epoch,
                                                args.visualize,
                                                args.downsize_input,
                                                args.gpu)

        time_elapsed = time.time() - start
        valid_losses.append(val_loss)
        valid_accs.append(val_acc)
        print('Epoch [%d/%d], Val Loss: %.4f, Val Acc: %.1f%%, Time(s): %d' % (
            epoch + 1, args.epochs, val_loss, val_acc, time_elapsed))

    # Plot training curve
    plt.figure()
    plt.plot(train_losses, "ro-", label="Train")
    plt.plot(valid_losses, "go-", label="Validation")
    plt.legend()
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.savefig(save_dir + "/training_curve.png")

    if args.checkpoint:
        print('Saving model...')
        torch.save(cnn.state_dict(), args.checkpoint)

    return cnn

if __name__ == '__main__':
    args = AttrDict()
    args_dict = {
        'gpu': True,
        'valid': False,
        'checkpoint': "",
        'colours': './data/colours/colour_kmeans24_cat7.npy',
        'model': "CNN",  # ["CNN","Unet"]
        'kernel': 3,
        'num_filters': 32,
        'learn_rate': 0.001,
        'batch_size': 100,
        'epochs': 5,
        'seed': 0,
        'plot': True,
        'experiment_name': 'colourization_cnn',
        'visualize': False,
        'downsize_input': False,  # [False, True] Using 'True' to do super-resolution experiment
    }
    args.update(args_dict)
    cnn = train(args)
    '''
    # To visualize CNN
    args = AttrDict()
    args_dict = {
                  'colours':'./data/colours/colour_kmeans24_cat7.npy', 
                  'index':0,
                  'experiment_name': 'colourization_cnn',
                  'downsize_input':False,
    }
    args.update(args_dict)
    plot_activation(args, cnn)
    
    # To visualize Unet
    args = AttrDict()
    args_dict = {
                  'colours':'./data/colours/colour_kmeans24_cat7.npy', 
                  'index':0,
                  'experiment_name': 'colourization_unet',
                  'downsize_input':False,
    }
    args.update(args_dict)
    plot_activation(args, unet_cnn)
    
    # To visualize super-resolution
    args = AttrDict()
    args_dict = {
                  'colours':'./data/colours/colour_kmeans24_cat7.npy', 
                  'index':0,
                  'experiment_name': 'super_res_unet',
                  'downsize_input':True,
    }
    args.update(args_dict)
    plot_activation(args, sr_cnn)
    '''
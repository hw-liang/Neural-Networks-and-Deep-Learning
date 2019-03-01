from __future__ import print_function
from data_processor_r import *
from load_data import *
import time
from colour_regression import RegressionCNN

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

    # LOAD THE MODEL
    if cnn is None:
        cnn = RegressionCNN(args.kernel, args.num_filters)

    # Set up L2 loss
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=args.learn_rate)

    # Loading & transforming data
    print("Loading data...")
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    train_rgb, train_grey = process(x_train, y_train)
    test_rgb, test_grey = process(x_test, y_test)

    # Create the outputs folder if not created already
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Beginning training ...")
    if args.gpu:
        cnn.cuda()
    start = time.time()

    for epoch in range(args.epochs):
        # Train the Model
        cnn.train()  # Change model to 'train' mode
        for i, (xs, ys) in enumerate(get_batch(train_grey,
                                               train_rgb,
                                               args.batch_size)):
            images, labels = get_torch_vars(xs, ys, args.gpu)
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, args.epochs, loss.data.item()))

        # Evaluate the model
        cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
        losses = []
        for i, (xs, ys) in enumerate(get_batch(test_grey,
                                               test_rgb,
                                               args.batch_size)):
            images, labels = get_torch_vars(xs, ys, args.gpu)
            outputs = cnn(images)

            val_loss = criterion(outputs, labels)
            losses.append(val_loss.data.item())

        val_loss = np.mean(losses)
        print('Epoch [%d/%d], Val Loss: %.4f' % (epoch + 1, args.epochs, val_loss))

    print("Generating predictions...")
    plot(xs, ys, outputs.cpu().data,
         path=save_dir + "/regression_output.png", visualize=args.visualize)
    #     grey = test_grey[:15]
    #     gtrgb = test_rgb[:15]
    #     grey, gtrgb = get_torch_vars(test_grey[:15], test_rgb[:15], args.gpu)
    #     predrgb = cnn(grey)
    #     plot(grey.cpu().data, gtrgb.cpu().data, predrgb.cpu().data, colours=None,
    #          path=save_dir+"/regression_output.png", visualize=args.visualize)

    if args.checkpoint:
        print('Saving model...')
        torch.save(cnn.state_dict(), args.checkpoint)
    return cnn

if __name__ == '__main__':
    args = AttrDict()
    args_dict = {
                  'gpu': True,
                  'valid': False,
                  'checkpoint':"",
                  'kernel':3,
                  'num_filters':32,
                  'learn_rate':0.001,
                  'batch_size':100,
                  'epochs':5,
                  'seed':0,
                  'plot': True,
                  'experiment_name': 'regression_cnn',
                  'visualize': False,
                  'downsize_input': False,
    }
    args.update(args_dict)
    cnn = train(args)
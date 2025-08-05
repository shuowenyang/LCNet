import argparse
import os
import warnings
from LCNet_model import *
import torch.optim as optim
from data_processor import *
from trainer_new import *
from scheduler import *
from ptflops import get_model_complexity_info
warnings.filterwarnings("ignore")


def main():
    global args
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    model_dir = "./%s/%s_group_%d_ratio_%.2f_%.4f_new_s5_a2_gamma0" % (args.save_dir, args.model, args.group_num, args.sensing_rate,args.prior_sigma)
    log_file_name = "./%s/%s_group_%d_ratio_%d_%.2f.txt" % (model_dir, args.model, args.group_num, args.sensing_rate,args.alpha)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    torch.backends.cudnn.benchmark = True

    model = LCNet(sensing_rate=args.sensing_rate, LayerNo=args.layer_num)
    model = nn.DataParallel(model)
    model = model.to(device)

    criterion = F.mse_loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs - args.warm_epochs, eta_min=5e-5)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warm_epochs,
                                       after_scheduler=scheduler_cosine)


    train_loader = data_loader(args)

    if args.start_epoch > 0:
        pre_model_dir = model_dir
        checkpoint = torch.load("%s/net_params_%d.pth" % (pre_model_dir, args.start_epoch))
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint["epoch"] + 1
        for i in range(0, start_epoch):
            scheduler.step()
    else:
        start_epoch = args.start_epoch + 1
        scheduler.step()

    print("Model: %s , Sensing Rate: %.2f , Epoch: %d , Initial LR: %f\n" % (
        args.model, args.sensing_rate, args.epochs, args.lr))

    print('Start training')
    for epoch in range(start_epoch, args.epochs + 1):
        print('current lr {:.5e}'.format(scheduler.get_lr()[0]))
        loss = train(train_loader, model, criterion, args.sensing_rate, optimizer, device,args.prior_sigma,args.alpha)
        scheduler.step()
        print_data = "[%02d/%02d]Total Loss: %f\n" % (epoch, args.epochs, loss)

        output_file = open(log_file_name, 'a')
        output_file.write(print_data)
        output_file.close()

        if epoch > 50 and epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, "%s/net_params_%d.pth" % (model_dir, epoch))

    print('Trained finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LCNet', help='model name')
    parser.add_argument('--sensing_rate', type=float, default=0.10000, help='set sensing rate')
    parser.add_argument('--group_num', type=int, default=1, help='group number for training')
    parser.add_argument('--start_epoch', default=0, type=int, help='epoch number of start training')
    parser.add_argument('--warm_epochs', default=3, type=int, help='number of epochs to warm up')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=32, type=int, help='mini-batch size')
    parser.add_argument('--block_size', default=32, type=int, help='block size')
    parser.add_argument('--lr', '--learning_rate', default=2e-4, type=float, help='initial learning rate')
    parser.add_argument('--save_dir', help='The directory used to save the trained models',
                        default='save_temp', type=str)
    parser.add_argument('--layer_num', type=int, default=16, help='D2fm number of the Net')
    parser.add_argument('--beta', default=1e-5, type=float, help="weight for one prior, (default: 1e-5)")
    parser.add_argument('--alpha', default=1e-2, type=float, help="weight for another prior, (default: 1e-2)")

    main()

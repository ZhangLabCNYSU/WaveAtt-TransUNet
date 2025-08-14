import torch
import argparse
import math
import torch.optim.lr_scheduler as lr_scheduler 
from utils import *       
from WaveATT import WaveATT


def main(args):
    mkdir()
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print("device:",device)

    info = f"[train hyper-parameters: {args}]\n\n"
    write_info(info)

    trainDataset,valDataset,trainLoader,valLoader=get_data(
        img_size=args.imgSize,img_suff=args.img_f,mask_suff=args.mask_f,batch_size=args.batch_size
    )

    plot(data_loader=next(iter(trainLoader)))
    print('visual trainSet save successfully !!------>runs')


    model = TransUnet(
    img_dim=224,  
    in_channels=4,  
    classes=1,  
    vit_blocks=12,  
    vit_heads=12, 
    vit_dim_linear_mhsa_block=3072 
)
    model.to(device)


    optimizer = get_optim(args.optim,model,args.lr)

    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_mean_iou = 0.0
    train_loss_list = []       
    val_loss_list = []         
    train_miou_list = []        
    val_miou_list = []         
    train_mdice_list = []        
    val_mdice_list = []        
    lr_list = []

    info = "epoch\ttrain_loss\ttrain_mdice\ttrain_miou\tval_loss\tval_mdice\tval_miou\t" + "\n"
    write_info(info)

    print('\n\n-----------train-----------\n')
    lr = args.lr
    for epoch in range(args.epochs):
        print("[epoch:%d/%d]" % (epoch + 1, args.epochs),'learning rate:%f'%(lr))

        lr,train_loss, train_mdice,train_miou  = train_one_epoch(model=model, optim=optimizer, loader=trainLoader, device=device,num=len(trainLoader))
        scheduler.step()       
        val_loss,val_mdice,val_miou= evaluate(model=model, loader=valLoader, device=device,num=len(valLoader))
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_miou_list.append(train_miou)
        val_miou_list.append(val_miou)
        train_mdice_list.append(train_mdice)
        val_mdice_list.append(val_mdice)
        lr_list.append(lr)

        info = f"{epoch+1}\t\t{train_loss}\t\t{train_mdice}\t\t{train_miou}\t\t" \
               f"{val_loss}\t\t{val_mdice}\t\t{val_miou}" +"\n"
        write_info(info)

        if val_miou > best_mean_iou:   
            best_mean_iou = val_miou
            torch.save(model.state_dict(), './runs/weights/best.pth')

        if epoch == args.epochs-1 :
            torch.save(model.state_dict(),'./runs/weights/last.pth')


        print("train loss:%f\ttrain mdice:%f\ttrain miou:%f"%(train_loss,train_mdice,train_miou))
        print("val loss:%f\tval mdice:%f\tval miou:%f"%(val_loss,val_mdice,val_miou),end='\n\n')

    plt_metric(train_loss_list,val_loss_list,train_miou_list,val_miou_list,train_mdice_list,val_mdice_list)
    plot_lr_decay(lr_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TransUnet-SAM segmentation")
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--epochs", default=120, type=int)
    parser.add_argument("--optim", default='SGD',type=str, help='SGD、Adam、RMSProp')

    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--lrf',default=0.001,type=float)                 

    parser.add_argument("--img_f", default='.png', type=str)               
    parser.add_argument("--mask_f", default='.png', type=str)              

    parser.add_argument("--imgSize", default=[224,224],help='image size')              
    args = parser.parse_args()
    print(args)

    print('model: TransUnet-SAM')
    main(args)

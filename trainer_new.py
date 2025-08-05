import torch
from loss_unet import loss_fn
import torch.nn.functional as F
def train(train_loader, model, criterion, sensing_rate, optimizer, device,beta,alpha):
    model.train()
    sum_loss = 0
    for inputs in train_loader:
        inputs = inputs.to(device)#########B,1,96,96
        optimizer.zero_grad()
        outputs, sys_loss, initial_y, initial_Phi = model(inputs)###########B,2,96,96

        gamma1 = torch.Tensor([0]).to(device)
        I = torch.eye(int(sensing_rate * 1024)).to(device)
        loss_kl = loss_fn(outputs, inputs,
                          beta
                          )


        rec_y = F.conv2d(outputs[:, :1, ], initial_Phi, padding=0, stride=32, bias=None)
        loss_likelihood=criterion(rec_y, initial_y)

        loss_orth=torch.mul(criterion(sys_loss, I), gamma1)

        loss = loss_likelihood + loss_orth+alpha*loss_kl

        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
    return sum_loss

import os
import random
import time
import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
from torch.autograd import Variable
from dataloaders import split_cifar10 as dataloader
from networks import CNN as Predictor
from networks import Generator
import copy
import logging
from DSAAug import DiffAugment, ParamDiffAug

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')

    parser.add_argument('--model_path', type=str, default='saved_models/', help='path to save the model')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--saveout', type=str, default="save_result/",help='experiment result save path')

    parser.add_argument('--outer_loop', type=int, default=1, help='outer_loop number ')
    parser.add_argument('--inner_loop', type=int, default=1, help='inner_loop numbr')
    parser.add_argument('--Iteration', type=int, default=1, help='training iterations for generator')
    parser.add_argument('--epochs', type=int, default=1, help='num of task training epochs')
    parser.add_argument('--batch_real', type=int, default=512, help='batch size for real data in generator training')
    parser.add_argument('--traintype', type=str, default="KD",
                        choices=['joint','jointfilter',
                                 'KD',"KDfilter"])

    args = parser.parse_args()
    args.dsa_param = ParamDiffAug()
    args.dsa_strategy = "color_crop_cutout_flip_scale_rotate"
    return args


args = get_args()
loggername = "C10_conv2_{}_seed{}".format(args.traintype, args.seed)
logging.basicConfig(filename="loggerdir/{}.log".format(loggername), level=logging.DEBUG)
logger = logging.getLogger()
logger.info(args)
model_path = args.model_path

try:
    os.makedirs(model_path)
except FileExistsError:
    # directory already exists
    pass
try:
    os.makedirs(args.saveout)
except FileExistsError:
    # directory already exists
    pass

def main():
    torch.manual_seed(args.seed)
    train_queue, valid_queue, traindata_all, taskcla, inputsize = get_data()
    allacc = np.zeros((len(traindata_all), len(traindata_all)))
    model = Predictor.Net2(inputsize, taskcla)
    model = nn.DataParallel(model)
    model.to(device)
    buffer_generator = None
    for t in range(len(train_queue)):
        pre_model = None
        clock1 = time.time()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        if t < 1:
            for epoch in range(args.epochs):
                pre_model = None
                model = train(train_queue[t], t, model, optimizer)
        else:
            pre_model = copy.deepcopy(model)
            for epoch in range(args.epochs):
                if args.traintype == "joint":
                    model = train_seq(train_queue[t], t, model, optimizer, taskcla, buffer_generator,
                                      pre_model=pre_model)
                elif args.traintype == "jointfilter":
                    model = train_seq_filter(train_queue[t], t, model, optimizer, taskcla, buffer_generator,
                                             pre_model=pre_model)
                elif args.traintype == "KD":
                    model = train_kd(train_queue[t], t, model, optimizer, taskcla, buffer_generator,
                                     pre_model=pre_model)
                elif args.traintype == "KDfilter":
                    model = train_kd_filter(train_queue[t], t, model, optimizer, taskcla, buffer_generator,
                                            pre_model=pre_model)

        print("Done task {}.".format(t))
        for oldt in range(t + 1):
            acc = eval(model, valid_queue[oldt], oldt)
            allacc[t, oldt] = acc
            logger.info("=====> task {} acc {}".format(oldt, acc))

        clock2 = time.time()
        generatorfile = model_path + "C10_conv2_generator_task{}_seed{}.pt".format(t, args.seed)
        if os.path.exists(generatorfile):
            print("*" * 10, "load generator")
            buffer_generator = torch.load(generatorfile)
        else:
            logger.info("task:{}/{}".format(t, len(train_queue)))
            if t < len(train_queue) - 1:
                buffer_generator = Generator_learn(t, traindata_all, inputsize, taskcla, buffer_generator)
                torch.save(buffer_generator, generatorfile)

        clock3 = time.time()
        print('train time={:5.1f}ms, buffer-update time={:5.1f}ms'.format((clock2 - clock1), (clock3 - clock2)))

    np.save(args.saveout + loggername, allacc)
    logger.info(allacc)
    logger.info(np.mean(allacc[len(allacc) - 1]))


def train(train_queue, t, model, optimizer):
    for step, (input, target) in enumerate(train_queue):
        model.train()
        input = Variable(input, requires_grad=False).to(device)
        target = Variable(target, requires_grad=False).to(device)

        output = model(input)[t]
        loss = F.cross_entropy(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def train_seq_filter(train_queue, t, model, optimizer, taskcla, buffer_generator=None, pre_model=None, latent_dim=100):
    _, class_num = taskcla[0]
    hyper = 1
    model.train()
    for step, (x, y) in enumerate(train_queue):
        num_x = len(x)
        x, y = x.to(device), y.to(device)

        noise = torch.randn((len(x), latent_dim), device=device)
        lab_pre = Variable(torch.LongTensor([random.randint(0, class_num * t - 1) for _ in range(len(noise))])).to(
            device)
        img_pre = buffer_generator(noise, lab_pre)

        lab_pre_output = pre_model(img_pre)
        for pre_t in range(t):
            boolArr = (lab_pre / class_num).int() == pre_t
            if boolArr.sum() > 0:
                _, lab_pre_predict = lab_pre_output[pre_t][boolArr].max(1)
                BoolArr_true = lab_pre_predict == (lab_pre[boolArr] - class_num * pre_t)
                x = torch.cat((x, img_pre[boolArr][BoolArr_true]), 0)
                y = torch.cat((y, lab_pre[boolArr][BoolArr_true]), 0)

        seed_data = int(time.time() * 1000) % 100000
        x = DiffAugment(x, args.dsa_strategy, seed=seed_data, param=args.dsa_param)

        outputs = model(x)
        loss = F.cross_entropy(outputs[t][0:num_x], y[0:num_x])
        for pre_t in range(t):
            boolArr = (y[num_x:] / class_num).int() == pre_t
            if boolArr.sum() > 0:
                loss += F.cross_entropy(outputs[pre_t][num_x:][boolArr], y[num_x:][boolArr] - class_num * pre_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def train_kd_filter(train_queue, t, model, optimizer, taskcla, buffer_generator=None, pre_model=None, latent_dim=100):
    _, class_num = taskcla[0]
    T = 2
    pre_model.eval()
    model.train()
    for step, (x, y) in enumerate(train_queue):
        num_x = len(x)
        x, y = x.to(device), y.to(device)

        noise = torch.randn((len(x), latent_dim), device=device)
        lab_pre = Variable(torch.LongTensor([random.randint(0, class_num * t - 1) for _ in range(len(noise))])).to(
            device)
        img_pre = buffer_generator(noise, lab_pre)

        lab_pre_output = pre_model(img_pre)
        for pre_t in range(t):
            boolArr = (lab_pre / class_num).int() == pre_t
            if boolArr.sum() > 0:
                _, lab_pre_predict = lab_pre_output[pre_t][boolArr].max(1)
                BoolArr_true = lab_pre_predict == (lab_pre[boolArr] - class_num * pre_t)
                print("previous task:{},data len:{}/{}".format(pre_t, len(lab_pre_predict), BoolArr_true.sum()))
                x = torch.cat((x, img_pre[boolArr][BoolArr_true]), 0)
                y = torch.cat((y, lab_pre[boolArr][BoolArr_true]), 0)

        seed_data = int(time.time() * 1000) % 100000
        x = DiffAugment(x, args.dsa_strategy, seed=seed_data, param=args.dsa_param)

        outputs = model(x)
        outputs_T = pre_model(x)
        loss = F.cross_entropy(outputs[t][0:num_x], y[0:num_x])
        for pre_t in range(t):
            boolArr = (y[num_x:] / class_num).int() == pre_t
            if boolArr.sum() > 0:
                loss += nn.KLDivLoss()(F.log_softmax(outputs[pre_t][num_x:][boolArr] / T, dim=1),
                                       F.softmax(outputs_T[pre_t][num_x:][boolArr] / T, dim=1)) * (T * T)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def train_seq(train_queue, t, model, optimizer, taskcla, buffer_generator=None, pre_model=None, latent_dim=100):
    _, class_num = taskcla[0]
    model.train()
    for step, (x, y) in enumerate(train_queue):
        num_x = len(x)
        x, y = x.to(device), y.to(device)

        noise = torch.randn((len(x), latent_dim), device=device)
        lab_pre = Variable(torch.LongTensor([random.randint(0, class_num * t - 1) for _ in range(len(noise))])).to(
            device)
        img_pre = buffer_generator(noise, lab_pre)

        x = torch.cat((x, img_pre), 0).to(device)

        outputs = model(x)

        loss = F.cross_entropy(outputs[t][0:num_x], y)
        for pre_t in range(t):
            boolArr = (lab_pre / class_num).int() == pre_t
            if boolArr.sum() > 0:
                loss += F.cross_entropy(outputs[pre_t][num_x:][boolArr], lab_pre[boolArr] - class_num * pre_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def train_kd(train_queue, t, model, optimizer, taskcla, buffer_generator=None, pre_model=None, latent_dim=100):
    _, class_num = taskcla[0]
    hyper = 1
    T = 2
    pre_model.eval()
    model.train()
    for step, (x, y) in enumerate(train_queue):
        num_x = len(x)
        x, y = x.to(device), y.to(device)

        noise = torch.randn((len(x), latent_dim), device=device)
        lab_pre = Variable(torch.LongTensor([random.randint(0, class_num * t - 1) for _ in range(len(noise))])).to(
            device)
        img_pre = buffer_generator(noise, lab_pre)

        x = torch.cat((x, img_pre), 0).to(device)
        # x = DiffAugment(x)
        seed_data = int(time.time() * 1000) % 100000
        x = DiffAugment(x, args.dsa_strategy, seed=seed_data, param=args.dsa_param)
        outputs = model(x)
        loss = F.cross_entropy(outputs[t][0:num_x], y)

        outputs_T = pre_model(x)
        for pre_t in range(t):
            boolArr = (lab_pre / class_num).int() == pre_t
            if boolArr.sum() > 0:
                loss += nn.KLDivLoss()(F.log_softmax(outputs[pre_t][num_x:][boolArr] / T, dim=1),
                                       F.softmax(outputs_T[pre_t][num_x:][boolArr] / T, dim=1)) * (T * T)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def get_data():
    data, taskcla, inputsize = dataloader.get(seed=args.seed, tasknum=5)
    train_queue = []
    valid_queue = []
    traindata_all = []
    for t, ncla in taskcla:
        taskdata = []
        for i in range(len(data[t]['train']['x'])):
            taskdata.append([data[t]['train']['x'][i], data[t]['train']['y'][i], t])
        traindata_all.append(taskdata)

        taskdata = []
        for i in range(len(data[t]['train']['x'])):
            taskdata.append([data[t]['train']['x'][i], data[t]['train']['y'][i]])
        train_queue.append(torch.utils.data.DataLoader(
            taskdata, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=16, pin_memory=True))

        taskdata = []
        for i in range(len(data[t]['test']['x'])):
            taskdata.append([data[t]['test']['x'][i], data[t]['test']['y'][i]])
        valid_queue.append(torch.utils.data.DataLoader(
            taskdata, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=16, pin_memory=True))
    return train_queue, valid_queue, traindata_all, taskcla, inputsize


def eval(model, dataloader, t=100):
    total_acc = 0
    total_num = 0
    model.eval()
    for step, (input, target) in enumerate(dataloader):
        input = Variable(input, requires_grad=False).to(device)
        target = Variable(target, requires_grad=False).to(device)
        if t < 100:
            output = model.forward(input)[t]
        else:
            output = model.forward(input)
        _, pred = output.max(1)
        hits = (pred == target).float()

        total_acc += hits.sum().data.cpu().numpy()
        total_num += len(input)
    return total_acc / total_num


def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4:  # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2:  # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1:  # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return 0

    dis_weight = torch.sum(
        1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis


def match_loss(gw_syn, gw_real):
    dis = torch.tensor(0.0).to(device)
    for ig in range(len(gw_real)):
        gwr = gw_real[ig]
        gws = gw_syn[ig]
        dis += distance_wb(gwr, gws)
    return dis


def update_epoch(generator, net, optimizer_net, total_class_num):
    net = net.to(device)
    generator.eval()
    net.train()
    gen_datasize = args.batch_size

    noise = torch.randn((gen_datasize, 100), device=device)
    lab = Variable(torch.LongTensor([random.randint(0, total_class_num - 1) for _ in range(gen_datasize)])).to(device)
    img = generator(noise, lab)

    output = net(img)
    loss = F.cross_entropy(output, lab)

    optimizer_net.zero_grad()
    loss.backward()
    optimizer_net.step()

    return net


def Generator_learn(t, traindata_all, inputsize, taskcla, buffer_generator=None):
    datatep = []
    for i in range(len(traindata_all[t])):
        datatep.append([traindata_all[t][i][0], traindata_all[t][i][1]])
    traindataloder = torch.utils.data.DataLoader(
        datatep, batch_size=int(args.batch_real / (t + 1)), shuffle=True, drop_last=False, num_workers=16,
        pin_memory=True)

    ''' initialize the generator '''
    latent_dim = 100
    total_class_num = 0
    for pre_t in range(t + 1):
        total_class_num += taskcla[pre_t][1]
    print("total_class_num,", total_class_num)

    generator = Generator.Generator_Conv2(inputsize, class_num=total_class_num, latent_dim=latent_dim)
    generator = nn.DataParallel(generator)
    generator.to(device)

    gen_lr = 0.005
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=gen_lr, betas=(0.5, 0.999))
    gen_optimizer.zero_grad()

    ''' training generator'''
    for it in range(args.Iteration):
        if it in [int(0.25 * args.Iteration), int(0.5 * args.Iteration), int(0.8 * args.Iteration)]:
            gen_lr = gen_lr * 0.2
            gen_optimizer = torch.optim.Adam(generator.parameters(), lr=gen_lr, betas=(0.5, 0.999))
        net = Predictor.Net(inputsize, total_class_num)
        net = nn.DataParallel(net)
        net.to(device)
        net.train()
        net_parameters = list(net.parameters())
        optimizer_net = torch.optim.SGD(net.parameters(), lr=1e-3)
        optimizer_net.zero_grad()

        ## outer loop
        for ol in range(args.outer_loop):
            for img_real, lab_real in traindataloder:
                generator.train()

                img_real, lab_real = img_real.to(device), lab_real.to(device)
                lab_real = lab_real + total_class_num - 2

                if buffer_generator is not None:
                    noise = torch.randn((args.batch_real - len(img_real), latent_dim), device=device)
                    lab_pre = Variable(torch.LongTensor(
                        [random.randint(0, total_class_num - taskcla[t][1] - 1) for _ in range(len(noise))])).to(device)
                    img_pre = buffer_generator(noise, lab_pre)

                    img_real = torch.cat((img_real, img_pre), 0)
                    lab_real = torch.cat((lab_real, lab_pre), 0)

                output_real = net(img_real)
                loss_real = F.cross_entropy(output_real, lab_real)
                gw_real = torch.autograd.grad(loss_real, net_parameters)
                gw_r = []
                for _ in gw_real:
                    if _ is not None:
                        gw_r.append(_.detach().clone())

                # Sample random vectors
                noise = torch.randn((len(img_real), latent_dim), device=device)
                lab_syn = lab_real.clone().to(device)
                img_syn = generator(noise, lab_syn)

                output_syn = net(img_syn)
                loss_syn = F.cross_entropy(output_syn, lab_syn)
                gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                loss = match_loss(gw_syn, gw_r)
                gen_optimizer.zero_grad()
                loss.backward()
                gen_optimizer.step()

                ## inner loop
                for il in range(args.inner_loop):
                    update_epoch(generator, net, optimizer_net, total_class_num)

                print("=>iter {}, out_loop {}, matchloss:{},loss_real:{} => ".format(it, ol, loss.item(),
                                                                                     loss_real.item()))
    return generator


if __name__ == '__main__':
    main()

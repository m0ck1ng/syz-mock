import os
import subprocess
import sys
import torch
import torch.nn as nn
import random
import datetime
import shutil
from torch.utils.data import DataLoader
import torch.optim as optim

from django.http import HttpResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt

from .lang_model.utils import read_corpus, read_syscalls
from .lang_model.dataset import Vocab, SysDataset
from .lang_model.model import RelationModel, train

@csrf_exempt
def model_train(request):
    if request.method == "POST":
        now_time = datetime.datetime.now()
        syzdir = request.POST.get("syzdir", None)
        workdir = request.POST.get("workdir", None)
        pretrain = request.POST.get("workdir", "false")

        if not syzdir or not workdir \
            or not os.path.isdir(syzdir) \
            or not os.path.isdir(workdir):
            return HttpResponse("error: invalid parameters")

        corpus_dir = "/tmp/syz-mocking-corpus"
        if os.path.exists(corpus_dir):
            shutil.rmtree(corpus_dir)

        cmd = f"{syzdir}/bin/syz-db unpack {workdir}/corpus.db {corpus_dir}"
        ret = subprocess.run(cmd, shell=True, timeout=30)
        if ret.returncode != 0:
            print(f"error: {ret}")
            return HttpResponse("error: fail to unpack corpus")
        
        syscalls = read_syscalls(f"{settings.BASE_DIR}/api/lang_model/data/targetSyscalls")
        corpus = read_corpus(corpus_dir)

        # syzcorpus_dir = '/data5/corpus/syzkaller_corpus'
        # corpus = read_corpus(syzcorpus_dir)

        vocab = Vocab()
        vocab.addDict(syscalls)
        sys_dataset = SysDataset(corpus, vocab)

        train_size =  int(0.9*len(sys_dataset))
        test_size = len(sys_dataset)-train_size
        train_dataset, test_dataset = torch.utils.data.random_split(sys_dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        num_epoch = 50
        lr = 0.001
        embed_dim = 64
        hidden_dim = 128

        model = RelationModel(hidden_dim, embed_dim, len(vocab), device).to(device)
        if pretrain == "true":
            pretrain_model_path = f"{settings.BASE_DIR}/api/lang_model/data/pretrain_model"
            model.load_state_dict(torch.load(pretrain_model_path))

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        model_path = f"{settings.BASE_DIR}/api/lang_model/checkpoints"
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_file = train(train_loader, test_loader, model, \
            model_path, num_epoch, optimizer, \
            lr, criterion, device)

        time_cost = datetime.datetime.now() - now_time
        print(f"model_update_time_cost: {time_cost}")
        
        return HttpResponse(model_file)
        

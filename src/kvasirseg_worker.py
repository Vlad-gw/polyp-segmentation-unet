# -*- coding: utf-8 -*-
"""ЛР №4, вариант №4: исполняемый модуль Kaggle (создаётся автоматически из Notebook).
Одна команда = одна эпоха / калибровка / тестирование / предсказание.
Новые результаты не смешиваются с результатами предыдущего Notebook.
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')  # Используется первая GPU; T4 x2 не требует DataParallel.
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

cv2.setNumThreads(0)
torch.set_num_threads(2)

SEED = 2026
SIZE = int(os.environ.get('POLYP_SIZE', '256'))
BATCH_SIZE = int(os.environ.get('POLYP_BATCH', '4'))
CUSTOM_EPOCHS = int(os.environ.get('POLYP_UNET_EPOCHS', '32'))
READY_EPOCHS = int(os.environ.get('POLYP_READY_EPOCHS', '25'))
PATIENCE = int(os.environ.get('POLYP_PATIENCE', '9'))
DATA_ROOT = Path(os.environ.get('POLYP_DATA_ROOT', '/kaggle/input'))
ROOT = Path(os.environ.get('POLYP_OUTPUT', '/kaggle/working/kvasirseg_variant4_new'))
PRETRAINED_ENCODER = os.environ.get('POLYP_PRETRAINED', '1') == '1'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CKPT = ROOT / 'checkpoints'
TABLES = ROOT / 'tables'
FIGURES = ROOT / 'figures'
PRED = ROOT / 'predictions'
for p in (ROOT, CKPT, TABLES, FIGURES, PRED):
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    name: str
    kind: str = 'custom'
    batch_norm: bool = False
    dropout: float = 0.0
    loss: str = 'bce'
    augment: bool = False
    scheduler: bool = False
    upsample: str = 'bilinear'
    pool: str = 'max'
    activation: str = 'relu'
    kernel: int = 3
    filters: int = 16
    lr: float = 0.001
    epochs: int = CUSTOM_EPOCHS


# E0 и E1 обучаются с ОДИНАКОВОЙ взвешенной BCE: единственное изменение — BN и Dropout.
# E2 добавляет Dice, E3 — аугментации, E4 — планировщик LR, E5 — transpose upsampling.
UNET = [
    Config('E0_UNet_base'),
    Config('E1_UNet_BN_Dropout', batch_norm=True, dropout=.10),
    Config('E2_UNet_BCE_Dice', batch_norm=True, dropout=.10, loss='bce_dice'),
    Config('E3_UNet_Augmentation', batch_norm=True, dropout=.10, loss='bce_dice', augment=True),
    Config('E4_UNet_LR_Schedule', batch_norm=True, dropout=.10, loss='bce_dice', augment=True, scheduler=True),
    Config('E5_UNet_Transpose', batch_norm=True, dropout=.10, loss='bce_dice', augment=True, scheduler=True, upsample='transpose'),
]
EXTRA = [
    Config('E6_UNet_AvgPool', batch_norm=True, dropout=.10, loss='bce_dice', augment=True, scheduler=True, pool='avg'),
    Config('E7_UNet_LeakyReLU', batch_norm=True, dropout=.10, loss='bce_dice', augment=True, scheduler=True, activation='leaky_relu'),
]
READY = [
    Config('FPN_ResNet34', kind='fpn', loss='bce_dice', augment=True, scheduler=True, lr=.0003, epochs=READY_EPOCHS),
    Config('DeepLabV3Plus_ResNet34', kind='deeplab', loss='bce_dice', augment=True, scheduler=True, lr=.0003, epochs=READY_EPOCHS),
]
CONFIGS = {c.name: c for c in UNET + EXTRA + READY}


def config(name):
    if name not in CONFIGS:
        raise ValueError('Неизвестная модель: ' + str(name))
    return CONFIGS[name]


def atomic_json(payload, target):
    target = Path(target)
    tmp = Path(str(target) + '.part')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, target)


def atomic_checkpoint(payload, target):
    tmp = Path(str(target) + '.part')
    torch.save(payload, tmp)                    # Пишем сначала временный файл.
    os.replace(tmp, target)                   # Недописанный checkpoint не заменяет рабочий.


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def find_pairs():
    """Связываем исходное изображение и маску по одинаковому имени без расширения."""
    candidates = []
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f'Отсутствует {DATA_ROOT}: подключите Kvasir-SEG через Add Input')
    for image_dir in DATA_ROOT.rglob('images'):
        masks_dir = image_dir.parent / 'masks'
        if masks_dir.is_dir():
            candidates.append((int('kvasir' in str(image_dir).lower()), image_dir, masks_dir))
    if not candidates:
        raise FileNotFoundError('Не найдены соседние папки images и masks. Нужен debeshjha1/kvasirseg')
    candidates.sort(key=lambda t: (t[0], str(t[1])), reverse=True)
    _, images_dir, masks_dir = candidates[0]
    allowed = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    images = {p.stem: p for p in images_dir.iterdir() if p.suffix.lower() in allowed}
    masks = {p.stem: p for p in masks_dir.iterdir() if p.suffix.lower() in allowed}
    if not images or set(images) != set(masks):
        raise ValueError(f'Изображения и маски не совпадают: {len(images)} / {len(masks)}')
    if len(images) != 1000:
        raise ValueError(f'Для полного Kvasir-SEG ожидается 1000 пар, найдено {len(images)}')
    pairs = [(str(images[k]), str(masks[k]), k) for k in sorted(images)]
    train, holdout = train_test_split(pairs, test_size=.2, random_state=SEED, shuffle=True)
    val, test = train_test_split(holdout, test_size=.5, random_state=SEED, shuffle=True)
    train, val, test = [sorted(x, key=lambda y: y[2]) for x in (train, val, test)]
    assert not (set(x[2] for x in train) & set(x[2] for x in val))
    assert not (set(x[2] for x in train) & set(x[2] for x in test))
    manifest = [(item[2], split) for split, group in [('train', train), ('val', val), ('test', test)] for item in group]
    split_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    old = TABLES / 'split_fingerprint.json'
    if old.exists() and load_json(old)['sha256'] != split_hash:
        raise RuntimeError('Изменились исходные данные или разбиение. Продолжать обучение по прежним весам нельзя.')
    if not old.exists():
        with (TABLES / 'data_split.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f); writer.writerow(['filename','split']); writer.writerows(manifest)
        atomic_json({'sha256': split_hash, 'seed': SEED, 'size': {'train':len(train), 'val':len(val), 'test':len(test)}, 'images_dir':str(images_dir)}, old)
    return train, val, test


def read_pair(image_path, mask_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise RuntimeError(f'Не прочитан снимок или маска: {image_path}, {mask_path}')
    image = cv2.cvtColor(cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    mask = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)  # Никакого размытия классов.
    mask = (mask > 127).astype(np.uint8)                                   # Только 0 (фон) и 1 (полип).
    return image, mask


class PolypDataset(Dataset):
    def __init__(self, pairs, augment=False, imagenet=False):
        self.pairs, self.augment, self.imagenet = pairs, augment, imagenet

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image, mask = read_pair(self.pairs[index][0], self.pairs[index][1])
        if self.augment:
            if random.random() < .5:
                image, mask = np.fliplr(image), np.fliplr(mask)     # Одинаковое геометрическое преобразование.
            if random.random() < .5:
                image, mask = np.flipud(image), np.flipud(mask)
            if random.random() < .5:
                k = random.randint(1, 3)
                image, mask = np.rot90(image, k), np.rot90(mask, k)
            if random.random() < .30:
                image = np.clip(image.astype(np.float32)*random.uniform(.85,1.15)
                                + random.uniform(-18,18), 0, 255).astype(np.uint8) # Яркость только RGB.
        image = image.astype(np.float32) / 255.0
        if self.imagenet:
            image = (image - np.array([.485,.456,.406], dtype=np.float32)) / np.array([.229,.224,.225], dtype=np.float32)
        x = torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1)))
        y = torch.from_numpy(np.ascontiguousarray(mask[None].astype(np.float32)))
        return x, y


def data_loader(items, train=False, augment=False, imagenet=False, epoch=0):
    gen = torch.Generator().manual_seed(SEED+epoch*31)
    return DataLoader(PolypDataset(items, augment=train and augment, imagenet=imagenet),
                      batch_size=BATCH_SIZE, shuffle=train, num_workers=0, pin_memory=False, generator=gen)


class DoubleConv(nn.Module):
    """Две свёртки 3×3. Дополнительно: BatchNorm2d и Dropout2d."""
    def __init__(self, in_ch, out_ch, batch_norm=False, dropout=0, activation='relu', kernel=3):
        super().__init__()
        act = (nn.ReLU if activation=='relu' else lambda:nn.LeakyReLU(.1, inplace=True))
        layers = [nn.Conv2d(in_ch,out_ch,kernel,padding=kernel//2,bias=not batch_norm)]
        if batch_norm: layers.append(nn.BatchNorm2d(out_ch))
        layers.append(act())
        layers.append(nn.Conv2d(out_ch,out_ch,kernel,padding=kernel//2,bias=not batch_norm))
        if batch_norm: layers.append(nn.BatchNorm2d(out_ch))
        layers.append(act())
        if dropout: layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self,x): return self.block(x)


class Down(nn.Module):
    """Кодер: уменьшаем H,W в два раза, затем извлекаем признаки."""
    def __init__(self,in_ch,out_ch,pool='max',**kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(2) if pool=='max' else nn.AvgPool2d(2)
        self.conv = DoubleConv(in_ch,out_ch,**kwargs)

    def forward(self,x): return self.conv(self.pool(x))


class Up(nn.Module):
    """Декодер: увеличиваем разрешение и объединяем со skip из кодера."""
    def __init__(self,in_ch,skip_ch,out_ch,upsample='bilinear',**kwargs):
        super().__init__()
        self.upsample = upsample
        self.up = nn.ConvTranspose2d(in_ch,in_ch//2,2,stride=2) if upsample=='transpose' else None
        self.conv = DoubleConv((in_ch//2 if self.up else in_ch)+skip_ch,out_ch,**kwargs)

    def forward(self,x,skip):
        if self.up is None:
            x = F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
        else:
            x = self.up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x,size=skip.shape[-2:],mode='nearest')
        return self.conv(torch.cat([skip,x],dim=1))  # Skip-connections восстанавливают границы.


class UNet(nn.Module):
    """Собственная архитектура U-Net: четыре блока кодера + мост + четыре блока декодера."""
    def __init__(self,base_filters=16,batch_norm=False,dropout=0,upsample='bilinear',pool='max',activation='relu',kernel=3):
        super().__init__()
        b=base_filters
        kwargs=dict(batch_norm=batch_norm,dropout=dropout,activation=activation,kernel=kernel)
        self.inc=DoubleConv(3,b,**kwargs)
        self.down1=Down(b,2*b,pool=pool,**kwargs)
        self.down2=Down(2*b,4*b,pool=pool,**kwargs)
        self.down3=Down(4*b,8*b,pool=pool,**kwargs)
        self.down4=Down(8*b,16*b,pool=pool,**kwargs)
        self.up1=Up(16*b,8*b,8*b,upsample=upsample,**kwargs)
        self.up2=Up(8*b,4*b,4*b,upsample=upsample,**kwargs)
        self.up3=Up(4*b,2*b,2*b,upsample=upsample,**kwargs)
        self.up4=Up(2*b,b,b,upsample=upsample,**kwargs)
        self.out=nn.Conv2d(b,1,1)  # Логит вероятности полипа на каждом пикселе.

    def forward(self,x):
        a=self.inc(x)
        b=self.down1(a)
        c=self.down2(b)
        d=self.down3(c)
        e=self.down4(d)
        return self.out(self.up4(self.up3(self.up2(self.up1(e,d),c),b),a))


def create_model(cfg, init_pretrained=False):
    if cfg.kind=='custom':
        return UNet(base_filters=cfg.filters,batch_norm=cfg.batch_norm,dropout=cfg.dropout,
                    upsample=cfg.upsample,pool=cfg.pool,activation=cfg.activation,kernel=cfg.kernel), False
    try:
        import segmentation_models_pytorch as smp
    except ImportError as ex:
        raise RuntimeError('Нужна библиотека segmentation-models-pytorch. Включите Internet и выполните первую ячейку установки.') from ex
    cls = smp.FPN if cfg.kind=='fpn' else smp.DeepLabV3Plus
    kwargs=dict(encoder_name='resnet34',in_channels=3,classes=1,activation=None)
    if init_pretrained and PRETRAINED_ENCODER:
        try:
            return cls(**kwargs,encoder_weights='imagenet'), True
        except Exception as ex:
            raise RuntimeError('Не удалось загрузить ImageNet-веса ResNet34. Включите Internet или в конфигурации установите PRETRAINED_ENCODER=False. ') from ex
    return cls(**kwargs,encoder_weights=None), False


def pos_weight_value():
    cfg=load_json(TABLES/'run_settings.json')
    return float(cfg['positive_pixel_weight'])


def loss_function(logits,targets,loss_kind):
    weight=torch.as_tensor([pos_weight_value()],device=logits.device,dtype=logits.dtype)
    # Отрицательные пиксели имеют вес 1; положительные — рассчитанный только по train.
    bce=F.binary_cross_entropy_with_logits(logits,targets,pos_weight=weight)
    if loss_kind=='bce': return bce
    probs=logits.sigmoid()
    intersection=(probs*targets).sum((1,2,3))
    dice=(2*intersection+1e-6)/((probs+targets).sum((1,2,3))+1e-6)
    return .5*bce+.5*(1-dice.mean())


@torch.no_grad()
def scores(logits,targets,threshold=.5):
    predicted=logits.sigmoid()>=threshold
    truth=targets>=.5
    inter=(predicted&truth).sum((1,2,3)).float()
    pred=predicted.sum((1,2,3)).float()
    gt=truth.sum((1,2,3)).float()
    union=pred+gt-inter
    return ((inter+1e-7)/(union+1e-7)).cpu().numpy(), ((2*inter+1e-7)/(pred+gt+1e-7)).cpu().numpy()


def train_or_val_epoch(model,dataloader,cfg,optimizer=None):
    training=optimizer is not None
    model.train(training)
    total_loss=total_iou=total_dice=total_n=0.0
    with torch.set_grad_enabled(training):
        for x,y in dataloader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            if training:optimizer.zero_grad(set_to_none=True)
            logits=model(x)
            loss=loss_function(logits,y,cfg.loss)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Некорректная функция ошибки: {loss}')
            if training:
                loss.backward()              # Обратное распространение ошибки.
                optimizer.step()             # Обновление весов нейросети.
            iou,dice=scores(logits.detach(),y,threshold=.5)
            n=len(x)
            total_loss+=float(loss.item())*n
            total_iou+=float(iou.sum())
            total_dice+=float(dice.sum())
            total_n+=n
    return dict(loss=total_loss/total_n,iou=total_iou/total_n,dice=total_dice/total_n)


def model_paths(cfg):
    return (CKPT/f'{cfg.name}_last.pth',CKPT/f'{cfg.name}_best.pth',
            TABLES/f'{cfg.name}_history.csv',TABLES/f'{cfg.name}_info.json')


def ensure_training_setup():
    if DEVICE.type=='cpu' and os.environ.get('POLYP_ALLOW_CPU') != '1':
        raise RuntimeError('Для обучения включите GPU: Kaggle → Settings → Accelerator → GPU')


def prepare():
    train,val,test=find_pairs()
    settings_file=TABLES/'run_settings.json'
    settings=dict(seed=SEED,image_size=SIZE,epochs_unet=CUSTOM_EPOCHS,
                  epochs_ready=READY_EPOCHS,batch_size=BATCH_SIZE,pretrained=PRETRAINED_ENCODER,
                  split=load_json(TABLES/'split_fingerprint.json')['sha256'])
    if settings_file.exists():
        old=load_json(settings_file)
        for k,v in settings.items():
            if old.get(k)!=v:
                raise RuntimeError(f'Сохранённый запуск несовместим с настройкой {k}: было {old.get(k)}, стало {v}')
        pos_weight=old['positive_pixel_weight']
    else:
        positives=total=0
        for _,mask_path,_ in train:
            mask=cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
            if mask is None:raise FileNotFoundError(mask_path)
            mask=cv2.resize(mask,(SIZE,SIZE),interpolation=cv2.INTER_NEAREST)
            positives+=int((mask>127).sum())
            total+=mask.size
        if positives==0 or positives==total:
            raise RuntimeError('Все маски train одного класса: проверьте датасет')
        raw=(total-positives)/positives
        pos_weight=float(np.clip(raw, 2.0, 8.0))  # Ограничение не даёт огромному весу дестабилизировать обучение.
        settings.update(positive_pixel_fraction=positives/total, positive_pixel_weight=pos_weight)
        atomic_json(settings,settings_file)
    print(f'Подготовка данных: train={len(train)}, val={len(val)}, test={len(test)}',flush=True)
    print(f'Положительная доля пикселей train={load_json(settings_file)["positive_pixel_fraction"]:.3f}, '
          f'вес полипа в BCE={pos_weight:.3f}',flush=True)
    x,y=PolypDataset(train[:1])[0]
    print('Проверка формы:',tuple(x.shape),tuple(y.shape),'маска:',torch.unique(y).tolist(),flush=True)


def one_epoch(cfg,requested_epoch):
    ensure_training_setup()
    train,val,_=find_pairs()
    last,best,hist_path,info_path=model_paths(cfg)
    if info_path.exists() and load_json(info_path).get('completed') and best.exists():
        print(f'{cfg.name}: УЖЕ ОБУЧЕНА, повторно не запускаем',flush=True)
        return
    existing=torch.load(last,map_location='cpu',weights_only=True) if last.exists() else None
    completed=int(existing['epoch']) if existing else 0
    if completed>=requested_epoch:
        print(f'{cfg.name}: эпоха {requested_epoch} уже сохранена',flush=True)
        return
    if requested_epoch!=completed+1:
        raise RuntimeError(f'Нельзя пропускать эпохи: завершено {completed}, запрошено {requested_epoch}')
    random.seed(SEED+requested_epoch)
    np.random.seed(SEED+requested_epoch)
    torch.manual_seed(SEED)
    if DEVICE.type=='cuda':torch.cuda.manual_seed_all(SEED)
    model,pretrained=create_model(cfg,init_pretrained=not existing)
    model=model.to(DEVICE)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg.lr)
    scheduler=(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=3,min_lr=1e-6)
               if cfg.scheduler else None)
    best_iou=-1.0
    best_loss=float('inf')
    best_epoch=0
    bad_epochs=0
    if existing:
        if existing['config']!=asdict(cfg):
            raise RuntimeError(f'{cfg.name}: конфигурация изменена, нельзя продолжить старый checkpoint')
        model.load_state_dict(existing['model_state'])
        optimizer.load_state_dict(existing['optimizer_state'])
        if scheduler and existing.get('scheduler_state'):
            scheduler.load_state_dict(existing['scheduler_state'])
        best_iou=float(existing['best_iou'])
        best_loss=float(existing['best_loss'])
        best_epoch=int(existing['best_epoch'])
        bad_epochs=int(existing['bad_epochs'])
        pretrained=existing['pretrained_loaded']
    t0=time.monotonic()
    tr=train_or_val_epoch(model,data_loader(train,train=True,augment=cfg.augment,imagenet=cfg.kind!='custom',epoch=requested_epoch),cfg,optimizer)
    vl=train_or_val_epoch(model,data_loader(val,imagenet=cfg.kind!='custom'),cfg)
    if scheduler:scheduler.step(vl['iou'])
    # При одинаковом IoU используем val Loss как второй критерий: модель не застывает на 1-й эпохе.
    higher_iou=vl['iou']>best_iou+1e-5
    equal_iou=abs(vl['iou']-best_iou)<=1e-5
    lower_loss=vl['loss']<best_loss-1e-4
    improved=higher_iou or (equal_iou and lower_loss)
    if improved:
        best_iou,best_loss,best_epoch,bad_epochs=vl['iou'],vl['loss'],requested_epoch,0
        atomic_checkpoint({'model_state':model.state_dict(),'config':asdict(cfg),
                           'best_val_iou_05':best_iou,'best_val_loss':best_loss,
                           'best_epoch':best_epoch,'pretrained_loaded':pretrained},best)
    else:
        bad_epochs+=1
    row=dict(epoch=requested_epoch,train_loss=tr['loss'],train_iou=tr['iou'],train_dice=tr['dice'],
             val_loss=vl['loss'],val_iou_05=vl['iou'],val_dice_05=vl['dice'],
             lr=optimizer.param_groups[0]['lr'],best_epoch=best_epoch)
    # Храним всю историю внутри last checkpoint — падение между записью CSV и чекпоинта не теряет эпохи.
    history=existing.get('history',[]) if existing else []
    history=history+[row]
    payload=dict(config=asdict(cfg),epoch=requested_epoch,model_state=model.state_dict(),
                 optimizer_state=optimizer.state_dict(),scheduler_state=scheduler.state_dict() if scheduler else None,
                 best_iou=best_iou,best_loss=best_loss,best_epoch=best_epoch,bad_epochs=bad_epochs,
                 pretrained_loaded=pretrained,history=history)
    atomic_checkpoint(payload,last)
    with hist_path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(row))
        writer.writeheader();writer.writerows(history)
    # Первые 14 эпох базовой U-Net не останавливаем: сеть может долго учиться отличать полип от фона.
    min_epochs=14 if cfg.name=='E0_UNet_base' else 8
    complete=(requested_epoch>=cfg.epochs or (requested_epoch>=min_epochs and bad_epochs>=PATIENCE))
    info=dict(name=cfg.name,kind=cfg.kind,completed=complete,epochs_trained=requested_epoch,
              best_epoch=best_epoch,best_val_iou_05=best_iou,best_val_loss=best_loss,
              pretrained_loaded=pretrained,bad_epochs=bad_epochs,config=asdict(cfg))
    atomic_json(info,info_path)
    print(f'{cfg.name}  эпоха {requested_epoch:02d}/{cfg.epochs}  train Loss={tr["loss"]:.4f}  '
          f'val Loss={vl["loss"]:.4f}  train IoU={tr["iou"]:.4f}  val IoU={vl["iou"]:.4f}  '
          f'лучше={best_iou:.4f} (эпоха {best_epoch})  '
          f'{"готово" if complete else "сохранено"}  {time.monotonic()-t0:.1f} с',flush=True)


def load_best(cfg):
    _,best,_,info_path=model_paths(cfg)
    if not best.exists() or not info_path.exists() or not load_json(info_path)['completed']:
        raise RuntimeError(f'Нет завершённых весов модели {cfg.name}; сначала обучите её')
    model,_=create_model(cfg,init_pretrained=False)
    checkpoint=torch.load(best,map_location='cpu',weights_only=True)
    if checkpoint['config']!=asdict(cfg):raise RuntimeError(f'{cfg.name}: контрольная точка другого эксперимента')
    model.load_state_dict(checkpoint['model_state'])
    return model.to(DEVICE).eval()


THRESHOLD_GRID=[.20,.25,.30,.35,.40,.45,.50,.55,.60,.65,.70]


def calibrate(cfg):
    """Подбираем порог ТОЛЬКО по validation, ни один тестовый кадр не используется."""
    _,val,_=find_pairs()
    model=load_best(cfg)
    aggregate={t:[] for t in THRESHOLD_GRID}
    foreground_ratios=[]
    with torch.inference_mode():
        for x,y in data_loader(val,imagenet=cfg.kind!='custom'):
            logits=model(x.to(DEVICE))
            probs=logits.sigmoid()
            foreground_ratios.extend(probs.mean(dim=(1,2,3)).cpu().tolist())
            for t in THRESHOLD_GRID:
                ious,_=scores(logits,y.to(DEVICE),threshold=t)
                aggregate[t].extend(ious.tolist())
    rows=[{'threshold':t,'val_mean_iou':float(np.mean(aggregate[t]))} for t in THRESHOLD_GRID]
    result=TABLES/f'{cfg.name}_threshold_search.csv'
    with result.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # Если одинаковый IoU, предпочитаем порог, близкий к стандартному 0.5.
    winner=max(rows,key=lambda r:(r['val_mean_iou'],-abs(r['threshold']-.5)))
    info_path=model_paths(cfg)[3]
    info=load_json(info_path)
    info.update(threshold=winner['threshold'],val_iou_calibrated=winner['val_mean_iou'],
                mean_val_probability=float(np.mean(foreground_ratios)),val_iou_05=next(r['val_mean_iou'] for r in rows if r['threshold']==.5))
    atomic_json(info,info_path)
    print(f'{cfg.name}: val IoU при 0.50={info["val_iou_05"]:.4f}; '
          f'лучший порог={winner["threshold"]:.2f}, val IoU={winner["val_mean_iou"]:.4f}',flush=True)


def test_model(cfg):
    """Тест используется ОДИН РАЗ после обучения и настройки на validation."""
    _,_,test=find_pairs()
    model=load_best(cfg)
    threshold=float(load_json(model_paths(cfg)[3])['threshold'])
    records=[]
    indices=set(np.linspace(0,len(test)-1,num=5,dtype=int).tolist())
    sampled=[]
    offset=0
    with torch.inference_mode():
        for x,y in data_loader(test,imagenet=cfg.kind!='custom'):
            logits=model(x.to(DEVICE))
            ious,dices=scores(logits,y.to(DEVICE),threshold=threshold)
            ious_05,_=scores(logits,y.to(DEVICE),threshold=.5)
            masks=(logits.sigmoid()>=threshold).cpu().numpy().astype(np.uint8)
            for j in range(len(x)):
                idx=offset+j
                records.append(dict(filename=test[idx][2],IoU=float(ious[j]),Dice=float(dices[j]),IoU_fixed_05=float(ious_05[j])))
                if idx in indices:sampled.append((test[idx][2],masks[j,0]))
            offset+=len(x)
    result=TABLES/f'{cfg.name}_test_per_image.csv'
    with result.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    np.savez_compressed(PRED/f'{cfg.name}_five_masks.npz',names=np.array([x[0] for x in sampled]),
                        masks=np.stack([x[1] for x in sampled]))
    summary=dict(name=cfg.name,threshold=threshold,test_IoU=float(np.mean([r['IoU'] for r in records])),
                 test_Dice=float(np.mean([r['Dice'] for r in records])),
                 test_IoU_fixed_05=float(np.mean([r['IoU_fixed_05'] for r in records])),
                 count=len(records))
    atomic_json(summary,TABLES/f'{cfg.name}_test.json')
    print(f'{cfg.name}: TEST n={len(records)}, IoU={summary["test_IoU"]:.4f}, '
          f'Dice={summary["test_Dice"]:.4f}, IoU при 0.50={summary["test_IoU_fixed_05"]:.4f}',flush=True)


def predict(cfg,index):
    """Реальный forward одной сохранённой модели на выбранном тестовом снимке."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _,_,test=find_pairs()
    if not 0 <= index < len(test):raise IndexError(f'Индекс 0..{len(test)-1}')
    model=load_best(cfg)
    image_path,mask_path,image_id=test[index]
    image,truth=read_pair(image_path,mask_path)
    x=PolypDataset([test[index]],imagenet=cfg.kind!='custom')[0][0].unsqueeze(0).to(DEVICE)
    threshold=load_json(model_paths(cfg)[3])['threshold']
    with torch.inference_mode():
        pred=(model(x).sigmoid()[0,0]>=threshold).cpu().numpy().astype(np.uint8)
    intersection=int((pred.astype(bool)&truth.astype(bool)).sum())
    union=int((pred.astype(bool)|truth.astype(bool)).sum())
    iou=intersection/union if union>0 else 1.0
    dice=(2*intersection)/(int(pred.sum())+int(truth.sum())) if int(pred.sum())+int(truth.sum()) else 1.0
    overlay=image.copy()
    overlay[pred.astype(bool)]=(overlay[pred.astype(bool)]*.55+np.array([240,35,35])*.45).astype(np.uint8)
    fig,axes=plt.subplots(1,4,figsize=(16,4))
    for ax,data,label,cmap in zip(axes,[image,truth,pred,overlay],
          ['Исходное изображение','Эталонная маска','Предсказанная маска','Наложение'],[None,'gray','gray',None]):
        ax.imshow(data,cmap=cmap,vmin=0 if cmap else None,vmax=1 if cmap else None)
        ax.set_title(label);ax.axis('off')
    fig.suptitle(f'{cfg.name} · {image_id} · IoU {iou:.3f} · Dice {dice:.3f} · порог {threshold:.2f}')
    fig.tight_layout()
    path=FIGURES/'05_live_prediction.png'
    fig.savefig(path,dpi=160,bbox_inches='tight')
    plt.close(fig)
    print(f'Снимок {image_id}; test IoU={iou:.4f}, Dice={dice:.4f}; файл={path}',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','epoch','calibrate','test','predict'])
    parser.add_argument('--model',type=str)
    parser.add_argument('--epoch',type=int,default=1)
    parser.add_argument('--index',type=int,default=0)
    args=parser.parse_args()
    if args.command=='prepare':return prepare()
    cfg=config(args.model)
    if args.command=='epoch':return one_epoch(cfg,args.epoch)
    if args.command=='calibrate':return calibrate(cfg)
    if args.command=='test':return test_model(cfg)
    if args.command=='predict':return predict(cfg,args.index)


if __name__=='__main__':
    main()


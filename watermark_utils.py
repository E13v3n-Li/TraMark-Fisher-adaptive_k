from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import copy


def create_class_specific_datasets(dataset, num_train_per_class=None, num_clients=None):

    class_specific_datasets = {}
    class_indices = {label: [] for label in range(num_clients)}

    for idx, (_, label) in enumerate(dataset):
        # 水印源数据集（MNIST/SVHN等）本身可能有比 num_clients 更多的类别
        # （例如 MNIST 有10类，但 num_clients<10），这里只使用前 num_clients 个类别，
        # 其余类别的样本直接跳过，避免 KeyError
        if label in class_indices:
            class_indices[label].append(idx)

    for label in range(num_clients):
        if num_train_per_class is None:
            selected_indices = class_indices[label]
        else:
            selected_indices = class_indices[label][:num_train_per_class]

        class_specific_datasets[label] = Subset(dataset, selected_indices)

    return class_specific_datasets


def create_class_specific_loaders(train_dataset, test_dataset, num_train_per_class, batch_size=32, num_workers=2, num_clients=10):

    train_class_datasets = create_class_specific_datasets(train_dataset, num_train_per_class, num_clients=num_clients)
    test_class_datasets = create_class_specific_datasets(test_dataset, num_train_per_class=None, num_clients=num_clients)

    loaders = {
        label: {
            'train': DataLoader(train_class_datasets[label], batch_size=batch_size, shuffle=True, num_workers=num_workers),
            'test': DataLoader(test_class_datasets[label], batch_size=batch_size, shuffle=False, num_workers=num_workers)
        }
        for label in range(num_clients)
    }

    return loaders


def get_watermark_dataset(args):

    # Get the appropriate transformations for the mnist dataset based on the main dataset
    if args.dataset == 'tiny':
        transform_mnist = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ])
    elif args.dataset == 'fmnist':
        transform_mnist = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
        ])
    elif args.dataset == 'cifar10':
        transform_mnist = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
    elif args.dataset == 'cifar100':
        transform_mnist = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ])


    train_mnist = torchvision.datasets.MNIST(root='../../data', train=True, download=True, transform=transform_mnist)
    test_mnist = torchvision.datasets.MNIST(root='../../data', train=False, download=True, transform=transform_mnist)

    class_loaders = create_class_specific_loaders(train_mnist,
                                                  test_mnist,
                                                  num_train_per_class=100,
                                                  batch_size=32,
                                                  num_workers=args.num_workers,
                                                  num_clients=args.num_clients)


    return class_loaders


def generate_masks_topk(partition_ratio, model):
    """原版TraMark的重要性度量：直接用参数绝对值排序。
    ['方案设计文档'里的 TraMark-Mag baseline，就是这个函数。]"""

    partition_ratio = 1 - partition_ratio
    keys = [name for name in model.state_dict()]
    keys_parameters = [name for name, param in model.named_parameters()]

    param_dict = {name: param for name, param in model.named_parameters()}
    main_task_mask = {}
    watermarking_mask = {}

    for name in keys:
        if name in keys_parameters:
            param = param_dict[name]
            numel = param.numel()
            mask = torch.zeros(numel, dtype=torch.bool)

            flat_param = param.view(-1).abs()
            topk = int(numel * partition_ratio)
            if topk > 0:
                _, selected_indices = torch.topk(flat_param, topk, largest=True)
                mask[selected_indices] = True

        else:
            param = model.state_dict()[name]
            numel = param.numel()
            mask = torch.ones(numel, dtype=torch.bool)

        main_task_mask[name] = mask.view_as(param)
        watermarking_mask[name] = ~mask.view_as(param)

    return main_task_mask, watermarking_mask


def generate_masks_gradient_based(partition_ratio, model, dataloader, device, method='fisher', num_batches=5):
    """用基于梯度的重要性得分衡量每个参数对主任务损失的真实影响，
    替代 generate_masks_topk 里"参数绝对值越大越重要"这个不够准确的假设。

    对应"阶段A"里的两个强baseline：
      - method='fisher'：Fisher信息（梯度平方均值），参考EWC（Kirkpatrick et al., PNAS 2017）
      - method='grad'  ：梯度绝对值均值 E|g|，比Fisher更简单的一版
      - method='taylor'：一阶泰勒展开重要性 |梯度 x 权重|，参考Molchanov et al.（CVPR 2019）

    重要性最低的 partition_ratio 比例参数被划为水印区，其余划为主任务区，
    跟 generate_masks_topk 的输出格式完全一致，可以直接替换使用。

    Args:
        partition_ratio: 水印区占比（跟 generate_masks_topk 语义一致）
        model: 预热阶段结束时的全局模型
        dataloader: 用来估计重要性的主任务数据（服务器侧的公开测试集即可，几个batch足够，
                    不需要访问任何客户端私有数据，符合TraMark原本的威胁模型）
        device: 计算设备
        method: 'fisher'（默认，推荐） / 'grad' / 'taylor'
        num_batches: 用多少个batch来累积梯度，默认5个，越多估计越稳但越慢
    """
    assert method in ('fisher', 'grad', 'taylor'), "method 必须是 'fisher'、'grad' 或 'taylor'"

    model = copy.deepcopy(model).to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    importance_scores = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
    batch_count = 0
    for inputs, targets in dataloader:
        if batch_count >= num_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        model.zero_grad()
        outputs = model(inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        loss = criterion(logits, targets)
        loss.backward()
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                if method == 'fisher':
                    importance_scores[name] += p.grad.detach() ** 2
                elif method == 'grad':
                    importance_scores[name] += p.grad.detach().abs()
                else:  # taylor
                    importance_scores[name] += (p.grad.detach() * p.detach()).abs()
        batch_count += 1
    for name in importance_scores:
        importance_scores[name] /= max(batch_count, 1)

    partition_ratio = 1 - partition_ratio
    keys = [name for name in model.state_dict()]
    keys_parameters = [name for name, param in model.named_parameters()]
    param_dict = {name: param for name, param in model.named_parameters()}
    main_task_mask, watermarking_mask = {}, {}

    for name in keys:
        if name in keys_parameters:
            param = param_dict[name]
            numel = param.numel()
            mask = torch.zeros(numel, dtype=torch.bool)

            flat_score = importance_scores[name].view(-1)  # 得分越高越重要
            topk = int(numel * partition_ratio)
            if topk > 0:
                _, selected_indices = torch.topk(flat_score, topk, largest=True)
                mask[selected_indices] = True
        else:
            param = model.state_dict()[name]
            numel = param.numel()
            mask = torch.ones(numel, dtype=torch.bool)

        main_task_mask[name] = mask.view_as(param)
        watermarking_mask[name] = ~mask.view_as(param)

    return main_task_mask, watermarking_mask


def masks_to_vector(masks):

    vectors = []
    for name, mask in masks.items():
        vectors.append(mask.flatten())

    return torch.cat(vectors)


def merge_watermark_masks(old_watermarking_mask, candidate_watermarking_mask):
    """Adaptive-k 专用：watermark mask 只能扩张，不能释放已经学好的watermark参数。

    正确逻辑是取并集，而不是用candidate直接替换old：
        M_wm^new = M_wm^old ∪ M_wm^candidate

    这样当 k 从 1.0% 增加到 1.2% 时，只会新增约 0.2% 的watermark参数，
    已经属于watermark区域的参数会继续保留，避免已经学好的client-specific
    watermark被"踢出"watermark区而遗忘。

    Args:
        old_watermarking_mask: 当前正在使用的 watermarking_mask（dict: name -> bool tensor）
        candidate_watermarking_mask: 按新的k重新估计重要性后得到的候选watermarking_mask

    Returns:
        (merged_main_task_mask, merged_watermarking_mask)
    """
    merged_wm = {
        name: old_watermarking_mask[name] | candidate_watermarking_mask[name]
        for name in old_watermarking_mask
    }
    merged_main = {name: ~merged_wm[name] for name in merged_wm}
    return merged_main, merged_wm

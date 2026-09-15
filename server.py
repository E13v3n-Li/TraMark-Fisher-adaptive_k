from torch.nn.utils import parameters_to_vector
import logging
from utils import vector_to_model
from watermark_utils import get_watermark_dataset, generate_masks_topk, generate_masks_gradient_based, masks_to_vector, merge_watermark_masks
from adaptive_k import AdaptiveKConfig, AdaptiveKController
import torch.nn as nn
import torch.optim as optim
import torch
import copy
import builtins


class Server():
    def __init__(self, client_data_sizes, args, main_task_test_loader):
        self.client_data_sizes = client_data_sizes
        self.args = args
        self.main_task_test_loader = main_task_test_loader
        self.best_acc = -1
        self.best_veri = -1

        # Adaptive-k: 默认关闭，行为与原TraMark完全一致；开启后才会创建controller
        self.adaptive_k_enabled = getattr(args, 'adaptive_k', False)
        self.k_controller = None
        if self.adaptive_k_enabled:
            warmup_rounds = int(args.training_rounds * args.alpha)
            self.k_controller = AdaptiveKController(
                config=AdaptiveKConfig(
                    k_init=args.k,
                    k_min=args.k_min,
                    k_max=args.k_max,
                    k_step=args.k_step,
                    target_vr=args.target_vr,
                    target_acc=args.target_acc,
                    vr_tolerance=args.vr_tolerance,
                    acc_tolerance=args.acc_tolerance,
                    ema_beta=args.k_ema_beta,
                    eval_interval_ratio=args.k_eval_interval_ratio,
                    increase_only=True,
                ),
                training_rounds=args.training_rounds,
                warmup_rounds=warmup_rounds,
                log_dir=getattr(args, 'log_dir', None),
            )


    def aggregation(self, client_model_dict, round):
        
        if self.args.watermarking_method == 'fedavg':    
            aggregated_model_dict = self.fedavg(client_model_dict, round)

        if self.args.watermarking_method == 'tramark':
            # If in the warmup stage, we simply perform FedAvg to get a relatively good initial global model for mask generation. 
            if round < int(self.args.training_rounds * self.args.alpha):
                return self.fedavg(client_model_dict, round)
            
            # Right after the warmup stage, we:
            # 1. initialize the masks based on the current global model. 
            # 2. get backdoor dataset for each client.
            if round == int(self.args.training_rounds * self.args.alpha):
                current_global_model = copy.deepcopy(client_model_dict[0])
                self.tramark_initialization(current_global_model)
            
            aggregated_model_dict = self.agg_tramark(client_model_dict, round)
            
        return aggregated_model_dict



    def fedavg(self, client_model_dict, round=None):
        """ classic fed avg: Weighted average based on data size."""

        sm_updates, total_data = 0, 0
        for client_id, local_model in client_model_dict.items():
            local_vector = parameters_to_vector(
                    [local_model.state_dict()[name] for name in local_model.state_dict()]
                ).detach()
            n_agent_data = self.client_data_sizes[client_id]
            sm_updates += local_vector * n_agent_data  
            total_data += n_agent_data

        sm_updates /= total_data
        
        old_model = copy.deepcopy(local_model)
        new_model = vector_to_model(sm_updates, old_model)  

        aggregated_models_dict = {}
        for client_id, local_model in client_model_dict.items():
            aggregated_models_dict[client_id] = copy.deepcopy(new_model)
                    
        # Evaluate the aggregated global model on the main task test set
        accuracy = self.test_function(new_model, self.main_task_test_loader)
        logging.info(f"Main task accuracy of the aggregated global model at round {round}: {accuracy:.2f}%")

        if self.best_acc < accuracy:
            self.best_acc = accuracy
        return aggregated_models_dict


    def agg_tramark(self, client_model_dict, round):

        # Masked aggregation: we only aggregate the parameters in the main task region and keep the parameters in the watermarking task region unchanged for each client.
        aggregated_vector = torch.zeros_like(self.main_task_mask_flat, dtype=torch.float32).cuda()
        total_data = 0
        for client_id, model in client_model_dict.items():
            local_vector = parameters_to_vector(
                    [model.state_dict()[name] for name in model.state_dict()]
                ).detach()
            n_agent_data = self.client_data_sizes[client_id]
            aggregated_vector += local_vector * n_agent_data  
            total_data += n_agent_data

        aggregated_vector /= total_data
        aggregated_models_dict = {}

        for client_id, model in client_model_dict.items():
            local_vector = parameters_to_vector(
                        [model.state_dict()[name] for name in model.state_dict()]
                    ).detach()

            ########### Key: Masked aggregation ###########
            mask_aggregated_vector = self.main_task_mask_flat * aggregated_vector + self.watermarking_mask_flat * local_vector

            old_model = copy.deepcopy(model)
            new_model = vector_to_model(mask_aggregated_vector, old_model)
            aggregated_models_dict[client_id] = new_model


        ########### Watermark Injection ###########
        print('Watermarking process starts')
        main_task_acc = []
        for id, local_model in aggregated_models_dict.items():
            pre_inject_acc = self.test_function(local_model, self.main_task_test_loader)
    
            optimizer = optim.SGD(local_model.parameters(), lr=self.watermarking_lr, momentum=self.watermarking_momentum)

            local_model.train()
            for epoch in range(self.watermarking_epochs):
                
                for inputs, targets in self.class_loaders[id]['train']:
                    
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                    outputs = local_model(inputs)
                    
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    
                    loss = self.criterion(logits, targets)
                    loss.backward()

                    ########### Zeroing out gradients in the main task region ###########
                    for name, param in dict(local_model.named_parameters()).items():
                        if name in self.watermarking_mask:
                            if param.grad is not None:
                                param.grad.data.mul_(self.watermarking_mask[name].cuda())

                    optimizer.step()

                print(f"Client {id}, watermarking epoch {epoch + 1}/5 finished.")
            post_injection_acc = self.test_function(local_model, self.main_task_test_loader)
            logging.info(f"Client %d, accuracy changes before and after injection: %.2f%% --> %.2f%%" % (id, pre_inject_acc, post_injection_acc))
            main_task_acc.append(post_injection_acc)


        logging.info('-----'*4)

        verification = []
        current_acc = sum(main_task_acc) / len(main_task_acc)
        logging.info('--------Model leaker verification--------')

        for id, local_model in aggregated_models_dict.items():
            local_model.eval()

            current_acc_trigger = []
            for i in range(self.args.num_clients):
                acc = self.test_function(local_model, self.class_loaders[i]['test'])
                current_acc_trigger.append(builtins.round(acc, 2))

            max_index = current_acc_trigger.index(max(current_acc_trigger))

            if max_index == id:
                verification.append(1)
                logging.info(f"Client id: {id}, accuracy on trigger set are {current_acc_trigger}, verification success!")
            else:
                verification.append(0)
                logging.info(f"Client id: {id}, accuracy on trigger set are {current_acc_trigger}, verification fail!")

        current_veri = sum(verification) / len(verification) * 100
        logging.info(f'Round {round} Avg VR: {current_veri:.2f}%')
        logging.info(f'Round {round} Avg ACC: {current_acc:.2f}%')
        if self.best_acc < current_acc:
            self.best_acc = current_acc
        if self.best_veri < current_veri:
            self.best_veri = current_veri

        # Adaptive-k: 每 eval_interval 轮，根据这一轮已经算出来的 VR / main_acc 决定要不要增大k。
        # 不额外做evaluation forward，直接复用上面循环里算出的current_veri/current_acc。
        if self.adaptive_k_enabled and self.k_controller.should_update(round):
            old_k = self.k_controller.k
            new_k, info = self.k_controller.decide(
                round_id=round, vr=current_veri, main_acc=current_acc
            )
            if new_k > old_k:
                # 用当前这一轮聚合后的主任务backbone（aggregated_vector，即完全FedAvg平均后的
                # 参数）重新估计重要性，得到candidate mask，再和旧mask取并集（只扩张不释放）。
                actual_k = self._grow_watermark_region(
                    new_k=new_k,
                    aggregated_vector=aggregated_vector,
                    template_model=model,
                )
            else:
                actual_k = self.watermarking_mask_flat.float().mean().item()
            self.k_controller.record(info, actual_k)

        return aggregated_models_dict


    def _grow_watermark_region(self, new_k, aggregated_vector, template_model):
        """Adaptive-k: 按new_k重新估计参数重要性，得到候选watermark mask，
        与当前watermark mask取并集（只扩张，不释放已学好的watermark参数），
        然后刷新dict形式的mask以及aggregation实际使用的展平CUDA向量。

        Args:
            new_k: controller决定的新的watermark ratio（请求值）
            aggregated_vector: 本轮完全FedAvg平均后的参数向量，代表当前共享的
                主任务backbone，用它重新估计重要性比用某个client的（含其专属
                watermark参数）模型更接近warmup阶段初始化mask时的做法。
            template_model: 提供state_dict结构用的模板模型（随便取一个client的即可）

        Returns:
            actual_k: 实际的watermark参数占比（因为mask是并集，通常会略高于new_k）
        """
        importance_method = getattr(self.args, 'importance_method', 'topk')

        current_global_model = copy.deepcopy(template_model)
        current_global_model = vector_to_model(aggregated_vector.clone(), current_global_model)

        if importance_method == 'topk':
            _, candidate_wm_mask = generate_masks_topk(
                partition_ratio=new_k, model=current_global_model
            )
        else:
            _, candidate_wm_mask = generate_masks_gradient_based(
                partition_ratio=new_k, model=current_global_model,
                dataloader=self.main_task_test_loader, device=self.args.device,
                method=importance_method,
                num_batches=getattr(self.args, 'importance_num_batches', 5)
            )

        self.main_task_mask, self.watermarking_mask = merge_watermark_masks(
            old_watermarking_mask=self.watermarking_mask,
            candidate_watermarking_mask=candidate_wm_mask,
        )

        # 原TraMark的aggregation直接用这两个展平向量做masked aggregation，
        # mask扩张后必须同步刷新，否则agg_tramark用的还是旧mask。
        self.main_task_mask_flat = masks_to_vector(self.main_task_mask).cuda()
        self.watermarking_mask_flat = masks_to_vector(self.watermarking_mask).cuda()

        actual_k = self.watermarking_mask_flat.float().mean().item()
        logging.info(
            "[Adaptive-k] mask grown: requested k=%.4f, actual k=%.4f (watermark_params=%d / total_params=%d)"
            % (new_k, actual_k, int(self.watermarking_mask_flat.float().sum().item()),
               self.watermarking_mask_flat.numel())
        )
        return actual_k


    def tramark_initialization(self, current_model):
        # Set up the watermarking hyperparameters
        self.watermarking_epochs = 5
        self.watermarking_lr = 0.0001
        self.watermarking_momentum = 0.0
        
        self.class_loaders = get_watermark_dataset(self.args)

        # 参数重要性度量方式："阶段A"消融实验的接入点
        # topk（默认，原版TraMark）: 参数绝对值排序
        # grad/fisher/taylor: 用服务器侧的主任务测试集(main_task_test_loader)估计梯度类重要性，
        #                      不涉及任何客户端私有数据，符合TraMark原本的威胁模型
        importance_method = getattr(self.args, 'importance_method', 'topk')

        # Adaptive-k: 启用时用controller当前的k（初始值等于args.k），否则和原逻辑一样用args.k
        current_k = self.k_controller.k if self.adaptive_k_enabled else self.args.k

        if importance_method == 'topk':
            self.main_task_mask, self.watermarking_mask = generate_masks_topk(
                partition_ratio=current_k, model=current_model
            )
        else:
            self.main_task_mask, self.watermarking_mask = generate_masks_gradient_based(
                partition_ratio=current_k, model=current_model,
                dataloader=self.main_task_test_loader, device=self.args.device,
                method=importance_method,
                num_batches=getattr(self.args, 'importance_num_batches', 5)
            )
        logging.info(f'Watermark region selected using importance_method={importance_method}, k={current_k:.4f}')

        self.criterion = nn.CrossEntropyLoss()
        self.main_task_mask_flat = masks_to_vector(self.main_task_mask).cuda()
        self.watermarking_mask_flat = masks_to_vector(self.watermarking_mask).cuda()


    def test_function(self, model, dataloader):
        """Test the model with the given dataloader."""
        model.eval()
        
        total, correct = 0, 0
        with torch.no_grad():
        
            for inputs, targets in dataloader:
                inputs = inputs.cuda()
                targets = targets.cuda()  

                outputs = model(inputs)
                
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                
                _, predicted = logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        accuracy = 100. * correct / total

        return accuracy
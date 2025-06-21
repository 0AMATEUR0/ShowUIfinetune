import os
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import wandb
from functools import partial
from transformers import BitsAndBytesConfig, AutoProcessor, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from model.utils import find_target_linear_names
from main.trainer import train
from main.eval_aitw import validate_aitw
from main.eval_mind2web import validate_mind2web
from main.eval_screenspot import validate_screenspot
from main.evaluator import validate as validate_default
from data.dataset import HybridDataset, collate_fn
from utils.utils import save_args_to_json, create_log_dir


args = {
    # wandb配置参数
    "wandb_key": "7dd9f7e0d1d48f0b0296d469ce4b6365e615094d", # 修改为你的wandb API key

    # 需要修改的路径参数
    "model_path": "D:/Project/showui-2b", # 修改为你的基模型路径
    "train_dataset": "screenspot", # 修改为你的训练数据集路径
    "train_json": "metadata", # 修改为你的训练数据集标注文件名
    "val_dataset": "screenspot", # 修改为你的验证数据集路径
    "val_json": "metadata", # 修改为你的验证数据集标注文件名
    "dataset_dir": "D:/Project/my_dataset", # 修改为你的数据集目录路径
    "exp_dir": "D:/Project/logs/debug/2025-06-20_10-10-20", # 请修改为你的LoRA权重保存路径

    # 模型配置参数
    "model_id": "local_ShowUI-2B", # 模型ID
    "version": "showlab/ShowUI-2B", # 模型版本路径
    "min_visual_tokens": 256, # 最小视觉token数量
    "max_visual_tokens": 1280, # 最大视觉token数量
    "model_max_length": 8192, # 模型最大长度，8192表示支持长文本输入

    # ui图配置参数
    "uigraph_train": True, # Enable ui graph during training
    "uigraph_test": False, # Enable ui graph during inference
    "uigraph_diff": 1, # Pixel difference used for constructing ui graph
    "uigraph_rand": False, # Enable random graph construction
    "uimask_pre": True, # Prebuild patch selection mask in the preprocessor (not in model layers) for efficiency
    "uimask_ratio": 0.5, # Specify the percentage of patch tokens to skip per component
    "uimask_rand": False, # Enable random token selection instead of uniform selection
    
    "precision": "bf16", # precision for inference, options: "fp16", "bf16", "fp32"
    "use_qlora": False, # Use QLoRA for training
    
    # 语言和视觉层跳过参数
    "lm_skip_ratio": 0.5, # Ratio of language tokens to skip, e.g., 0.5 means skip 50% of language tokens
    "lm_skip_layer": '[1,28,0]', # Skip layers for language tokens, e.g., [1,28,0] means skip layer 1 and 28
    "vis_skip_layer": '[1,32,0]', # Skip layers for visual tokens, e.g., [1,32,0] means skip layer 1 and 32
    "attn_imple": "sdpa", # Attention implementation, options: "eager", "flash_attention_2", "sdpa"
    
    # LoRA微调配置参数
    "use_qlora": False, # Whether to use QLoRA for training
    "lora_r": 16, # Rank for LoRA
    "lora_alpha": 16, # Alpha for LoRA
    "lora_dropout": 0.05, # Dropout for LoRA
    "lora_target_modules": "qkv_proj", # Target modules for LoRA, e.g., "q_proj,v_proj,k_proj,o_proj"
    "tune_visual_encoder": False, # Whether to tune the visual encoder
    "freeze_lm_embed": False, # Whether to freeze the language model embedding

    # 梯度检查点配置参数
    "gradient_checkpointing": False, # Enable gradient checkpointing to reduce memory usage
    "tune_visual_encoder_projector": False, # Whether to tune the visual encoder projector

    # 数据集配置参数
    "train_ratio": "1.0", # Ratio of training data to use, can be a float between 0 and 1
    "val_ratio": "1.0", # Ratio of validation data to use, can be a float between 0 and 1
    "uniform_sample": False, # Whether to use uniform sampling for training data
    "random_sample": False, # Whether to use random sampling for training data
    "record_sample": False, # Whether to record the sampled data for debugging

    # 训练配置参数
    "log_base_dir": "D:/Project/logs", # Base directory for logs
    "exp_id": "debug", # Experiment ID for logging
    "lr": 5e-5, # Learning rate for training
    "beta1": 0.9, # Beta1 for Adam optimizer
    "beta2": 0.999, # Beta2 for Adam optimizer
    "epochs": 10, # Number of epochs for training
    "steps_per_epoch": 33, # Steps per epoch for training
    "warmup_steps": 100, # Warmup steps for learning rate scheduler
    "batch_size": 1, # Batch size for training
    "grad_accumulation_steps": 1, # Gradient accumulation steps
    "val_batch_size": 1, # Batch size for validation
    "workers": 8, # Number of workers for data loading

    # Grounding setting
    "num_turn": 100, # Interleaved Query-Action setting
    "shuffle_image_token": False, # shuffle image token for training
    "uniform_prompt": True, # Use uniform prompt for training
    "text2point": 1.0, # Text to point ratio for training
    "text2bbox": 0.0, # Text to bbox ratio for training
    "point2text": 0.0, # Point to text ratio for training
    "bbox2text": 0.0, # Bbox to text ratio for training
    "crop_min": 1.0 , # Minimum crop ratio for training
    "crop_max": 1.0, # Maximum crop ratio for training
    "xy_int": False, # Whether to use integer coordinates for x and y in grounding

    # Navigation setting
    "num_history": 4, # Number of history steps for navigation
    "interleaved_history": 'tttt', # Interleaved Vision-Action setting,choices=['tttt', 'vvvv', 'vtvt', 'tvtv', 'vvtt', 'ttvv']
    "skip_readme_train": False, # Whether to skip README training data
    "skip_readme_test": False, # Whether to skip README test data

    # 模型检测点和评估配置参数
    "eval_only": False, # Whether to only run evaluation
    "start_epoch": 0, # Start epoch for training
    "no_eval": False, # Whether to skip evaluation during training
    "debug": False, # for debugging, will not save model and monitor
    "print_freq": 1, # Frequency of printing training progress


    
}

def main(args):

    args.global_rank = int(os.environ.get("RANK", 0))
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    

    if args.attn_imple in ["eager", "sdpa"]:
        # suggested by https://github.com/Lightning-AI/litgpt/issues/327
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    args.distributed = args.world_size > 1


    args.log_dir = os.path.join(args.log_base_dir, args.exp_id, timestamp)
    args.tmp_dir = os.path.join(args.log_dir, "tmp")

    # must provide wandb-key
    # assert args.wandb_key is not None
    # wandb.login(key=args.wandb_key)


    writer = None  # TensorBoard writer, if needed, can be initialized later
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)
    save_args_to_json(args, os.path.join(args.log_dir, "args.json"))  # 保存参数
    if not args.debug:
        # 创建TensorBoard日志目录
        writer = SummaryWriter(os.path.join(args.log_dir, "tensorboard"))
        # 初始化wandb
        # wandb.init(
        #     project="ShowUI",
        #     group=args.exp_id,
        #     name=f'{args.exp_id}_{timestamp}',
        #     config=args,
        #     dir=args.log_dir,
        # )
    print(f"Start Job: {args.exp_id}")

    # 创建处理器

    from model.showui.processing_showui import ShowUIProcessor

    processor = ShowUIProcessor.from_pretrained(args.model_path,
                                                min_pixels=args.min_visual_tokens *28*28,
                                                max_pixels=args.max_visual_tokens *28*28,
                                                model_max_length=args.model_max_length,
                                                uigraph_train=args.uigraph_train, uigraph_test=args.uigraph_test,
                                                uigraph_diff=args.uigraph_diff,  uigraph_rand=args.uigraph_rand,
                                                uimask_pre=args.uimask_pre, uimask_ratio=args.uimask_ratio, uimask_rand=args.uimask_rand,
                                                size = {"shortest_edge": 3136, "longest_edge": 1003520}
                                              )
    
    # 创建模型
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    model_path = args.model_path
    
    bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["img_projection"],
                ) if args.use_qlora else None # 仅在使用QLoRA时才需要配置
    
    from model.utils import parse_layer_type
    from model.showui.modeling_showui import ShowUIForConditionalGeneration

    lm_qwen_layer = 28
    vis_qwen_layer = 32
    lm_skip_layer = parse_layer_type(args.lm_skip_layer, lm_qwen_layer)
    vis_skip_layer = parse_layer_type(args.vis_skip_layer, vis_qwen_layer)

    model = ShowUIForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype, # 模型精度
        low_cpu_mem_usage=True, # 低内存使用模式
        _attn_implementation=args.attn_imple, # 注意力实现方式
        # quantization_config=bnb_config, # 量化配置
        device_map="cuda", # 自动设备映射
        lm_skip_layer=lm_skip_layer, # 跳过语言层
        lm_skip_ratio=args.lm_skip_ratio, # 跳过语言层比例
        tie_word_embeddings=False, # 是否共享词嵌入
    )
    # 手动同步权重
    model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    # 保存 untied 模型
    model.save_pretrained("D:/Project/MODELS")
    model.config.save_pretrained("D:/Project/MODELS")

    # 加载模型检测点
    # if args.version != args.model_id:
    #     state_dict = torch.load(args.version, map_location="cpu")
    #     model.load_state_dict(state_dict, strict=False)

    model.config.use_cache = False # 禁用缓存以节省内存

    # 在评估模式下，不需要加载LoRA
    if args.eval_only:
        print("evaluation mode, thus set the `lora_r' as zero.")
        args.lora_r = 0
    if not args.eval_only and args.use_qlora:
        model = prepare_model_for_kbit_training(model)

    # 配置LoRA
    lora_r = args.lora_r
    if lora_r > 0:
        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        exclude_module = ["visual"] if not args.tune_visual_encoder else []
        exclude_module += ["lm_head"] if args.freeze_lm_embed else exclude_module
        lora_target_modules = find_target_linear_names(model, lora_namespan_exclude=exclude_module)

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        # model.print_trainable_parameters()

        # 如果使用LoRA，则原始模型被包装2次
        # 一次是peft的get_peft_model包装，一次是ShowUIForConditionalGeneration的包装
        model_child = model.model.model # 获取原始模型，疑似不可使用base_model方法
    else:
        # 如果不使用LoRA，则原始模型只被ShowUIForConditionalGeneration包装
        model_child = model.model
    
    # 梯度检查点，降低显存使用
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    
    if not args.tune_visual_encoder:
        # 冻结视觉编码器
        if args.lora_r > 0:
            for p in model.base_model.model.visual.parameters():
                p.requires_grad = False
        elif args.lora_r == 0:
            for p in model.visual.parameters():
                p.requires_grad = False
        
    if args.tune_visual_encoder_projector:
        for k, p in model.named_parameters():
            if 'visual.merger' in k:
                p.requires_grad = True
    
    if args.freeze_lm_embed:
        if args.lora_r > 0:
            for p in model_child.embed_tokens.parameters():
                p.requires_grad = False
        elif args.lora_r == 0:
            for p in model_child.embed_tokens.parameters():
                p.requires_grad = False
    
    # 检查可训练参数
    list_of_params_to_optimize = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            # print("[Name]", n, " [Shape]", p.shape)
            list_of_params_to_optimize.append(p)
    
    # 创建数据集
    args.samples_per_epoch = args.batch_size    \
                    * args.grad_accumulation_steps  \
                    * args.steps_per_epoch

    train_dataset = HybridDataset(
        processor,
        inference=False,  # 仅用于训练
        args=args,
    )
    
    val_dataset = HybridDataset(
        processor,
        inference=True,  # 仅用于验证
        args=args,
    )

    if args.val_dataset == "mind2web":
        validate = validate_mind2web
    elif args.val_dataset == "screenspot":
        validate = validate_screenspot
    elif args.val_dataset == "aitw":
        validate = validate_aitw
    else:
        validate = validate_default

    if not args.random_sample:
        args.steps_per_epoch = len(train_dataset) // (args.batch_size * args.world_size)

    # deepspeed参数（待完成）
    # 如果使用DeepSpeed，参考https://github.com/showlab/ShowUI/blob/main/train.py

    # LoRA微调
    if lora_r > 0:

        # 创建优化器
        optimizer = torch.optim.AdamW(
            list_of_params_to_optimize,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=0.0,
            )

        # DeepSpeed 用的是 WarmupDecayLR，PyTorch 没有内置这个，但可以用类似的调度器
        total_steps = args.epochs * args.steps_per_epoch
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps,
        )

        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=partial(collate_fn, processor=processor),
            num_workers=args.workers,  # 根据你的CPU核心数调整
        )
        

        # 模型引擎
        model_engine = model
        model_engine = model_engine.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        

    # 如果不使用LoRA微调
    # 暂时一样，但是方便后续扩展
    elif lora_r == 0 and not args.eval_only:
        # 创建优化器
        optimizer = torch.optim.AdamW(
            list_of_params_to_optimize,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=0.0,
            )

        # DeepSpeed 用的是 WarmupDecayLR，PyTorch 没有内置这个，但可以用类似的调度器
        total_steps = args.epochs * args.steps_per_epoch
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps,
        )

        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=partial(collate_fn, processor=processor),
            num_workers=args.workers,  # 根据你的CPU核心数调整
        )

        # 模型引擎
        model_engine = model
        model_engine = model_engine.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # 仅评估模式
    elif args.eval_only:
        for param in model.parameters():
            param.requires_grad = False 
        model_engine = model
    else:
        raise ValueError("Invalid setting")
    

    # 断点加载（待完成）

    # 验证集
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler= None,  # 若分布式训练，此处参考https://github.com/showlab/ShowUI/blob/main/train.py
            collate_fn=partial(collate_fn, processor=processor)
        )
    else:
        val_loader = None
    
    if args.eval_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_engine = model_engine.to(device)
        validate(val_loader, model_engine, processor, 0, 0, writer, args)
        exit()

    train_iter = iter(train_loader)
    best_score = 0.0
    # args.start_epoch 是为了支持断点恢复训练
    print("Start training")
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_iter, global_step = train(
            train_loader,
            model_engine,
            optimizer,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )

        if args.no_eval == False and val_loader is not None:
            score = validate(
                val_loader,
                model_engine,
                processor,
                epoch,
                global_step,
                writer,
                args,
            )
            is_best = score > best_score
            best_score = max(score, best_score)
        else:
            is_best = True
            score = 0.0
        
        if args.no_eval or is_best:
            save_dir = os.path.join(args.log_dir,"ckpt_model")
            
            os.makedirs(save_dir, exist_ok=True)
            torch.save(
                {"epoch": epoch},
                os.path.join(
                    save_dir,
                    "meta_log_epo{:.0f}_score{:.2f}.pth".format(
                            epoch, best_score
                        ),
                ),
            )
            if args.distributed:
                # 确保所有进程都完成保存
                torch.distributed.barrier()
            try:
                torch.save(
                    model_engine.state_dict(),
                    os.path.join(
                        save_dir,
                        "model_epo{:.0f}_score{:.2f}.pth".format(
                            epoch, best_score
                        ),
                    ),
                )
            except Exception as e:
                print("Failed to save checkpoint (): ", e)
    
    
    if args.global_rank == 0:
        if not args.debug:
            # wandb.finish()
            writer.close()




if __name__ == "__main__":
    from types import SimpleNamespace
    if isinstance(args, dict):
        args = SimpleNamespace(**args)
    main(args)
from deepspec.trainer import LagunaDSparkTrainer


project_name = "opjax"
exp_name = "laguna_dflash_matched_v1"
seed = 42

model = {
    "target_model_name_or_path": "poolside/Laguna-XS-2.1",
    "target_revision": "e9df9a59996d790b94b70f3fef343fe1d9e34bdf",
    "target_model_type": "laguna",
    "trust_remote_code": True,
    "target_device_map": None,
    "target_max_memory": None,
    "target_offload_folder": None,
    "initial_dflash_checkpoint": "/mnt/training/initialized/dflash",
    "block_size": 16,
    "proposal_length": 15,
    "num_draft_layers": 5,
    "target_layer_ids": [1, 13, 25, 33, 39],
    "mask_token_id": 12,
    "num_anchors": 64,
    "draft_num_attention_heads": 64,
    "draft_num_key_value_heads": 8,
    "draft_head_dim": 128,
    "draft_sliding_window": 512,
    "draft_rope_theta": 500000.0,
    "training_attn_implementation": "flex_attention",
    "markov_rank": 0,
    "markov_head_type": "vanilla",
    "confidence_head_alpha": 0.0,
    "confidence_head_with_markov": False,
    "loss_decay_gamma": 4.0,
    "ce_loss_alpha": 0.1,
    "l1_loss_alpha": 0.9,
}

train = {
    "trainer_cls": LagunaDSparkTrainer,
    "lr": 6.0e-4,
    "warmup_ratio": 0.04,
    "weight_decay": 0.0,
    "precision": "bf16",
    "local_batch_size": 1,
    "global_batch_size": 8,
    "num_train_epochs": 10,
    "max_grad_norm": 1.0,
    "sharding_strategy": "no_shard",
    "torch_compile": False,
}

logging = {
    "logging_steps": 1,
    "checkpointing_steps": 13,
    "checkpoint_dir": "/mnt/training/checkpoints/dflash",
    "tensorboard_dir": "/mnt/training/tensorboard/dflash",
}

data = {
    "target_cache_path": "/mnt/training/cache/train",
    "chat_template": "laguna_thinking",
    "max_length": 18432,
    "num_workers": 2,
}

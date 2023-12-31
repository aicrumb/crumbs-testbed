# crumbs-testbed
you can NOT judge the code because it runs on my local machine for my local machine

### trainer

example config.json setup for the trainer with comments (not valid json, dont use comments in your config)
```
{
    "model_id": "EleutherAI/pythia-70m", 
    "bf16": true, 
    "load_in_8bit": true,
    "load_in_4bit": false,
    "optim": "adamw_hf", 
    "lora": {
        "rank": 4, 
        "alpha": 32, 
        "target_modules": ["query_key_value"], // q_proj, k_proj, v_proj, o_proj is what you should use for llama models
        "lora_dropout": 0.05 // consider increasing for multiple epochs
    }, 
    // alpaca will utilize the dataset_* arguments to concat parts of the dataset
    "data_processing": "alpaca", // alpaca, packing, alpaca-packing
    "learning_rate": 0.0001, 
    
    // remember this is the *per_device_train_batch_size*, when using multiple gpus, you should keep that in mind
    // full batch size = batch_size * gradient_accumulation_steps * gpus
    "batch_size": 1,
    "gradient_accumulation_steps": 32, 
    "context_length": 256, 
    "logging_steps": 8, 
    
    // both of these can be set to -1 to negate eachother
    "max_steps": -1, 
    "num_train_epochs": 1, 
    "warmup_steps": 128,
    
    "dataset_text_column": "text", // keep this one as "text" even if using alpaca
    
    "dataset_instruction_column": "message_2", 
    "dataset_input_column": "none", // "none" if not applicable
    "dataset_output_column": "message_2", 
    
    // input will be considered part of human input concated after a newline
    "dataset_user_role": "### human:", 
    "dataset_assistant_role": "### response:", 
    "cluster": "none", // only if your dataset has a "cluster" argument
    "all_columns": ["role_1", "topic", "sub_topic", "message_1", "message_2"], // only if streaming is this required
    "dataset_kwargs": {
        "path": "camel-ai/math", 
        "name": null, // actually dont know what this param does
        "streaming": false, 
        "split": "train"
    }, 
    "output_is_repo": true, // if not repo, will save to folder
    "output": "crumb/trainer-model", 
}
```
```
python trainer.py --config config.json
```

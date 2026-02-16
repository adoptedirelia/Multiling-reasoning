"""
Utility functions and configuration module

Provides common utility functions and model configurations for attention visualization tools.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from matplotlib import pyplot as plt
from typing import List, Tuple, Optional, Dict, Any, Union
import os


# Predefined model configurations
MODEL_CONFIGS = {
    "llama-3.1-8b": {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "attn_implementation": "eager",
        "use_chat_template": True,
    },
    "llama-3.1-70b": {
        "model_name": "meta-llama/Llama-3.1-70B-Instruct",
        "attn_implementation": "eager",
        "use_chat_template": True,
    },
    "mistral-7b": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
        "attn_implementation": "eager",
        "use_chat_template": True,
    },
    "qwen-7b": {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "attn_implementation": "eager",
        "use_chat_template": True,
    },
    "phi-2": {
        "model_name": "microsoft/phi-2",
        "attn_implementation": "eager",
        "use_chat_template": False,
    },
}


class ModelAttentionAnalyzer:
    """Multi-Model Attention Analyzer - Supports various language models"""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        model_config: Optional[str] = None,
        torch_dtype: Union[str, torch.dtype] = "auto",
        device_map: str = "auto",
        attn_implementation: Optional[str] = None,
        use_chat_template: Optional[bool] = None,
        **kwargs
    ):
        """
        Initialize Model Attention Analyzer
        
        Args:
            model_name: Model name or path (overrides model_config if provided)
            model_config: Predefined model configuration key (e.g., "llama-3.1-8b")
            torch_dtype: Torch data type ("auto", torch.float16, etc.)
            device_map: Device mapping strategy
            attn_implementation: Attention implementation method ("eager", "flash_attention_2", etc.)
            use_chat_template: Whether to use chat template (auto-detected if None)
            **kwargs: Additional arguments passed to from_pretrained
        """
        # Load configuration if model_config is provided
        if model_config and model_config in MODEL_CONFIGS:
            config = MODEL_CONFIGS[model_config].copy()
            if model_name is None:
                model_name = config.pop("model_name")
            if attn_implementation is None:
                attn_implementation = config.pop("attn_implementation", "eager")
            if use_chat_template is None:
                use_chat_template = config.pop("use_chat_template", True)
            kwargs.update(config)
        elif model_name is None:
            # Default to llama-3.1-8b if nothing specified
            model_name = MODEL_CONFIGS["llama-3.1-8b"]["model_name"]
            if attn_implementation is None:
                attn_implementation = MODEL_CONFIGS["llama-3.1-8b"]["attn_implementation"]
            if use_chat_template is None:
                use_chat_template = MODEL_CONFIGS["llama-3.1-8b"]["use_chat_template"]
        
        self.model_name = model_name
        self.use_chat_template = use_chat_template if use_chat_template is not None else True
        self.model = None
        self.tokenizer = None
        self._load_model(torch_dtype, device_map, attn_implementation, **kwargs)
    
    def _load_model(
        self,
        torch_dtype: Union[str, torch.dtype],
        device_map: str,
        attn_implementation: Optional[str],
        **kwargs
    ) -> None:
        """
        Load model and tokenizer
        
        Args:
            torch_dtype: Torch data type
            device_map: Device mapping strategy
            attn_implementation: Attention implementation method
            **kwargs: Additional arguments for from_pretrained
        """
        print(f"Loading model: {self.model_name}")
        
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
        }
        
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        
        model_kwargs.update(kwargs)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Auto-detect chat template support if not specified
        if self.use_chat_template and self.tokenizer.chat_template is None:
            print("Warning: Model does not have chat template, using direct tokenization")
            self.use_chat_template = False
        
        print(f"Model loaded successfully (chat_template: {self.use_chat_template})")
    
    def prepare_input(
        self,
        prompt: str,
        role: str = "user",
        use_chat_template: Optional[bool] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare model input
        
        Args:
            prompt: Input prompt text
            role: Message role (default: "user")
            use_chat_template: Override default chat template usage
            
        Returns:
            Dictionary containing input_ids and other information
        """
        use_template = use_chat_template if use_chat_template is not None else self.use_chat_template
        
        if use_template and self.tokenizer.chat_template is not None:
            messages = [{"role": role, "content": prompt}]
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                print(f"Warning: Failed to apply chat template: {e}, using direct tokenization")
                text = prompt
        else:
            text = prompt
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        return model_inputs
    
    def generate_with_attention(
        self,
        model_inputs: Dict[str, torch.Tensor],
        max_new_tokens: int = 512,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Generate text and return attention weights
        
        Args:
            model_inputs: Model input
            max_new_tokens: Maximum number of tokens to generate
            use_cache: Whether to use cache
            
        Returns:
            Dictionary containing generation results and attention weights
        """
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            output_attentions=True,
            return_dict_in_generate=True,
            use_cache=use_cache
        )
        
        # Extract newly generated token IDs
        out_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(
                model_inputs.input_ids,
                generated_ids.sequences
            )
        ]
        
        # Decode response
        response = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
        
        return {
            "generated_ids": generated_ids,
            "out_ids": out_ids,
            "response": response,
            "attentions": generated_ids.attentions,
            "input_length": len(model_inputs.input_ids[0])
        }
    
    def find_text_token_indices(
        self,
        text: str,
        input_ids: List[int],
        multi_part: bool = False
    ) -> Tuple[int, int]:
        """
        Find the start and end positions of text tokens in the input sequence
        
        Args:
            text: Text to find
            input_ids: List of token IDs for the complete input
            
        Returns:
            Tuple of (start_index, end_index)
        """
        if not multi_part:
            text_tokens = self.tokenizer.encode(text, add_special_tokens=False)
        else:
            text_tokens = self.tokenizer.encode(text + '\n', add_special_tokens=False)
        start_idx = None
        end_idx = None

        # Find the position of text tokens in the input
        for i in range(len(input_ids) - len(text_tokens) + 1):
            if input_ids[i:i+len(text_tokens)] == text_tokens:
                start_idx = i
                end_idx = i + len(text_tokens)
                break

        if start_idx is None:
            # If no exact match found, try re-encoding
            text_tokens_retry = self.tokenizer.encode(text, add_special_tokens=False)
            for i in range(len(input_ids) - len(text_tokens_retry) + 1):
                if input_ids[i:i+len(text_tokens_retry)] == text_tokens_retry:
                    start_idx = i
                    end_idx = i + len(text_tokens_retry)
                    break
        
        return start_idx, end_idx
    
    def find_question_token_indices(
        self,
        prompt: str,
        input_ids: List[int]
    ) -> Tuple[int, int]:
        """
        Find the start and end positions of question tokens in the input sequence
        
        Args:
            prompt: Original question text
            input_ids: List of token IDs for the complete input
            
        Returns:
            Tuple of (start_index, end_index)
        """
        start_idx, end_idx = self.find_text_token_indices(prompt, input_ids)
        
        if start_idx is None:
            print("Warning: Could not find question token positions, using all tokens")
            start_idx = 0
            end_idx = len(input_ids)
        
        return start_idx, end_idx
    
    def find_multipart_token_indices(
        self,
        full_prompt: str,
        parts: Dict[str, str],
        input_ids: List[int]
    ) -> Dict[str, Tuple[int, int]]:
        """
        Find token indices for multiple parts in the prompt
        
        Args:
            full_prompt: The complete prompt text
            parts: Dictionary mapping part names to their text content
                   e.g., {"question": "...", "English_Question": "...", ...}
            input_ids: List of token IDs for the complete input
            
        Returns:
            Dictionary mapping part names to (start_index, end_index) tuples
        """
        part_indices = {}
        
        for part_name, part_text in parts.items():

            if part_text:

                start_idx, end_idx = self.find_text_token_indices(part_text, input_ids, multi_part=True)
                if start_idx is not None:
                    part_indices[part_name] = (start_idx, end_idx)
                    print(f"{part_name} token positions: {start_idx} to {end_idx}")
                else:
                    print(f"Warning: Could not find {part_name} token positions")
                    part_indices[part_name] = None
            else:
                print(f"Warning: {part_name} text is empty")
                part_indices[part_name] = None
        
        return part_indices
    
    def extract_question_attention(
        self,
        attention_tensor: torch.Tensor,
        question_start_idx: int,
        question_end_idx: int,
        question_token_texts: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract attention weights for the question part from attention tensor
        
        Args:
            attention_tensor: Attention tensor [batch, num_heads, seq_len, seq_len]
            question_start_idx: Question start index
            question_end_idx: Question end index
            question_token_texts: List of question token texts
            
        Returns:
            Tuple of (attention matrix, current token text list)
        """
        # Average all attention heads
        temp = attention_tensor.mean(dim=1).squeeze()  # [seq_len, seq_len]
        temp = torch.softmax(temp, dim=-1)
        temp = temp.detach().cpu().float().numpy()
        
        # Extract attention for question part
        seq_len = temp.shape[0]
        if question_end_idx <= seq_len:
            question_attention = temp[question_start_idx:question_end_idx]
            current_question_token_texts = question_token_texts
        else:
            # If sequence length is insufficient, only take available part
            available_end = min(question_end_idx, seq_len)
            question_attention = temp[
                question_start_idx:available_end,
                question_start_idx:available_end
            ]
            current_question_token_texts = question_token_texts[:available_end-question_start_idx]
        
        return question_attention, current_question_token_texts
    
    def extract_part_attention(
        self,
        attention_tensor: torch.Tensor,
        part_start_idx: int,
        part_end_idx: int,
        part_token_texts: List[str],
        answer_start_idx: int
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract attention weights from answer tokens to a specific part tokens
        
        Args:
            attention_tensor: Attention tensor [batch, num_heads, seq_len, seq_len]
            part_start_idx: Part start index in input
            part_end_idx: Part end index in input
            part_token_texts: List of part token texts
            answer_start_idx: Answer start index (where generation begins)
            
        Returns:
            Tuple of (attention matrix, token text list)
            Attention matrix shape: [answer_tokens, part_tokens]
            Each row represents attention from one answer token to all part tokens
        """
        # Average all attention heads

        temp = attention_tensor.mean(dim=1).squeeze() # [seq_len+answer_tokens]
        # temp = torch.softmax(temp, dim=-1)
        temp = temp[part_start_idx:part_end_idx]
        # temp = torch.softmax(temp, dim=-1)
        temp = temp.detach().cpu().float().numpy()
        

        return temp, part_token_texts
    
    def process_attention_layers(
        self,
        attention_step: List[torch.Tensor],
        question_start_idx: int,
        question_end_idx: int,
        question_token_texts: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Process all attention layers for a single generation step
        
        Args:
            attention_step: List of attention tensors for a single step (one tensor per layer)
            question_start_idx: Question start index
            question_end_idx: Question end index
            question_token_texts: List of question token texts
            
        Returns:
            Tuple of (attention matrices for all layers, token text list)
        """
        attention_maps = []
        current_question_token_texts = None
        
        for layer_idx in range(len(attention_step)):
            attention_tensor = attention_step[layer_idx]
            question_attention, token_texts = self.extract_question_attention(
                attention_tensor,
                question_start_idx,
                question_end_idx,
                question_token_texts
            )
            attention_maps.append(question_attention)
            if current_question_token_texts is None:
                current_question_token_texts = token_texts
        
        attention_maps = np.array(attention_maps)
        return attention_maps, current_question_token_texts
    
    def plot_attention(
        self,
        attention_maps: np.ndarray,
        token_texts: List[str],
        output_text: str,
        save_path: str,
        figsize: Tuple[int, int] = (10, 10),
        dpi: int = 150,
        y_axis_label: Optional[str] = None
    ) -> None:
        """
        Plot attention weight heatmap
        
        Args:
            attention_maps: Attention matrix array 
                - [num_layers, num_tokens, num_tokens] for layer-wise visualization
                - [num_tokens, num_tokens] for single matrix
                - [answer_tokens, part_tokens] for answer-to-part attention
            token_texts: List of token texts (for x-axis)
            output_text: Output text (for title)
            save_path: Save path
            figsize: Figure size
            dpi: Image resolution
            y_axis_label: Label for y-axis (default: 'Query Position' or 'Layer')
        """
        plt.figure(figsize=figsize)

        # Handle different attention map shapes
        if len(attention_maps.shape) == 3:
            # [num_layers, num_tokens, num_tokens] - show all layers
            im = plt.imshow(attention_maps, cmap="viridis", aspect='auto')
            y_axis_label = y_axis_label or 'Layer'
        elif len(attention_maps.shape) == 2:
            # [answer_tokens, part_tokens] or [num_tokens, num_tokens]
            im = plt.imshow(attention_maps, cmap="viridis", aspect='auto')
            y_axis_label = y_axis_label or 'Answer Token Position'
        else:
            raise ValueError(f"Unsupported attention_maps shape: {attention_maps.shape}")
        
        plt.colorbar(im, label='Attention Weight')

        num_tokens = attention_maps.shape[-1]

        plt.xticks(
            range(num_tokens),
            token_texts,
            rotation=45,
            ha='right',
            fontsize=8
        )
        plt.xlabel('Part Token Position', fontsize=10)
        plt.ylabel(y_axis_label, fontsize=10)
        plt.title(output_text, fontsize=12)
        plt.tight_layout()
        
        # Ensure save directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    
    def visualize_attentions(
        self,
        prompt: str,
        output_dir: str = "./attention_plots",
        max_new_tokens: int = 512,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Complete attention visualization pipeline
        
        Args:
            prompt: Input prompt text
            output_dir: Output directory
            max_new_tokens: Maximum number of tokens to generate
            use_cache: Whether to use cache
            
        Returns:
            Dictionary containing generation results and visualization information
        """
        # Prepare input
        model_inputs = self.prepare_input(prompt)
        
        # Generate text and get attention
        generation_result = self.generate_with_attention(
            model_inputs,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache
        )
        
        print(f"Generated response: {generation_result['response']}")
        print(f"Input length: {generation_result['input_length']}")
        
        # Find question token positions
        input_ids = model_inputs.input_ids[0].cpu().tolist()
        question_start_idx, question_end_idx = self.find_question_token_indices(
            prompt,
            input_ids
        )
        
        print(f"Question token positions: {question_start_idx} to {question_end_idx}")
        question_token_ids = input_ids[question_start_idx:question_end_idx]
        question_token_texts = [
            self.tokenizer.decode([tid]) for tid in question_token_ids
        ]
        print(f"Question token texts: {question_token_texts}")
        
        # Process attention for each generation step
        attentions = generation_result['attentions']
        out_ids = generation_result['out_ids']
        
        for idx, attention_step in enumerate(attentions[1:]):
            # Process attention for all layers
            attention_maps, current_token_texts = self.process_attention_layers(
                attention_step,
                question_start_idx,
                question_end_idx,
                question_token_texts
            )
            
            # Get output text for current step
            output_text = self.tokenizer.decode(
                out_ids[0][idx],
                skip_special_tokens=True
            )
            
            # Plot and save
            save_path = os.path.join(output_dir, f"attention_{idx}.png")
            self.plot_attention(
                attention_maps,
                current_token_texts,
                output_text,
                save_path
            )
        
        return generation_result
    
    def visualize_multipart_attentions(
        self,
        full_prompt: str,
        parts: Dict[str, str],
        method: str = "mean",
        output_dir: str = "./attention_plots",
        max_new_tokens: int = 512,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Visualize attention from answer to multiple parts of the prompt
        
        Args:
            full_prompt: The complete prompt text
            parts: Dictionary mapping part names to their text content
                   e.g., {
                       "question": "...",
                       "English_Question": "...",
                       "English_Thinking_Process": "...",
                       "English_Answer": "..."
                   }
            output_dir: Output directory
            max_new_tokens: Maximum number of tokens to generate
            use_cache: Whether to use cache
            
        Returns:
            Dictionary containing generation results and visualization information
        """
        # Prepare input
        model_inputs = self.prepare_input(full_prompt)
        
        # Generate text and get attention
        generation_result = self.generate_with_attention(
            model_inputs,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache
        )
        
        print(f"Generated response: {generation_result['response']}")
        print(f"Input length: {generation_result['input_length']}")
        
        # Find token positions for all parts
        input_ids = model_inputs.input_ids[0].cpu().tolist()

        part_indices = self.find_multipart_token_indices(full_prompt, parts, input_ids)

        # Get token texts for each part
        part_token_texts = {}
        for part_name, indices in part_indices.items():
            if indices is not None:
                start_idx, end_idx = indices
                part_token_ids = input_ids[start_idx:end_idx]
                part_token_texts[part_name] = [
                    self.tokenizer.decode([tid]) for tid in part_token_ids
                ]
                print(f"{part_name} token texts: {part_token_texts[part_name][:5]}...")  # Show first 5
        output_texts = []
        for values in part_token_texts.values():
            output_texts.extend(values)

        # Find answer start position (end of input)
        answer_start_idx = len(input_ids)
        
        # Process attention for each generation step
        attentions = generation_result['attentions']
        out_ids = generation_result['out_ids']
        final_output_map = []

        for step_idx, attention_step in enumerate(attentions[1:]):
            # Current sequence length includes all previous generated tokens
            current_seq_len = len(input_ids) + step_idx + 1
            
            # Process each part
            part_attention_maps = {}
            for part_name, indices in part_indices.items():
                if indices is None:
                    continue

                start_idx, end_idx = indices
                
                # Process attention for all layers
                attention_maps_list = []
                part_texts = None
                
                for layer_idx in range(len(attention_step)):
                    attention_tensor = attention_step[layer_idx]
                    
                    # Extract attention from answer to this part
                    part_attention, token_texts = self.extract_part_attention(
                        attention_tensor,
                        start_idx,
                        end_idx,
                        part_token_texts[part_name],
                        answer_start_idx
                    )

                    
                    if part_attention.size > 0:
                        # Stack layers: [num_layers, answer_tokens, part_tokens]
                        attention_maps_list.append(part_attention)
                        if part_texts is None:
                            part_texts = token_texts
                
                if attention_maps_list:
                    # Stack all layers: [num_layers, answer_tokens, part_tokens]
                    attention_maps = np.array(attention_maps_list)
                    
                    part_attention_maps[part_name] = attention_maps
            
            combined_attention = combine_attention_maps(part_attention_maps, method=method)
            final_output_map.append(combined_attention)
        save_path = os.path.join(
            output_dir,
            f"final_output_map.png"
        )
        final_output_map = np.array(final_output_map)
        final_output_map = final_output_map.mean(axis=0)

        self.plot_attention(
            final_output_map,
            output_texts if method == "original" else list(part_indices.keys()),   
            f"Final output map {method}",
            save_path,
            y_axis_label="Step"
        )
        return generation_result
    
    def generate_with_teacher_forcing(
        self,
        prompt: str,
        answer: str,
        role: str = "user"
    ) -> Dict[str, Any]:
        """
        Generate with teacher forcing (provide full input including answer)
        This is useful for analyzing attention patterns when the full sequence is known.
        
        Args:
            prompt: Input prompt text
            answer: Answer text to include in the input
            role: Message role (default: "user")
            
        Returns:
            Dictionary containing model outputs and attention weights
        """
        # Prepare input with prompt only (to get input length)
        if self.use_chat_template and self.tokenizer.chat_template is not None:
            messages = [{"role": role, "content": prompt}]
            try:
                text_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                print(f"Warning: Failed to apply chat template: {e}, using direct tokenization")
                text_prompt = prompt
                text_full = f"{prompt} {answer}"
        else:
            text_prompt = prompt
            text_full = f"{prompt} {answer}"
        
        model_inputs_prompt = self.tokenizer([text_prompt], return_tensors="pt").to(self.model.device)
        input_len = model_inputs_prompt.input_ids.shape[1]
        
        # Prepare full input with answer
        if self.use_chat_template and self.tokenizer.chat_template is not None:
            try:
                messages_with_answer = messages + [{"role": "assistant", "content": answer}]
                text_full = self.tokenizer.apply_chat_template(
                    messages_with_answer,
                    tokenize=False,
                )
            except Exception as e:
                print(f"Warning: Failed to apply chat template: {e}, using direct tokenization")
                text_full = f"{prompt} {answer}"
        
        model_inputs = self.tokenizer([text_full], return_tensors="pt").to(self.model.device)
        
        # Set labels for loss computation (mask prompt part)
        model_inputs["labels"] = model_inputs.input_ids.clone()
        model_inputs["labels"][:, :input_len] = -100
        if self.tokenizer.pad_token_id is not None:
            model_inputs["labels"][model_inputs.input_ids == self.tokenizer.pad_token_id] = -100
        
        # Forward pass with teacher forcing
        generated_outputs = self.model(
            **model_inputs,
            output_attentions=True,
            return_dict=True
        )
        
        # Extract generated token IDs
        generated_ids = generated_outputs.logits.argmax(dim=-1)[:, input_len-1:]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return {
            "outputs": generated_outputs,
            "input_length": input_len,
            "attentions": generated_outputs.attentions,
            "model_inputs": model_inputs
        }
    
    def extract_aggregated_question_attention(
        self,
        attentions: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
        input_len: int,
        question_start_idx: int,
        question_end_idx: int
    ) -> np.ndarray:
        """
        Extract aggregated attention from all generated tokens to question tokens
        For each layer, sum attention from all generated tokens to each question token.
        
        Args:
            attentions: Tuple of attention tensors, one per layer
                      Each tensor shape: [batch, num_heads, seq_len, seq_len]
            input_len: Length of input tokens (generation starts after this)
            question_start_idx: Question start index in input
            question_end_idx: Question end index in input
            
        Returns:
            Attention matrix of shape [num_layers, num_question_tokens]
            Each row represents one layer, each column represents one question token
        """
        final_lst = []
        
        for attention in attentions:
            # attention shape: [batch, num_heads, seq_len, seq_len]
            # Average over heads: [batch, seq_len, seq_len]
            temp = attention[0].mean(dim=0)  # [seq_len, seq_len]
            
            # Extract attention from generated tokens (input_len:) to all tokens
            # Then sum over all generated tokens: [seq_len]
            temp = temp[input_len:].mean(dim=0)  # [seq_len]
            
            # Extract question token range: [num_question_tokens]
            temp = temp[question_start_idx:question_end_idx]
            
            # Apply softmax
            temp = temp.softmax(dim=0)
            
            # Convert to numpy
            temp = temp.cpu().float().detach().numpy()
            final_lst.append(temp)
        
        return np.array(final_lst)  # [num_layers, num_question_tokens]
    
    def extract_aggregated_part_attention(
        self,
        attentions: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
        input_len: int,
        part_start_idx: int,
        part_end_idx: int
    ) -> np.ndarray:
        """
        Extract aggregated attention from all generated tokens to part tokens
        For each layer, sum attention from all generated tokens to each part token.
        
        Args:
            attentions: Tuple of attention tensors, one per layer
                      Each tensor shape: [batch, num_heads, seq_len, seq_len]
            input_len: Length of input tokens (generation starts after this)
            part_start_idx: Part start index in input
            part_end_idx: Part end index in input
            
        Returns:
            Attention matrix of shape [num_layers, num_part_tokens]
            Each row represents one layer, each column represents one part token
        """
        final_lst = []
        
        for attention in attentions:
            # attention shape: [batch, num_heads, seq_len, seq_len]
            # Average over heads: [batch, seq_len, seq_len]
            temp = attention[0].mean(dim=0)  # [seq_len, seq_len]
            
            # Extract attention from generated tokens (input_len:) to all tokens
            # Then sum over all generated tokens: [seq_len]
            temp = temp[input_len:].mean(dim=0)  # [seq_len]
            
            # Extract part token range: [num_part_tokens]
            temp = temp[part_start_idx:part_end_idx]
            
            # We should not apply softmax here, we will apply it later
            # temp = temp.softmax(dim=0)
            
            # Convert to numpy
            temp = temp.cpu().float().detach().numpy()
            final_lst.append(temp)
        
        return np.array(final_lst)  # [num_layers, num_part_tokens]
    
    def visualize_teacher_forcing_attention(
        self,
        prompt: str,
        answer: str,
        output_path: str = "./attention_map.png",
        figsize: Tuple[int, int] = (10, 10),
        dpi: int = 150
    ) -> Dict[str, Any]:
        """
        Visualize attention using teacher forcing approach (like llama_test2.py)
        
        Args:
            prompt: Input prompt text
            answer: Answer text to include in the input
            output_path: Path to save the attention map
            figsize: Figure size
            dpi: Image resolution
            
        Returns:
            Dictionary containing generation results and visualization information
        """
        # Generate with teacher forcing
        result = self.generate_with_teacher_forcing(prompt, answer)
        
        # Find question token positions
        input_ids = result["model_inputs"].input_ids[0].cpu().tolist()
        question_start_idx, question_end_idx = self.find_question_token_indices(
            prompt,
            input_ids
        )
        
        if question_start_idx is None:
            print("Warning: Could not find question token positions")
            return result
        
        print(f"Question token positions: {question_start_idx} to {question_end_idx}")
        print(f"Input length: {result['input_length']}")

        
        # Extract aggregated attention
        attention_maps = self.extract_aggregated_question_attention(
            result["attentions"],
            result["input_length"],
            question_start_idx,
            question_end_idx
        )
        
        # Get question token texts for visualization
        question_token_ids = input_ids[question_start_idx:question_end_idx]
        question_token_texts = [
            self.tokenizer.decode([tid]) for tid in question_token_ids
        ]
        
        # Plot attention map
        plt.figure(figsize=figsize)
        im = plt.imshow(attention_maps, cmap='viridis', aspect='auto')
        plt.colorbar(im, label='Attention Weight')
        
        # Set x-axis labels (question tokens)
        plt.xticks(
            range(len(question_token_texts)),
            question_token_texts,
            rotation=45,
            ha='right',
            fontsize=8
        )
        plt.xlabel('Question Token Position', fontsize=10)
        plt.ylabel('Layer', fontsize=10)
        plt.title(f'Attention from Generated Tokens to Question\nPrompt: {prompt[:50]}...', fontsize=12)
        plt.tight_layout()
        
        # Ensure save directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Attention map saved to: {output_path}")
        
        return result
    
    def visualize_multipart_teacher_forcing_attention(
        self,
        full_prompt: str,
        answer: str,
        parts: Dict[str, str],
        method: str = "mean",
        output_path: str = "./attention_map_multipart.png",
        figsize: Tuple[int, int] = (10, 10),
        dpi: int = 150
    ) -> Dict[str, Any]:
        """
        Visualize attention from generated tokens to multiple parts using teacher forcing
        
        Args:
            full_prompt: The complete prompt text
            answer: Answer text to include in the input
            parts: Dictionary mapping part names to their text content
                   e.g., {
                       "question": "...",
                       "English_Question": "...",
                       "English_Thinking_Process": "...",
                       "English_Answer": "..."
                   }
            method: Combination method, one of "mean", "max", "original"
            output_path: Path to save the attention map
            figsize: Figure size
            dpi: Image resolution
            
        Returns:
            Dictionary containing generation results and visualization information
        """
        # Generate with teacher forcing
        result = self.generate_with_teacher_forcing(full_prompt, answer)
        
        # Find token positions for all parts
        input_ids = result["model_inputs"].input_ids[0].cpu().tolist()
        part_indices = self.find_multipart_token_indices(full_prompt, parts, input_ids)
        
        # Get token texts for each part
        part_token_texts = {}
        for part_name, indices in part_indices.items():
            if indices is not None:
                start_idx, end_idx = indices
                part_token_ids = input_ids[start_idx:end_idx]
                part_token_texts[part_name] = [
                    self.tokenizer.decode([tid]) for tid in part_token_ids
                ]
                print(f"{part_name} token positions: {start_idx} to {end_idx}")
                print(f"{part_name} token texts: {part_token_texts[part_name][:5]}...")  # Show first 5
        
        print(f"Input length: {result['input_length']}")
        
        # Extract aggregated attention for each part
        part_attention_maps = {}
        for part_name, indices in part_indices.items():
            if indices is None:
                continue
            
            start_idx, end_idx = indices
            
            # Extract aggregated attention for this part
            # Shape: [num_layers, num_part_tokens]
            part_attention = self.extract_aggregated_part_attention(
                result["attentions"],
                result["input_length"],
                start_idx,
                end_idx
            )

            # For multipart visualization with teacher forcing:
            # - We already aggregated over answer tokens (sum)
            # - Now we need to handle part tokens based on method
            if method == "mean":
                # Average over part tokens: [num_layers, num_part_tokens] -> [num_layers]
                part_attention = part_attention.mean(axis=-1)  # [num_layers]
            elif method == "max":
                # Max over part tokens: [num_layers, num_part_tokens] -> [num_layers]
                part_attention = part_attention.max(axis=-1)  # [num_layers]
            elif method == "original":
                # Keep all part tokens: [num_layers, num_part_tokens]
                pass
            else:
                raise ValueError(f"Invalid method: {method}. Must be one of 'mean', 'max', 'original'")
            
            part_attention_maps[part_name] = part_attention
        
        # Combine attention maps from all parts
        if method == "original":
            # Concatenate all parts: [num_layers, total_tokens]
            combined_attention = np.concatenate(
                [part_attention_maps[part_name] for part_name in part_indices.keys() 
                 if part_indices[part_name] is not None],
                axis=-1
            )
        else:
            # Stack all parts: [num_layers, num_parts]
            combined_attention = np.stack(
                [part_attention_maps[part_name] for part_name in part_indices.keys() 
                 if part_indices[part_name] is not None],
                axis=-1
            )
            # Apply softmax over parts dimension

        combined_attention = torch.softmax(torch.tensor(combined_attention), dim=-1).numpy()
        
        # Prepare labels for visualization
        if method == "original":
            # Concatenate all part token texts
            output_texts = []
            for part_name in part_indices.keys():
                if part_indices[part_name] is not None:
                    output_texts.extend(part_token_texts[part_name])
            labels = output_texts
        else:
            # Use part names
            labels = [part_name for part_name in part_indices.keys() if part_indices[part_name] is not None]
        
        # Plot attention map
        plt.figure(figsize=figsize)
        im = plt.imshow(combined_attention, cmap='viridis', aspect='auto')
        plt.colorbar(im, label='Attention Weight')
        
        # Set x-axis labels
        plt.xticks(
            range(len(labels)),
            labels,
            rotation=45,
            ha='right',
            fontsize=8
        )
        plt.xlabel('Part' if method != "original" else 'Token Position', fontsize=10)
        plt.ylabel('Layer', fontsize=10)
        title = f'Attention from Generated Tokens to Parts ({method})\nPrompt: {full_prompt[:50]}...'
        plt.title(title, fontsize=12)
        plt.tight_layout()
        
        # Ensure save directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Multipart attention map saved to: {output_path}")
        
        return result


def combine_attention_maps(part_attention_maps: Dict[str, np.ndarray], method: str = "mean") -> torch.Tensor:
    """
    Combine attention maps from multiple parts
    
    Args:
        part_attention_maps: Dictionary mapping part names to attention matrix arrays
                           shape: [num_layers, answer_tokens, part_tokens]
        method: Combination method, one of "mean", "max", "original"
                - "mean": Average over part_tokens dimension
                - "max": Maximum over part_tokens dimension
                - "original": Preserve original shape, concatenate all parts
    
    Returns:
        Combined attention tensor with softmax applied
        - method="original": shape [num_layers, answer_tokens, total_part_tokens]
        - method="mean" or "max": shape [num_layers, answer_tokens, num_parts]
    """
    res = []
    for part_name, attention_maps in part_attention_maps.items():
        if method == "mean":
            mean_attention_maps = attention_maps.mean(axis=-1)
        elif method == "max":
            mean_attention_maps = attention_maps.max(axis=-1)
        elif method == "original":
            mean_attention_maps = attention_maps
        else:
            raise ValueError(f"Invalid method: {method}. Must be one of 'mean', 'max', 'original'")
        res.append(mean_attention_maps)

    if method == "original":
        res = torch.tensor(np.concatenate(res, axis=-1))
    else:
        res = torch.tensor(np.stack(res, axis=-1))

    return torch.softmax(res, dim=-1)


def list_available_models() -> None:
    """
    List all available predefined model configurations
    """
    print("Available predefined model configurations:")
    print("-" * 60)
    for key, config in MODEL_CONFIGS.items():
        print(f"\n{key}:")
        print(f"  Model: {config['model_name']}")
        print(f"  Attention: {config.get('attn_implementation', 'default')}")
        print(f"  Chat Template: {config.get('use_chat_template', True)}")
    print("-" * 60)

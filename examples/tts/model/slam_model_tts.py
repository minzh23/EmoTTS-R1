import torch
import os
import logging
import torch.nn.functional as F
from slam_llm.models.slam_model import (
    slam_model,
    setup_tokenizer,
    setup_llm,
)
from slam_llm.utils.train_utils import print_model_size
from typing import List, Optional
from slam_llm.utils.metric import compute_accuracy
from tqdm import tqdm
from utils.codec_utils import setup_codec
from utils.codec_utils import layershift as layer_shift, simple_shift
from utils.projector_utils import setup_group_decode_adapter
from slam_llm.utils.config_utils import generate_peft_config
from peft import get_peft_model

# Add transformers imports for the wrapper
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger(__name__)

class SlamTTSConfig(PretrainedConfig):
    """
    Configuration class for SlamTTS model that inherits from PretrainedConfig
    to make it compatible with Transformers.
    """
    model_type = "slam_tts"
    
    def __init__(
        self,
        train_config=None,
        model_config=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.train_config = train_config
        self.model_config = model_config


class SlamTTSForCausalLM(PreTrainedModel):
    """
    Wrapper class that makes slam_model_tts compatible with Transformers library.
    This allows using Transformers Trainer for training.
    """
    config_class = SlamTTSConfig
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = ["slam_model"]
    
    def __init__(self, config):
        super().__init__(config)
        
        # Store the configuration
        self.train_config = config.train_config
        self.model_config = config.model_config
        
        # Initialize the underlying slam_model_tts
        # This will be set by the model_factory function
        self.slam_model = None
        
    def _init_weights(self, module):
        """Initialize weights - delegated to the underlying model"""
        # The weights are initialized by the underlying slam_model_tts
        pass

    def set_slam_model(self, slam_model):
        """Set the underlying slam_model_tts instance"""
        self.slam_model = slam_model
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """
        Forward pass that delegates to slam_model_tts and returns Transformers-compatible output.
        """
        if self.slam_model is None:
            raise RuntimeError("slam_model is not set. Call set_slam_model() first.")
            
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Call the original slam_model_tts forward method
        model_outputs, text_acc, audio_acc, loss_recorder = self.slam_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            **kwargs,
        )
        
        # Wrap the output in a format compatible with Transformers
        if not return_dict:
            return (model_outputs.loss, model_outputs.logits) if model_outputs.loss is not None else (model_outputs.logits,)
        
        return CausalLMOutputWithPast(
            loss=model_outputs.loss,
            logits=model_outputs.logits,
            past_key_values=model_outputs.past_key_values if hasattr(model_outputs, 'past_key_values') else None,
            hidden_states=model_outputs.hidden_states if hasattr(model_outputs, 'hidden_states') else None,
            attentions=model_outputs.attentions if hasattr(model_outputs, 'attentions') else None,
        )
        
    def generate(self, **kwargs):
        """Delegate generation to the underlying slam_model"""
        if self.slam_model is None:
            raise RuntimeError("slam_model is not set. Call set_slam_model() first.")
        return self.slam_model.generate(**kwargs)
        
    def serial_generate(self, **kwargs):
        """Delegate serial generation to the underlying slam_model"""
        if self.slam_model is None:
            raise RuntimeError("slam_model is not set. Call set_slam_model() first.")
        return self.slam_model.serial_generate(**kwargs)
        
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """Prepare inputs for generation"""
        return {
            "input_ids": input_ids,
            **kwargs
        }
        
    def save_pretrained(self, save_directory, **kwargs):
        """Save the model and configuration to a directory"""
        super().save_pretrained(save_directory, **kwargs)
        
        # Also save the underlying slam_model state dict
        if self.slam_model is not None:
            slam_model_path = os.path.join(save_directory, "slam_model.pt")
            torch.save(self.slam_model.state_dict(), slam_model_path)
            
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load a pretrained model"""
        # Load the wrapper configuration
        config = cls.config_class.from_pretrained(pretrained_model_name_or_path)
        model = cls(config)
        
        # Load the underlying slam_model if saved separately
        slam_model_path = os.path.join(pretrained_model_name_or_path, "slam_model.pt")
        if os.path.exists(slam_model_path):
            # You'll need to recreate the slam_model here
            # This requires having access to the original configuration
            train_config = config.train_config
            model_config = config.model_config
            
            # Recreate the original slam_model_tts
            slam_model, _ = model_factory(train_config, model_config, **kwargs)
            slam_model.load_state_dict(torch.load(slam_model_path, map_location="cpu"))
            model.set_slam_model(slam_model)
            
        return model
        
    def get_input_embeddings(self):
        """Get input embeddings for compatibility"""
        if self.slam_model is not None and hasattr(self.slam_model, 'llm'):
            if hasattr(self.slam_model.llm, 'get_input_embeddings'):
                return self.slam_model.llm.get_input_embeddings()
            elif hasattr(self.slam_model.llm.model, 'embed_tokens'):
                return self.slam_model.llm.model.embed_tokens
            elif hasattr(self.slam_model.llm.model.model, 'embed_tokens'):
                return self.slam_model.llm.model.model.embed_tokens
        return None
        
    def set_input_embeddings(self, value):
        """Set input embeddings for compatibility"""
        if self.slam_model is not None and hasattr(self.slam_model, 'llm'):
            if hasattr(self.slam_model.llm, 'set_input_embeddings'):
                self.slam_model.llm.set_input_embeddings(value)
            elif hasattr(self.slam_model.llm.model, 'embed_tokens'):
                self.slam_model.llm.model.embed_tokens = value
            elif hasattr(self.slam_model.llm.model.model, 'embed_tokens'):
                self.slam_model.llm.model.model.embed_tokens = value
                
    def get_output_embeddings(self):
        """Get output embeddings for compatibility"""
        if self.slam_model is not None and hasattr(self.slam_model, 'llm'):
            if hasattr(self.slam_model.llm, 'get_output_embeddings'):
                return self.slam_model.llm.get_output_embeddings()
            elif hasattr(self.slam_model.llm, 'lm_head'):
                return self.slam_model.llm.lm_head
        return None
        
    def set_output_embeddings(self, new_embeddings):
        """Set output embeddings for compatibility"""
        if self.slam_model is not None and hasattr(self.slam_model, 'llm'):
            if hasattr(self.slam_model.llm, 'set_output_embeddings'):
                self.slam_model.llm.set_output_embeddings(new_embeddings)
            elif hasattr(self.slam_model.llm, 'lm_head'):
                self.slam_model.llm.lm_head = new_embeddings
                
    def resize_token_embeddings(self, new_num_tokens):
        """Resize token embeddings"""
        if self.slam_model is not None and hasattr(self.slam_model, 'llm'):
            if hasattr(self.slam_model.llm, 'resize_token_embeddings'):
                return self.slam_model.llm.resize_token_embeddings(new_num_tokens)
        return None
        
    @property
    def device(self):
        """Get the device of the model"""
        if self.slam_model is not None:
            return next(self.slam_model.parameters()).device
        return torch.device('cpu')

def model_factory(train_config, model_config, **kwargs):
    # return necessary components for training
    tokenizer = setup_tokenizer(train_config, model_config, **kwargs)
    
    encoder = None
    encoder_projector = None

    llm = setup_llm(train_config, model_config, **kwargs)

    codec_decoder = None
    if model_config.codec_decode:
        codec_decoder = setup_codec(train_config, model_config, **kwargs)

    group_decode_adapter = None
    if model_config.group_decode:
        group_decode_adapter = setup_group_decode_adapter(model_config, train_config, **kwargs)
        if train_config.freeze_group_decode_adapter:
            for name, param in group_decode_adapter.named_parameters():
                param.requires_grad = False
            group_decode_adapter.eval()

    model = slam_model_tts(
        encoder,
        llm,
        encoder_projector,
        tokenizer,
        codec_decoder,
        group_decode_adapter,
        train_config,
        model_config,
        **kwargs,
    )

    ckpt_path = kwargs.get("ckpt_path", None)
    if ckpt_path is not None:
        logger.info("loading other parts from: {}\n".format(ckpt_path))
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt_dict, strict=False)

    if train_config.use_peft:
        logger.info("setup peft for llm")
        peft_config = generate_peft_config(train_config)
        model.llm = get_peft_model(model.llm, peft_config)
        if int(os.environ.get("RANK", "0")) == 0:
            model.llm.print_trainable_parameters()

    if kwargs.get("peft_ckpt", None):
        logger.info("loading peft-stage ckpt from: {}\n".format(kwargs.get("peft_ckpt")))
        ckpt_dict = torch.load(kwargs.get("peft_ckpt"), map_location="cpu")
        model.load_state_dict(ckpt_dict, strict=False)
        
    print_model_size(
        model,
        train_config,
        (
            int(os.environ["RANK"])
            if train_config.enable_fsdp or train_config.enable_ddp
            else 0
        ),
    )

    config = SlamTTSConfig(
        train_config=train_config,
        model_config=model_config
    )
    
    wrapper_model = SlamTTSForCausalLM(config)
    wrapper_model.set_slam_model(model)
    
    return wrapper_model, tokenizer


def model_factory_transformers(train_config, model_config, **kwargs):
    """
    Convenience function that returns a Transformers-wrapped model.
    This is equivalent to calling model_factory with use_transformers_wrapper=True.
    """
    kwargs["use_transformers_wrapper"] = True
    return model_factory(train_config, model_config, **kwargs)


class slam_model_tts(slam_model):
    def __init__(
        self,
        encoder,
        llm,
        encoder_projector,
        tokenizer,
        codec_decoder,
        group_decode_adapter,
        train_config,
        model_config,
        **kwargs,
    ):
        super().__init__(
            encoder,
            llm,
            encoder_projector,
            tokenizer,
            train_config,
            model_config,
            **kwargs,
        )

        # resize llm embedding layer
        self.original_vocabsize = self.llm.lm_head.weight.size(0)
        if self.model_config.vocab_config.total_vocabsize != self.original_vocabsize:
            self.llm.resize_token_embeddings(self.model_config.vocab_config.total_vocabsize)
            if int(os.environ.get("RANK", "0")) == 0:
                logger.info("Resize llm embedding layer's vocab size to {}\n".format(self.model_config.vocab_config.total_vocabsize))

        self.codec_decoder = codec_decoder
        self.code_layer = self.model_config.vocab_config.code_layer
        self.group_decode_adapter = group_decode_adapter

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        modality_mask = kwargs.get("modality_mask", None)
        encoder_outs = None

        if input_ids is not None:
            input_ids[input_ids == -1] = 0  # [btz, code_layer + 1, seq_length]
            if hasattr(self.llm.model, "embed_tokens"):
                inputs_embeds = self.llm.model.embed_tokens(input_ids)
            elif hasattr(self.llm.model.model, "embed_tokens"):
                inputs_embeds = self.llm.model.model.embed_tokens(input_ids)
            else:
                inputs_embeds = self.llm.model.model.model.embed_tokens(input_ids)

        if modality_mask is not None and encoder_outs is not None:
            if self.train_config.modeling_paradigm == "parallel":
                modality_mask = modality_mask.unsqueeze(1).repeat(1, self.code_layer, 1)  # [btz, code_layer, seq_length]
                modality_mask_start_indices = (modality_mask == True).float().argmax(dim=2)
                modality_lengths = torch.clamp(modality_mask.sum(dim=2), max=encoder_outs.shape[1]).tolist()

                encoder_outs_pad = torch.zeros_like(inputs_embeds)
                for i in range(encoder_outs.shape[0]):
                    for j in range(self.code_layer):
                        start_idx = modality_mask_start_indices[i, j].item()
                        length = modality_lengths[i][j]
                        encoder_outs_pad[i, j, start_idx:start_idx+length] = encoder_outs[i, :length]
                
                inputs_embeds[:, :self.code_layer, :, :] = encoder_outs_pad[:, :self.code_layer, :, :] + inputs_embeds[:, :self.code_layer, :, :] * (~modality_mask[:, :, :, None])
        
                inputs_embeds = torch.mean(inputs_embeds, dim=1)  # [btz, seq_length, emb_dim], average over the code layers

            elif self.train_config.modeling_paradigm == "interleaved":
                inputs_embeds = inputs_embeds.squeeze(1)  # [btz, seq_length, emb_dim]
                modality_mask_start_indices = (modality_mask == True).float().argmax(dim=1)
                modality_lengths = torch.clamp(modality_mask.sum(dim=1), max=encoder_outs.shape[1]).tolist()

                encoder_outs_pad = torch.zeros_like(inputs_embeds)
                for i in range(encoder_outs.shape[0]):
                    encoder_outs_pad[
                        i, modality_mask_start_indices[i]:modality_mask_start_indices[i]+modality_lengths[i]
                    ] = encoder_outs[i][:modality_lengths[i]]
                
                inputs_embeds = encoder_outs_pad + inputs_embeds * (~modality_mask[:, :, None])
            
            else:
                raise NotImplementedError
        
        inputs_embeds = torch.mean(inputs_embeds, dim=1)  # [btz, seq_length, emb_dim], average over the code layers

        if kwargs.get("inference_mode", False):
            return inputs_embeds, attention_mask

        if self.train_config.modeling_paradigm == "serial":
            temp_labels = labels[:,self.code_layer - 1] if labels is not None else None
            model_outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels= temp_labels)    # here we use the text token layer as the target label

            eot = self.model_config.vocab_config.eot
            text_pad_token = self.model_config.vocab_config.pad_t
            audio_pad_token = self.model_config.vocab_config.pad_a
            text_labels = torch.full_like(labels, -100)
            audio_labels = torch.full_like(labels, -100)
            batch_size, seq_size, length = labels.shape
            for i in range(batch_size):
                eot_position = (labels[i, 0] == eot).nonzero(as_tuple=True)[0]
                eot_pos = eot_position.item()  
                text_labels[i, :, :eot_pos+1] = labels[i, :, :eot_pos+1]
                text_labels[i, :, eot_pos+1:] = text_pad_token

                audio_labels[i, :, eot_pos+1:] = labels[i, :, eot_pos+1:]
                ignore_pos = torch.where(labels[i, 0]!= -100)[0][0].item()
                audio_labels[i, :, ignore_pos:eot_pos+1] = audio_pad_token
            text_labels = text_labels[:, 0, :]
        else:
            text_labels = labels[:,self.code_layer] if labels is not None else None
            audio_labels = labels[:, :self.code_layer] if labels is not None else None
            model_outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=text_labels)    # here we use the text token layer as the target label

        if self.train_config.modeling_paradigm == "parallel" or self.train_config.modeling_paradigm == "serial":
            x_ori = model_outputs.logits
            text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize
            audio_vocab_size = self.model_config.vocab_config.padded_audio_vocabsize
            xt = x_ori[..., :text_vocab_size]
            xa = []

            if self.group_decode_adapter is not None:
                x_audio_ori = x_ori[..., text_vocab_size:]
                x_audio = self.group_decode_adapter(x_audio_ori)
                for i in range(self.code_layer):
                    xa.append(x_audio[..., i * audio_vocab_size : (i + 1) * audio_vocab_size])
            else:
                for i in range(self.code_layer):
                    xa.append(x_ori[..., text_vocab_size + audio_vocab_size * i : text_vocab_size + audio_vocab_size * (i + 1)])

            loss_recorder = []
            total_loss, loss_recorder = self.compute_parallel_loss(xt, text_labels, xa, audio_labels)
            model_outputs.loss = total_loss
        elif self.train_config.modeling_paradigm == "interleaved":
            x_ori = model_outputs.logits
        else:
            raise NotImplementedError

        text_acc = -1
        audio_acc = [-1 for _ in range(self.code_layer)] if self.code_layer > 0 else -1
        if self.metric:
            with torch.no_grad():
                if self.train_config.modeling_paradigm == "parallel" or self.train_config.modeling_paradigm == "serial":
                    preds = torch.argmax(xt, -1)
                    text_acc = compute_accuracy(preds.detach()[:, :-1], text_labels.detach()[:, 1:], ignore_label=-100)

                    preds_audio = [torch.argmax(xa[i], -1) for i in range(self.code_layer)]
                    audio_acc = [compute_accuracy(preds_audio[i].detach()[:, :-1], audio_labels[:, i, 1:], ignore_label=-100) for i in range(self.code_layer)]
                elif self.train_config.modeling_paradigm == "interleaved":
                    preds_start_idx = (text_labels != -100).float().argmax(dim=1)
                    preds = torch.argmax(x_ori, -1)
                    text_preds, text_labels, audio_preds, audio_labels = self.extract_interleaved_tokens(preds, text_labels, preds_start_idx) 

                    text_pad_token = self.model_config.vocab_config.pad_t
                    text_labels[text_labels == text_pad_token] = -100
                    
                    text_acc = compute_accuracy(text_preds.detach()[:, :-1], text_labels.detach()[:, 1:], ignore_label=-100)

                    audio_pad_token = self.model_config.vocab_config.pad_a
                    audio_labels[audio_labels == audio_pad_token] = -100
                    
                    audio_acc = [ compute_accuracy(audio_preds.detach()[:, :-1], audio_labels.detach()[:, 1:], ignore_label=-100)]

                    loss_recorder = None
                else:
                    raise NotImplementedError

        return model_outputs, text_acc, audio_acc, loss_recorder

    def compute_parallel_loss(self, xt, text_labels, xa, audio_labels):
        """
        Compute the parallel loss for text and audio layers.
        """
        text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize
        audio_vocab_size = self.model_config.vocab_config.padded_audio_vocabsize
        layer_loss = [0 for _ in range(self.code_layer+1) ] 
        
        if text_labels is not None:
            text_loss = F.cross_entropy(xt[:, :-1, :].reshape(-1, text_vocab_size), text_labels[:, 1:].reshape(-1), ignore_index=-100)
            layer_loss[self.code_layer] = text_loss
        else:
            text_loss = 0

        total_audio_loss = 0
        single_audio_loss = 0
        for i in range(self.code_layer):
            if audio_labels[:,i] is not None:
                single_audio_loss = F.cross_entropy(xa[i][:, :-1, :].reshape(-1, audio_vocab_size), audio_labels[:, i, 1:].reshape(-1), ignore_index=-100)
                layer_loss[i] = single_audio_loss
                total_audio_loss += single_audio_loss

        total_loss = (text_loss + total_audio_loss) / (self.code_layer+1)
        return total_loss, layer_loss


    @torch.no_grad()
    def generate(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs,
                ):
        kwargs["inference_mode"] = True

        inputs_embeds, attention_mask = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        max_new_tokens = kwargs.get("max_new_tokens", 360)
        generated_ids = [torch.zeros((max_new_tokens,), dtype=torch.long, device=input_ids.device) for _ in range(self.code_layer + 1)]
        current_input_text = None
        current_audio_tokens = [None for _ in range(self.code_layer)]
        past_key_values = None

        text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize
        audio_vocab_size = self.model_config.vocab_config.padded_audio_vocabsize

        num_latency_tokens = kwargs.get("num_latency_tokens", 0)
        text_repetition_penalty = kwargs.get("text_repetition_penalty", 1.0)
        audio_repetition_penalty = kwargs.get("audio_repetition_penalty", 1.0)
        decode_text_only = kwargs.get("decode_text_only", False)
        upsampling_factor = kwargs.get("upsampling_factor", 1)
        do_layershift = kwargs.get("do_layershift", True)
        if do_layershift:
            layershift = layer_shift
        else:
            layershift = simple_shift

        pad_t = self.model_config.vocab_config.pad_t
        pad_a = self.model_config.vocab_config.pad_a
        eot = self.model_config.vocab_config.eot
        eoa = self.model_config.vocab_config.eoa

        text_end = False
        audio_end = False

        if self.train_config.modeling_paradigm == "interleaved":
            model_outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens,
                num_beams=kwargs.get("num_beams", 4),
                do_sample=kwargs.get("do_sample", False),
                min_length=kwargs.get("min_length", 1),
                top_p=kwargs.get("top_p", 1.0),
                repetition_penalty=text_repetition_penalty,
                length_penalty=kwargs.get("length_penalty", 1.0),
                temperature=kwargs.get("temperature", 1.0),
                attention_mask=attention_mask,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=layershift(eoa, 0),
                pad_token_id=self.tokenizer.pad_token_id
            )
            model_outputs = self.process_interleaved_output(model_outputs)
            return model_outputs

        for step in tqdm(range(max_new_tokens), desc="Generating"):
            if current_input_text is not None:
                audio_tokens = torch.cat([layershift(current_audio_tokens[i], i).unsqueeze(1) for i in range(self.code_layer)], dim=1)
                combined_input_ids = torch.cat([audio_tokens, current_input_text.unsqueeze(1)], dim=1)
                if self.train_config.use_peft:
                    inputs_embeds = self.llm.model.model.embed_tokens(combined_input_ids)
                else:
                    inputs_embeds = self.llm.model.embed_tokens(combined_input_ids)
                inputs_embeds = torch.mean(inputs_embeds, dim=1).unsqueeze(1)
            
            outputs = self.llm(
                inputs_embeds=inputs_embeds,                  # [btz, seq_len / 1, emb_dim]
                attention_mask=attention_mask,                # single sample, no need for attention mask
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            logits = outputs.logits[0]                      # batch size is 1
            past_key_values = outputs.past_key_values       # Update past_key_values for the next step

            # Split logits into text and audio layers based on vocab size
            xt_logits = logits[..., :text_vocab_size]
            if self.group_decode_adapter is not None:
                xa_logits = self.group_decode_adapter(logits[..., text_vocab_size:])
                xa_logits = [xa_logits[..., i * audio_vocab_size : (i + 1) * audio_vocab_size] for i in range(self.code_layer)]
            else:
                xa_logits = [logits[..., text_vocab_size + audio_vocab_size * i : text_vocab_size + audio_vocab_size * (i + 1)] for i in range(self.code_layer)]

            # Apply repetition penalty to the logits
            xt_logits = self.repetition_penalty(xt_logits, generated_ids[self.code_layer][:step], text_repetition_penalty)
            for i in range(self.code_layer):
                xa_logits[i] = self.repetition_penalty(xa_logits[i], generated_ids[i][:step], audio_repetition_penalty)

            if not text_end:
                next_token_text = self.sample_next_token(xt_logits[-1, :], **kwargs)
            else:
                next_token_text = torch.tensor([pad_t], device=input_ids.device)
            
            next_tokens_audio = []
            for i in range(self.code_layer):
                if not audio_end and not decode_text_only and num_latency_tokens <= step:
                    next_token_audio = self.sample_next_token(xa_logits[i][-1, :], **kwargs)
                else:
                    next_token_audio = torch.full((input_ids.size(0),), pad_a, device=input_ids.device)
                next_tokens_audio.append(next_token_audio)

            if eoa in next_tokens_audio or decode_text_only:
                audio_end = True
            if next_token_text == eot:
                text_end = True
            
            # Update input_ids for the next step
            current_input_text = next_token_text
            for i in range(self.code_layer):
                current_audio_tokens[i] = next_tokens_audio[i]

            attention_mask = torch.cat([attention_mask, torch.ones((input_ids.size(0), 1), device=input_ids.device)], dim=1)

            # Append generated tokens to the tensor
            for i in range(self.code_layer):
                generated_ids[i][step] = next_tokens_audio[i]  # Audio layers
            generated_ids[self.code_layer][step] = next_token_text  # Text layer

            if self.model_config.use_text_stream:
                if audio_end and text_end:
                    for i in range(self.code_layer):
                        generated_ids[i] = generated_ids[i][:step+1]
                    break       
            else:
                if audio_end:
                    for i in range(self.code_layer):
                        generated_ids[i] = generated_ids[i][:step+1]
                    break     

        # Concatenate the generated tokens to form the complete sequence
        text_tokens = generated_ids[self.code_layer]
        generated_ids[self.code_layer] = text_tokens[: (text_tokens == eot).nonzero(as_tuple=True)[0][0]] if eot in text_tokens else text_tokens

        if eoa in generated_ids[self.code_layer - 1] and do_layershift:
            end_ids = (generated_ids[self.code_layer - 1] == eoa).nonzero(as_tuple=True)[0][0]
            for i in range(self.code_layer):
                audio_tokens = generated_ids[i]
                generated_ids[i] = audio_tokens[:end_ids]

        if upsampling_factor > 1:
            generated_ids[self.code_layer] = generated_ids[self.code_layer][::upsampling_factor]
            
        return generated_ids


    @torch.no_grad()
    def serial_generate(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs,
                ):
        kwargs["inference_mode"] = True

        inputs_embeds, attention_mask = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        max_new_tokens = kwargs.get("max_new_tokens", 360)
        generated_ids = [torch.zeros((max_new_tokens,), dtype=torch.long, device=input_ids.device) for _ in range(self.code_layer )]
        current_tokens = [None for _ in range(self.code_layer)]
        past_key_values = None

        text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize
        audio_vocab_size = self.model_config.vocab_config.padded_audio_vocabsize
        
        text_repetition_penalty = kwargs.get("text_repetition_penalty", 1.0)
        audio_repetition_penalty = kwargs.get("audio_repetition_penalty", 1.0)
        decode_text_only = kwargs.get("decode_text_only", False)
        do_layershift = kwargs.get("do_layershift", True)
        if do_layershift:
            layershift = layer_shift
        else:
            layershift = simple_shift
        eot = self.model_config.vocab_config.eot
        eoa = self.model_config.vocab_config.eoa

        text_end = False     # Track whether text generation has ended
        audio_end = False    # Track whether audio generation has ended

        for step in tqdm(range(max_new_tokens), desc="Generating"):
            if None not in current_tokens:
                if text_end:
                    if current_tokens[0].item() == eot:
                        combined_input_ids = torch.cat([ current_tokens[i].unsqueeze(1) for i in range(self.code_layer)], dim=1)
                        text_step = step
                    else:
                        combined_input_ids = torch.cat([layershift(current_tokens[i], i).unsqueeze(1) for i in range(self.code_layer)], dim=1)
                else:
                    combined_input_ids = torch.cat([ current_tokens[i].unsqueeze(1) for i in range(self.code_layer)], dim=1)
                if self.train_config.use_peft:
                    inputs_embeds = self.llm.model.model.embed_tokens(combined_input_ids)
                else:
                    inputs_embeds = self.llm.model.embed_tokens(combined_input_ids)
                inputs_embeds = torch.mean(inputs_embeds, dim=1).unsqueeze(1)
            
            outputs = self.llm(
                inputs_embeds=inputs_embeds,                  # [btz, seq_len / 1, emb_dim]
                attention_mask=attention_mask,                # single sample, no need for attention mask
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            logits = outputs.logits[0]                      # batch size is 1
            past_key_values = outputs.past_key_values       # Update past_key_values for the next step

            if not text_end:
                xt_logits = logits[..., :text_vocab_size]
                xt_logits = self.repetition_penalty(xt_logits, generated_ids[0][:step], text_repetition_penalty)
                next_token_text = self.sample_next_token(xt_logits[-1, :], **kwargs)

                if next_token_text == eot:
                    text_end = True
                
                for i in range(self.code_layer):
                    current_tokens[i] = next_token_text

                attention_mask = torch.cat([attention_mask, torch.ones((input_ids.size(0), 1), device=input_ids.device)], dim=1)

                for i in range(self.code_layer):
                    generated_ids[i][step] = next_token_text
                
            else:
                if self.group_decode_adapter is not None:
                    xa_logits = self.group_decode_adapter(logits[..., text_vocab_size:])
                    xa_logits = [xa_logits[..., i * audio_vocab_size : (i + 1) * audio_vocab_size] for i in range(self.code_layer)]
                else:
                    xa_logits = [logits[..., text_vocab_size + audio_vocab_size * i : text_vocab_size + audio_vocab_size * (i + 1)] for i in range(self.code_layer)]

                for i in range(self.code_layer):
                    xa_logits[i] = self.repetition_penalty(xa_logits[i], generated_ids[i][text_step:step], audio_repetition_penalty)

                next_tokens_audio = []
                for i in range(self.code_layer):
                    next_token_audio = self.sample_next_token(xa_logits[i][-1, :], **kwargs)
                    next_tokens_audio.append(next_token_audio)
                
                if eoa in next_tokens_audio or decode_text_only:
                    audio_end = True

                for i in range(self.code_layer):
                    current_tokens[i] = next_tokens_audio[i]

                attention_mask = torch.cat([attention_mask, torch.ones((input_ids.size(0), 1), device=input_ids.device)], dim=1)
               
                for i in range(self.code_layer):
                    generated_ids[i][step] = next_tokens_audio[i]  # Audio layers

                if audio_end:
                    for i in range(self.code_layer):
                        generated_ids[i] = generated_ids[i][:step+1]
                    break            

        text_tokens = generated_ids[0]
        if eot in text_tokens:
            end_text_id = (text_tokens == eot).nonzero(as_tuple=True)[0][0]
            generated_ids.append(text_tokens[: end_text_id ] if eot in text_tokens else text_tokens)
            for i in range(self.code_layer):
                generated_ids[i] = generated_ids[i][ end_text_id+1: ]
        else:
            generated_ids.append(text_tokens)
            for i in range(self.code_layer):
                generated_ids[i] = []     

        return generated_ids
    
    @torch.no_grad()
    def sample_next_token(self, logits, **kwargs):
        """
        Generate the next token based on the model output logits.
        Supports both greedy decoding, top-k sampling, and top-p (nucleus) sampling.
        """
        do_sample = kwargs.get("do_sample", False)
        temperature = kwargs.get("temperature", 1.0)
        top_k = kwargs.get("top_k", 0)
        top_p = kwargs.get("top_p", 1.0)
        num_samples = kwargs.get("num_samples", 1)

        # Adjust logits with temperature
        logits = logits.squeeze(0)
        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))  # Make sure top_k is within the vocab size
            values, indices = torch.topk(logits, top_k)
            logits[logits < values[..., [-1]]] = -float('Inf')  # Filter tokens not in top_k

        # Top-p filtering (nucleus sampling)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float('Inf')

        if do_sample:
            # Perform sampling
            return torch.multinomial(F.softmax(logits, dim=-1), num_samples=num_samples)
        else:
            # Greedy decoding (argmax)
            return torch.argmax(logits, dim=-1, keepdim=True)

    def repetition_penalty(self, logits, generated_ids, repetition_penalty):
        """
        Apply repetition penalty to the logits.
        """
        if repetition_penalty == 1.0:
            return logits

        # Gather the logits for generated_ids
        score = torch.gather(logits, -1, generated_ids.unsqueeze(0))

        # Apply penalty
        score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)

        # Scatter the updated scores back into logits
        logits.scatter_(-1, generated_ids.unsqueeze(0), score)

        return logits
        
    def extract_interleaved_tokens(self, preds, text_labels, preds_start_idx):
        """
        Extract predictions and labels in interleaved mode.
        """
        interleaved_text_num = self.train_config.interleaved_text_token_num
        interleaved_audio_num = self.train_config.interleaved_audio_token_num
        
        text_preds_batch = []
        text_labels_batch = []
        audio_preds_batch = []
        audio_labels_batch = []

        for i in range(preds.size(0)):
            text_preds = []
            text_labels_list = []
            audio_preds = []
            audio_labels = []

            start_idx = preds_start_idx[i].item()
            total_length = preds.size(1)
            idx = start_idx
            
            while idx < total_length:
                if idx + interleaved_text_num <= total_length:
                    text_preds.append(preds[i, idx:idx + interleaved_text_num].unsqueeze(0))
                    text_labels_list.append(text_labels[i, idx:idx + interleaved_text_num].unsqueeze(0))
                idx += interleaved_text_num
                
                if idx + interleaved_audio_num <= total_length:
                    audio_preds.append(preds[i, idx:idx + interleaved_audio_num].unsqueeze(0))
                    audio_labels.append(text_labels[i, idx:idx + interleaved_audio_num].unsqueeze(0))
                idx += interleaved_audio_num
            
            text_preds_batch.append(torch.cat(text_preds, dim=1))
            text_labels_batch.append(torch.cat(text_labels_list, dim=1))
            audio_preds_batch.append(torch.cat(audio_preds, dim=1))
            audio_labels_batch.append(torch.cat(audio_labels, dim=1))

        text_preds = torch.cat(text_preds_batch, dim=0) if text_preds_batch else torch.empty(0, interleaved_text_num)
        text_labels = torch.cat(text_labels_batch, dim=0) if text_labels_batch else torch.empty(0, interleaved_text_num)
        audio_preds = torch.cat(audio_preds_batch, dim=0) if audio_preds_batch else torch.empty(0, interleaved_audio_num)
        audio_labels = torch.cat(audio_labels_batch, dim=0) if audio_labels_batch else torch.empty(0, interleaved_audio_num)
        
        return text_preds, text_labels, audio_preds, audio_labels
        
    def process_interleaved_output(self, model_outputs):
        """
        Parse the interleaved generation results and separate tokens into audio and text.
        """
        batch_size, seq_len = model_outputs.shape
        interleaved_audio_token_num = self.train_config.interleaved_audio_token_num
        interleaved_text_token_num = self.train_config.interleaved_text_token_num
        audio_shift = self.model_config.vocab_config.padded_text_vocabsize

        audio_tokens, text_tokens = [], []

        for i in range(batch_size):
            current_audio, current_text = [], []
            sequence = model_outputs[i]

            idx = 0
            while idx < seq_len:
                text_chunk = sequence[idx: idx + interleaved_text_token_num]
                current_text.append(text_chunk)
                idx += interleaved_text_token_num

                if idx < seq_len:
                    audio_chunk = sequence[idx: idx + interleaved_audio_token_num] - audio_shift
                    current_audio.append(audio_chunk)
                    idx += interleaved_audio_token_num

            audio_tokens.append(torch.cat(current_audio) if current_audio else torch.tensor([], device=model_outputs.device))
            text_tokens.append(torch.cat(current_text) if current_text else torch.tensor([], device=model_outputs.device))

        audio_tokens = torch.stack(audio_tokens) if audio_tokens else torch.empty((batch_size, 0), device=model_outputs.device)
        text_tokens = torch.stack(text_tokens) if text_tokens else torch.empty((batch_size, 0), device=model_outputs.device)

        return {
            "audio": audio_tokens,
            "text": text_tokens.squeeze(0),
        }


# Register the model for auto discovery
try:
    from transformers import AutoConfig, AutoModelForCausalLM
    
    # Register the configuration
    AutoConfig.register("slam_tts", SlamTTSConfig)
    
    # Register the model
    AutoModelForCausalLM.register(SlamTTSConfig, SlamTTSForCausalLM)
    
    logger.info("SlamTTS model registered with transformers AutoModel")
except ImportError:
    logger.warning("Could not register SlamTTS with transformers AutoModel")
except Exception as e:
    logger.warning(f"Error registering SlamTTS with transformers: {e}")
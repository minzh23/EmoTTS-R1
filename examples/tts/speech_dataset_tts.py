import re
import json
import copy
import torch
import numpy as np
from utils.codec_utils import layershift, simple_shift, get_single_layer_answer_token, get_group_answer_token

class SpeechDatasetJsonl(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_config,
                 tokenizer=None,
                 split='train',
                 ):
        super().__init__()
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        
        self.IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        self.use_emo = dataset_config.get("use_emo", False)
        self.en_dataset = dataset_config.get("en_dataset", False)
        self.prompt_template = "<SYSTEM>: {}\n "
        self.answer_template = "{}"
        self.inference_mode = dataset_config.get("inference_mode", False)
        self.seed = dataset_config.get("seed", 42)
        self.modeling_paradigm = dataset_config.get("modeling_paradigm", "parallel")
        self.interleaved_text_token_num = dataset_config.get("interleaved_text_token_num", 12)
        self.interleaved_audio_token_num = dataset_config.get("interleaved_audio_token_num", 36)
        assert self.modeling_paradigm in ["parallel", "interleaved", "serial"], "modeling_paradigm must be one of [parallel, interleaved]"

        # vocab config
        self.vocab_config = dataset_config.get("vocab_config", None)
        self.text_vocabsize = self.vocab_config.text_vocabsize
        self.text_specialtokens = self.vocab_config.text_specialtokens
        self.audio_vocabsize = self.vocab_config.audio_vocabsize
        self.audio_specialtokens = self.vocab_config.audio_specialtokens
        self.padded_text_vocabsize = self.vocab_config.padded_text_vocabsize
        self.padded_audio_vocabsize = self.vocab_config.padded_audio_vocabsize
        self.total_vocabsize = self.vocab_config.total_vocabsize
        self._eot = self.vocab_config.eot
        self._pad_t = self.vocab_config.pad_t
        self._input_t = self.vocab_config.input_t
        self._answer_t = self.vocab_config.answer_t
        self._asr = self.vocab_config.asr
        self._eoa = self.vocab_config.eoa
        self._pad_a = self.vocab_config.pad_a
        self._input_a = self.vocab_config.input_a
        self._answer_a = self.vocab_config.answer_a
        self._split = self.vocab_config.split
        self.code_layer = self.vocab_config.code_layer
        self.special_token_a = self._answer_a
        self.special_token_t = self._answer_t
        self.num_latency_tokens = dataset_config.get("num_latency_tokens", 0)

        # layershift config
        self.do_layershift = dataset_config.get("do_layershift", True)
        if self.do_layershift:
            self.layershift = layershift
        else:
            self.layershift = simple_shift
        
        self.data_list = []
        if split == "train":
            with open(dataset_config.train_data_path, encoding='utf-8') as fin:
                for line in fin:
                    data_dict = json.loads(line.strip())
                    self.data_list.append(data_dict)
        else:
            with open(dataset_config.val_data_path, encoding='utf-8') as fin:
                for line in fin:
                    data_dict = json.loads(line.strip())
                    self.data_list.append(data_dict)

    def get_source_len(self, data_dict):
        return data_dict["source_len"]

    def get_target_len(self, data_dict):
        return data_dict["target_len"] if "target_len" in data_dict else 0
    
    def __len__(self):
        return len(self.data_list)

    def extract_audio_feature(self, audio_path):
        audio_tokens = audio_path
        if self.code_layer <= 1:
            audio_res, audio_length = get_single_layer_answer_token(audio_tokens, self.num_latency_tokens, self._pad_a, self._eoa)
        else:
            audio_res, audio_length = get_group_answer_token(audio_tokens, self.num_latency_tokens, self._pad_a, self._eoa, self.code_layer)
        return audio_res, audio_length

    def get_input_ids(self, length, special_token_a, special_token_t):
        input_ids = []
        if self.code_layer == 0:
            input_ids_item = []
            input_ids_item.append(self.layershift(self._input_a, 0))
            input_ids_item += [self.layershift(self._pad_a, 0)] * length
            input_ids_item += [(self.layershift(self._eoa, 0)), self.layershift(special_token_a, 0)]
            input_ids = torch.tensor(input_ids_item).unsqueeze(0).unsqueeze(0)
            return input_ids

        for i in range(self.code_layer):
            input_ids_item = []
            input_ids_item.append(self.layershift(self._input_a, i))
            input_ids_item += [self.layershift(self._pad_a, i)] * length
            input_ids_item += [(self.layershift(self._eoa, i)), self.layershift(special_token_a, i)]
            input_ids.append(torch.tensor(input_ids_item).unsqueeze(0))
        input_id_T = torch.tensor([self._input_t] + [self._pad_t] * length + [self._eot, special_token_t])
        input_ids.append(input_id_T.unsqueeze(0))
        return input_ids

    def get_padded_input(self, text_input_idx, text_index_length):
        padded_input = []
        for i in range(self.code_layer):
            padded_input_item = [self.layershift(self._pad_a, i)] * text_index_length
            padded_input.append(torch.tensor(padded_input_item).unsqueeze(0))
        
        final_layer_input = torch.tensor(text_input_idx)
        padded_input.append(final_layer_input.unsqueeze(0))
        return padded_input

    def get_answer_ids(self, length):
        answer_ids = []
        for i in range(self.code_layer):
            answer_ids_item = []
            answer_ids_item += [self.layershift(self._pad_a, i)] * length
            answer_ids.append(torch.tensor(answer_ids_item).unsqueeze(0))
        answer_id_T = torch.tensor([self._pad_t] * length)
        answer_ids.append(answer_id_T.unsqueeze(0))
        return answer_ids

    def pad_interleaved_chunks(self, answer_text_ids, target_audio):
        audio_chunk_num = (len(target_audio) + self.interleaved_audio_token_num - 1) // self.interleaved_audio_token_num
        text_chunk_num = (len(answer_text_ids) + self.interleaved_text_token_num - 1) // self.interleaved_text_token_num

        padding_needed_text = self.interleaved_text_token_num * text_chunk_num - len(answer_text_ids)
        if padding_needed_text > 0:
            pad_tensor_text = torch.full((padding_needed_text,), self._pad_t, dtype=answer_text_ids.dtype)
            answer_text_ids = torch.cat([answer_text_ids, pad_tensor_text])

        padding_needed_audio = self.interleaved_audio_token_num * audio_chunk_num - len(target_audio)
        if padding_needed_audio > 0:
            pad_tensor_audio = torch.full((padding_needed_audio,), self._pad_a, dtype=target_audio.dtype)
            target_audio = torch.cat([target_audio, pad_tensor_audio])

        if audio_chunk_num >= text_chunk_num:
            chunk_diff = audio_chunk_num - text_chunk_num
            pad_tensor_text = torch.full((self.interleaved_text_token_num * chunk_diff,), self._pad_t, dtype=answer_text_ids.dtype)
            answer_text_ids = torch.cat([answer_text_ids, pad_tensor_text])
        else:
            chunk_diff = text_chunk_num - audio_chunk_num
            pad_tensor_audio = torch.full((self.interleaved_audio_token_num * chunk_diff,), self._pad_a, dtype=target_audio.dtype)
            target_audio = torch.cat([target_audio, pad_tensor_audio])

        return answer_text_ids, target_audio

    def interleave_chunks(self, answer_text_ids, target_audio):
        interleaved_tokens = []
        text_chunk_size = self.interleaved_text_token_num
        audio_chunk_size = self.interleaved_audio_token_num

        num_chunks = max(
            len(answer_text_ids) // text_chunk_size, 
            len(target_audio) // audio_chunk_size
        )

        for i in range(num_chunks):
            text_chunk = answer_text_ids[i * text_chunk_size:(i + 1) * text_chunk_size]
            audio_chunk = target_audio[i * audio_chunk_size:(i + 1) * audio_chunk_size]

            interleaved_tokens.extend(text_chunk)
            interleaved_tokens.extend(audio_chunk)

        return interleaved_tokens

    def __getitem__(self, index):
        data_dict = self.data_list[index]
        example_ids = None
        key = None
        audio_length = 0
        target_audio_length = 0

        source_audio = data_dict.get("source_wav", None)
        target_audio = data_dict.get("answer_cosyvoice_speech_token", None)
        source_text = data_dict.get("source_text", None)
        target_text = data_dict.get("target_text", None)
        key = data_dict.get("key", None)
        neutral_speaker_wav = data_dict.get("neutral_speaker_wav", None)
        emotion_text_prompt = data_dict.get("emotion_text_prompt", None)

        target_audio, target_audio_length = self.extract_audio_feature(target_audio)

        prompt="Say this sentence. "
        if self.use_emo:
            if "emotion_text_prompt" in data_dict:
                emotion_text_prompt = data_dict.get("emotion_text_prompt" , None)
                emotion_text_prompt = re.sub(r'[。！？\.,!\?]$', '', emotion_text_prompt)
                emotion_text_prompt = re.sub(r'\.(?=.)', ',', emotion_text_prompt)
                prompt = "Say this sentence with emotion of {}. ".format(emotion_text_prompt)
  
        prompt = self.prompt_template.format(prompt) #'<SYSTEM>: 请说这句话. \n '
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_ids = [self._input_t] + prompt_ids + [self._eot]
        prompt_length = len(prompt_ids)
        prompt_ids = self.get_padded_input(prompt_ids, prompt_length)

        target_text_ids = self.tokenizer.encode(source_text)
        target_text_length = len(target_text_ids)
        target_text_ids = torch.tensor(target_text_ids, dtype=torch.int64)
        example_ids = self.get_input_ids(target_text_length, self.special_token_a, self.special_token_t) # <prompt> <bos> <text> <eos> <task>
        text_layer = example_ids[self.code_layer]
        text_layer = torch.cat((text_layer[:,:1], target_text_ids.unsqueeze(0), text_layer[:,-2:]), dim=1)
        example_ids[self.code_layer] = text_layer
        example_ids = [torch.cat((prompt_ids[i], example_ids[i]), dim = 1) for i in range(self.code_layer + 1)]
        if self.modeling_paradigm == "serial":
            only_text_layer = example_ids[self.code_layer]
            example_ids = []
            for i in range(self.code_layer):
                example_ids.append(only_text_layer) 
        input_length = target_text_length
        
        if self.inference_mode:
            example_mask = example_ids[0][0].ge(-1)  # [True,True]
            example_ids = torch.stack(example_ids).squeeze() if torch.stack(example_ids).shape[0]!=1 else torch.stack(example_ids).squeeze(0)
            return {
                "input_ids": example_ids,
                "attention_mask": example_mask,
                "input_length": input_length,
                "audio_length": audio_length,
                "target_audio": target_audio,
                "target_audio_length": target_audio_length,
                "key": key,
                "source_text": source_text,
                "target_text": target_text,
                "prompt_length": prompt_length,
                "neutral_speaker_wav": neutral_speaker_wav,
                "emotion_text_prompt": emotion_text_prompt,
            }

        answer_text = self.answer_template.format(target_text)
        answer_text_ids = self.tokenizer.encode(answer_text)  # [answer]
        answer_text_ids.append(self._eot) # [answer,eos]
        answer_text_ids = torch.tensor(answer_text_ids, dtype=torch.int64)

        if self.modeling_paradigm == "parallel":
            answer_length = max(len(answer_text_ids), target_audio_length)
            answer_ids = self.get_answer_ids(answer_length)                 # NOTE: somtimes answer_text_ids is longer than target_audio_length 
            if self.dataset_config.get("use_text_stream","True"):
                answer_ids[self.code_layer] = torch.cat((answer_text_ids.unsqueeze(0), answer_ids[self.code_layer][:,len(answer_text_ids):]),dim=1)     # [answer_text,eos]
            text_padding_length = target_audio_length - len(answer_text_ids)

            labels_ids = copy.deepcopy(answer_ids)
            ori_example_ids = copy.deepcopy(example_ids)
            
            if target_audio is not None:    
                for i in range(self.code_layer):
                    labels_ids[i] = torch.cat((target_audio[i].unsqueeze(0), answer_ids[i][:,target_audio_length:]), dim=1)
                    answer_ids[i] = torch.cat((self.layershift(target_audio[i], i).unsqueeze(0), labels_ids[i][:,target_audio_length:]), dim=1)
            for i in range(self.code_layer + 1):
                example_ids[i] = torch.cat((ori_example_ids[i], answer_ids[i]), dim=1)  # [prompt,audio,answer,eos]
                labels_ids[i] = torch.cat((ori_example_ids[i], labels_ids[i]), dim=1)

            example_ids = torch.stack(example_ids).squeeze()
            labels_ids = torch.stack(labels_ids).squeeze()
            labels_ids[:,:input_length + prompt_length + 3] = -1  # [-1,-1,answer,eos]; NOTE: here 3 include <bos> <eos> <ans_t>

            if text_padding_length > 0:
                labels_ids[self.code_layer,-text_padding_length:] = -1   # [-1,-1,answer_text,eos,-1]
            else:
                audio_padding_length = -text_padding_length
                labels_ids[:self.code_layer,-audio_padding_length:] = -1  # [-1,-1,answer_text,eos,-1]

        elif self.modeling_paradigm == "serial":
            answer_length = len(answer_text_ids) + target_audio_length
            answer_ids = torch.full((self.code_layer, answer_length), -1)

            labels_ids = copy.deepcopy(answer_ids)
            ori_example_ids = copy.deepcopy(example_ids)

            for i in range(self.code_layer):
                labels_ids[i] = torch.cat( (answer_text_ids.unsqueeze(0), target_audio[i].unsqueeze(0)), dim=1 )
                answer_ids[i] = torch.cat( (answer_text_ids.unsqueeze(0), self.layershift(target_audio[i], i).unsqueeze(0)), dim=1 )

            example_ids=[]
            whole_labels_ids=[]
            for i in range(self.code_layer):
                example_ids.append(torch.cat((ori_example_ids[0], answer_ids[i].unsqueeze(0)),dim=1))  # [prompt,audio,answer,eos]
                whole_labels_ids.append(torch.cat((ori_example_ids[0], labels_ids[i].unsqueeze(0)),dim=1))

            example_ids = torch.stack(example_ids).squeeze(1)
            labels_ids = torch.stack(whole_labels_ids).squeeze(1)
            labels_ids[:,:input_length + prompt_length + 3] = -1  # [-1,-1,answer,eos]; NOTE: here 3 include <bos> <eos> <ans_t>
    
        elif self.modeling_paradigm == "interleaved":
            target_audio = target_audio.squeeze(0)
            example_ids = example_ids[0]
            answer_text_ids, target_audio = self.pad_interleaved_chunks(answer_text_ids, target_audio)
            target_audio_labels = self.layershift(target_audio, 0)
            interleaved_sequence = self.interleave_chunks(answer_text_ids, target_audio_labels)

            example_ids = torch.cat((example_ids, torch.tensor(interleaved_sequence).unsqueeze(0)), dim=1)
            labels_ids = example_ids.clone()
            labels_ids[:,:input_length + prompt_length + 3] = -1  # [-1,-1,answer,eos]; NOTE: here 3 include <bos> <eos> <ans_t>


        example_mask = example_ids[0].ge(-1)  # [True,True,True,True]

        label_mask = labels_ids.ge(0)  # [False,False,True,True]
        labels_ids[~label_mask] = self.IGNORE_INDEX  # [-100,-100,answer,eos]

        return {
            "input_ids": example_ids,
            "labels": labels_ids,
            "attention_mask": example_mask,
            "input_length": input_length,
            "audio_length": audio_length,
            "target_audio": target_audio,
            "target_audio_length": target_audio_length,
            "key": key,
            "source_text": source_text,
            "target_text": target_text,
            "prompt_length": prompt_length,
        }

    def pad(self, sequence, max_length, padding_idx=0):
        if isinstance(sequence, (int, list, tuple)):
            if len(sequence) < max_length:
                sequence = sequence + [padding_idx] * (max_length - len(sequence))
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, torch.Tensor):
            if len(sequence) < max_length:
                sequence = torch.cat(
                    (sequence, torch.full(([max_length - len(sequence)] + list(sequence.size())[1:]), padding_idx)))
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, np.ndarray):
            if len(sequence) < max_length:
                sequence = np.concatenate(
                    (sequence, np.full((max_length - len(sequence),) + sequence.shape[1:], padding_idx)))
            else:
                sequence = sequence[:max_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence
        
    @classmethod
    def padding(cls, sequence, padding_length, padding_idx=0, padding_side="right"):
        if isinstance(sequence, (int, list, tuple)):
            if padding_length >= 0:
                sequence = sequence + [padding_idx] * padding_length
            else:
                sequence = sequence[:padding_length]
        elif isinstance(sequence, torch.Tensor):
            if sequence.ndimension() == 2:
                if padding_length >= 0:
                    padding_tensor = torch.full((sequence.size(0), padding_length), padding_idx, dtype=sequence.dtype)
                    if padding_side == "left":
                        sequence = torch.cat((padding_tensor, sequence), dim=1)
                    else:
                        sequence = torch.cat((sequence, padding_tensor), dim=1)
                else:
                    sequence = sequence[:, :padding_length]
            else:
                if padding_length >= 0:
                    if padding_side == "left":
                        sequence = torch.cat((torch.full(([padding_length] + list(sequence.size())[1:]), padding_idx), sequence))
                    else:
                        sequence = torch.cat((sequence, torch.full(([padding_length] + list(sequence.size())[1:]), padding_idx)))
                else:
                    sequence = sequence[:padding_length]
        elif isinstance(sequence, np.ndarray):
            if padding_length >= 0:
                sequence = np.concatenate(
                    (sequence, np.full((padding_length,) + sequence.shape[1:], padding_idx)))
            else:
                sequence = sequence[:padding_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence

    def collator(self, samples):
        assert samples is not None 
        input_prompt_lengths = [s["input_length"] + s['prompt_length'] + 3 for s in samples]
        input_answer_lengths = [len(s["input_ids"][0]) - s["input_length"] - s['prompt_length'] - 3 for s in samples]

        input_prompt_max_length = max(input_prompt_lengths)
        input_answer_max_length = max(input_answer_lengths)
        
        # NOTE: left padding for prompt and right padding for answer 
        input_ids = torch.stack([
            self.padding(
                self.padding(samples[index]["input_ids"], input_prompt_max_length - input_prompt_lengths[index], self.tokenizer.pad_token_id, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], self.tokenizer.pad_token_id
            ) for index in range(len(samples))
        ])

        attention_mask = torch.stack([
            self.padding(
                self.padding(samples[index]["attention_mask"], input_prompt_max_length - input_prompt_lengths[index], False, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], False
            ) for index in range(len(samples))
        ])

        input_length = torch.tensor([s["input_length"] for s in samples])
        audio_length = torch.tensor([s["audio_length"] for s in samples])
        audio_raw = None
        audio_mask = None

        modality_mask = torch.zeros_like(attention_mask)
        for index in range(len(samples)):
            padding_left = input_prompt_max_length - input_prompt_lengths[index] + 1 + samples[index]['prompt_length'] # +1 for <bos>
            modality_mask[index, padding_left:padding_left+samples[index]["audio_length"]] = True

        if self.inference_mode:
            keys = [s['key'] for s in samples]
            target_text = [s['target_text'] for s in samples]
            source_text = [s['source_text'] for s in samples]
            target_audio = [s['target_audio'] for s in samples]
            neutral_speaker_wav = [s['neutral_speaker_wav'] for s in samples]

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "audio": audio_raw,
                "audio_mask": audio_mask,
                "input_length": input_length,
                "audio_length": audio_length,
                "modality_mask": modality_mask,
                "keys": keys,
                "target_texts": target_text,
                "source_texts": source_text,
                "target_audio": target_audio,
                "neutral_speaker_wav": neutral_speaker_wav,
            }
        
        labels = torch.stack([
            self.padding(
                self.padding(samples[index]['labels'], input_prompt_max_length - input_prompt_lengths[index], self.IGNORE_INDEX, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], self.IGNORE_INDEX)
            for index in range(len(samples))
        ])
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "audio": audio_raw,
            "audio_mask": audio_mask,
            "input_length": input_length,
            "audio_length": audio_length,
            "modality_mask": modality_mask,
        }


def get_speech_dataset(dataset_config, tokenizer, split):
    dataset = SpeechDatasetJsonl(dataset_config, tokenizer, split)
    return dataset

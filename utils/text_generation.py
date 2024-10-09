import re
import random
import torch
from openai import OpenAI
from rouge_score import rouge_scorer


def postprocess(model_outputs, input_prompts, pipeline_output=False):
    """
    Compatible with both tokens and strings.
    """
    records = []
    for output, input_prompt in zip(model_outputs, input_prompts):
        prompt_length = len(input_prompt)
        if pipeline_output:
            record = output[0]['generated_text'][prompt_length:]
            record = [{'generated_text': record}]
        else:
            record = output[prompt_length:]
        records.append(record)

    return records


class AgentManager:
    def __init__(self, ppo_trainer, tokenizer, gen_kwargs, system_prompt):
        self.ppo_trainer = ppo_trainer
        self.tokenizer = tokenizer
        self.gen_kwargs = gen_kwargs
        self.system_prompt = system_prompt
        self.query_tensors = None

    def get_response(self, user_inputs, drop_eos=False, other_model_outputs=None, return_custom_prompt_tensor=False, output_length_sampler=None):
        if other_model_outputs is None:
            custom_prompts = [self.system_prompt.format(user_input) for user_input in user_inputs]
        else:
            # assert len(user_inputs) == len(other_model_outputs), "Inputs and model outputs must be of the same length"
            custom_prompts = [self.system_prompt.format(user_input, other_output) for user_input, other_output in zip(user_inputs, other_model_outputs)]


        query_ids = self.tokenizer(custom_prompts)['input_ids']
        self.query_tensors = [torch.tensor(q) for q in query_ids]

        outputs = self.ppo_trainer.generate(self.query_tensors,
                                            batch_size=len(user_inputs),
                                            return_prompt=False,
                                            length_sampler=output_length_sampler,
                                            **self.gen_kwargs)
        if drop_eos:
            outputs = [output[:-1] for output in outputs]

        if return_custom_prompt_tensor:
            return outputs, self.query_tensors
        else:
            return outputs



class TaskManager:
    def __init__(self, generator, gen_kwargs, dataset_name):
        self.generator = generator
        self.gen_kwargs = gen_kwargs
        self.dataset_name = dataset_name

    def _set_prefix_batch(self, prompts, task_prefix):
        """
        Set prefixes for prompts.
        :param prompts: List[str]
        :param task_prefix: str type including '{}'
        :return:
        """
        task_prompt = []
        for prompt in prompts:
            task_prompt.append(task_prefix.format(prompt))
        return task_prompt

    def _set_prefix(self, dataset_name):
        if dataset_name == "ShareGPT":
            self.task_prompts = self._set_prefix_batch(self.context, "Continue the conversation: {}")
        elif dataset_name == "Arxiv":
            self.task_prompts = self._set_prefix_batch(self.context, "{}\n TL;DR:")
        elif dataset_name == "GSM8K":
            raise NotImplementedError
        elif dataset_name == "BBH":
            raise NotImplementedError

    def get_answer(self, context, batch_size=1):
        self.context = context
        self._set_prefix(self.dataset_name)

        outputs = self.generator(self.task_prompts,
                                 batch_size=batch_size,
                                 return_full_text=False,
                                 **self.gen_kwargs)

        return [output[0]['generated_text'] for output in outputs]

class InferenceManager:
    def __init__(self, generator, gen_kwargs, dataset_name):
        self.generator = generator
        self.gen_kwargs = gen_kwargs
        self.dataset_name = dataset_name

    def _set_prefix_batch(self, prompts, task_prefix):
        """
        Set prefixes for prompts.
        :param prompts: List[str]
        :param task_prefix: str type including '{}'
        :return:
        """
        task_prompt = []
        for prompt in prompts:
            task_prompt.append(task_prefix.format(prompt))
        return task_prompt

    def _set_prefix(self, dataset_name):
        if dataset_name == "ShareGPT":
            self.task_prompts = self._set_prefix_batch(self.context, "Continue the conversation: {}")
        elif dataset_name == "Arxiv":
            self.task_prompts = self._set_prefix_batch(self.context, "{}\n TL;DR:")
        elif dataset_name == "GSM8K":
            raise NotImplementedError
        elif dataset_name == "BBH":
            raise NotImplementedError

    def get_answer(self, context, batch_size=1):
        self.context = context
        self._set_prefix(self.dataset_name)

        if isinstance(self.generator, OpenAI):
            outputs = []
            for prompt in self.task_prompts:
                response_obj = self.generator.chat.completions.create(
                    model='gpt-3.5-turbo',
                    messages=[{"role": "user", "content": prompt}],
                    seed=0,)
                outputs.append(response_obj.choices[0].message.content)

        else:
            outputs = self.generator(self.task_prompts,
                                     batch_size=batch_size,
                                     return_full_text=False,
                                     **self.gen_kwargs)
            outputs = [output[0]['generated_text'] for output in outputs]

        return outputs


class Ranker:
    def __init__(self, generator, gen_kwargs, prompt_temp):
        self.generator = generator
        self.gen_kwargs = gen_kwargs
        self.prompt_temp = prompt_temp
        self.targe_model = None

    def _extract_dict_from_text(self, text):
        pattern = r"\{'model':'\w+', 'rank': \d+}"
        extracted_data = re.findall(pattern, text)

        if extracted_data == []:
            print("Error with regular expression: =========================\n{}".format(text))
            pass

        result_list = []
        for item in extracted_data:
            model_rank_dict = eval(item)
            result_list.append(model_rank_dict)
        return result_list

    def _get_rank(self, queries, ref_responses, responses):
        bs = len(queries) if len(queries) < 16 else 16

        # randomly shuffle the order of the models
        if random.random() > 0.5:
            prompts = [self.prompt_temp.format(instruction=query, output_1=response, output_2=ref_response)
                      for query, ref_response, response in zip(queries, ref_responses, responses)]
            self.targe_model = 'model_1'
        else:
            prompts = [self.prompt_temp.format(instruction=query, output_1=ref_response, output_2=response)
                      for query, ref_response, response in zip(queries, ref_responses, responses)]
            self.targe_model = 'model_2'

        if isinstance(self.generator, OpenAI):
            outputs = []
            for prompts in prompts:
                response_obj = self.generator.chat.completions.create(
                    model='gpt-3.5-turbo',
                    messages=[{"role": "user", "content": prompts}],
                    seed=0,)
                outputs.append(response_obj.choices[0].message.content)
        else:
            outputs = self.generator(prompts, batch_size=bs, return_full_text=False, **self.gen_kwargs)
            outputs = [output[0]['generated_text'] for output in outputs]

        outputs = [self._extract_dict_from_text(output) for output in outputs]
        return outputs

    def get_reward(self, queries, ref_responses, responses):
        outputs = self._get_rank(queries, ref_responses, responses)
        rewards = []
        print("targe_model:{}".format(self.targe_model))
        for i, models_list in enumerate(outputs):
            # for model_dict in models_list:
            #     if model_dict['model'] == 'model_2' and model_dict['rank'] == 1:
            #         rewards.append(1.)
            #         break
            # rewards.append(0.)
            print(i, models_list)
            if {'model': self.targe_model, 'rank': 1} in models_list:
                rewards.append(1.)
            else:
                rewards.append(0.)
        return rewards

    # def get_rouge_reward(self, queries, ref_responses, responses):
    #     # calculate the rouge score between ref_responses and responses.
    #     rewards = []
    #     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    #     for ref_res, res in zip(ref_responses, responses):
    #         rouge1 = scorer.score(res, ref_res)['rouge1'].fmeasure
    #         rouge2 = scorer.score(res, ref_res)['rouge2'].fmeasure
    #         rougeL = scorer.score(res, ref_res)['rougeL'].fmeasure
    #         rewards.append((rouge1 + rouge2 + rougeL) / 3)
    #     return rewards
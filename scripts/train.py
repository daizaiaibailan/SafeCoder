import os
import argparse
import sys

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sven.constant import MODEL_DIRS
from sven.trainer import PrefixTrainer, TextPromptTrainer
from sven.utils import set_logging, set_devices


class ResumablePrefixTrainer(PrefixTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.resume_from = getattr(args, 'resume_from', None)

    def load_model(self):
        super().load_model()
        if self.resume_from:
            prefix_file = os.path.join(self.resume_from, 'pytorch_model.bin')
            if os.path.exists(prefix_file):
                state_dict = torch.load(prefix_file, map_location='cpu')
                self.model.prefix_params.load_state_dict(state_dict)
                self.args.logger.info('Resumed prefix params from %s', prefix_file)
            else:
                self.args.logger.warning('Resume file not found: %s', prefix_file)


def main():
    parser = argparse.ArgumentParser(description='SVEN training')
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--pretrain_dir', type=str, default='350m',
                        choices=['350m', '2b', '6b', 'incoder', 'santa'])
    parser.add_argument('--data_dir', type=str, default='data_train_val')
    parser.add_argument('--output_dir', type=str, default='trained')
    parser.add_argument('--model_type', type=str, default='prefix',
                        choices=['prefix', 'text-prompt', 'lm'])
    parser.add_argument('--vul_type', type=str, default=None)
    parser.add_argument('--n_prefix_token', type=int, default=5)
    parser.add_argument('--num_train_epochs', type=int, default=8)
    parser.add_argument('--kl_loss_ratio', type=int, default=1600)
    parser.add_argument('--contrastive_loss_ratio', type=int, default=400)
    parser.add_argument('--lm_loss_ratio', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--max_num_tokens', type=int, default=1024)
    parser.add_argument('--grad_acc_steps', type=int, default=2)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--diff_level', type=str, default='mix',
                        choices=['prog', 'line', 'char', 'mix'])
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--save_epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume from checkpoint directory (e.g., trained/350m-prefix-new/checkpoint-epoch-5)')

    args = parser.parse_args()

    if args.pretrain_dir in MODEL_DIRS:
        args.pretrain_dir = MODEL_DIRS[args.pretrain_dir]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    args.data_dir = os.path.join(root_dir, args.data_dir)
    args.output_dir = os.path.join(root_dir, args.output_dir, args.output_name)

    if args.resume_from and not os.path.isabs(args.resume_from):
        args.resume_from = os.path.join(root_dir, args.resume_from)

    log_file = os.path.join(args.output_dir, 'train.log')
    set_logging(args, log_file)
    set_devices(args)

    if args.model_type == 'prefix':
        trainer = ResumablePrefixTrainer(args)
    elif args.model_type in ('text-prompt', 'lm'):
        trainer = TextPromptTrainer(args)
    else:
        raise ValueError(f'Unknown model_type: {args.model_type}')

    trainer.run()


if __name__ == '__main__':
    main()

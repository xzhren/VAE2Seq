import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--num_sampled', type=int, default=1000)
parser.add_argument('--max_len', type=int, default=150)
parser.add_argument('--word_dropout_rate', type=float, default=0.8)
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--rnn_size', type=int, default=256)
parser.add_argument('--beam_width', type=int, default=5)
parser.add_argument('--clip_norm', type=float, default=5.0)
parser.add_argument('--latent_size', type=int, default=256)
parser.add_argument('--anneal_max', type=float, default=1.0)
parser.add_argument('--anneal_bias', type=int, default=6000)
parser.add_argument('--num_epoch', type=int, default=30)
parser.add_argument('--display_loss_step', type=int, default=50)
parser.add_argument('--display_info_step', type=int, default=1000)

parser.add_argument('--vocab_limit', type=int, default=35000)
parser.add_argument('--model_name', type=str, default="vrae.ckpt")
parser.add_argument('--train_data', type=str, default="./corpus/reddit/train.txt")

parser.add_argument('--cuda', type=int, default=2)
parser.add_argument('--loss_type', type=int, default=0)
parser.add_argument('--exp', type=str, default="vaeseq_0220")
parser.add_argument('--batch_size', type=int, default=64)

args = parser.parse_args()
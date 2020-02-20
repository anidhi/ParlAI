import torch.nn as nn, torch, copy, tqdm, math
from torch.autograd import Variable
import torch.nn.functional as F
use_cuda = torch.cuda.is_available()


def max_out(x):
    # make sure s2 is even and that the input is 2 dimension
    if len(x.size()) == 2:
        s1, s2 = x.size()
        x = x.unsqueeze(1)
        x = x.view(s1, s2 // 2, 2)
        x, _ = torch.max(x, 2)

    elif len(x.size()) == 3:
        s1, s2, s3 = x.size()
        x = x.unsqueeze(1)
        x = x.view(s1, s2, s3 // 2, 2)
        x, _ = torch.max(x, 3)

    return x


class Seq2Seq(nn.Module):
    def __init__(self, options):
        super(Seq2Seq, self).__init__()
        self.base_enc = BaseEncoder(options.vocab_size, options.emb_size, options.ut_hid_size, options)
        self.ses_enc = SessionEncoder(options.ses_hid_size, options.ut_hid_size, options)
        self.dec = Decoder(options)
        
    def forward(self, sample_batch):
        u1, u1_lens, u2, u2_lens, u3, u3_lens = sample_batch[0], sample_batch[1], sample_batch[2], \
        sample_batch[3], sample_batch[4], sample_batch[5]
        if use_cuda:
            u1 = u1.cuda()
            u2 = u2.cuda()
            u3 = u3.cuda()
        o1, o2 = self.base_enc((u1, u1_lens)), self.base_enc((u2, u2_lens))
        qu_seq = torch.cat((o1, o2), 1)
        final_session_o = self.ses_enc(qu_seq)
        preds, lmpreds = self.dec((final_session_o, u3, u3_lens))
        
        return preds, lmpreds

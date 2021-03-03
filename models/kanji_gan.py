from torch import nn

class Generator(nn.Module):
    def __init__(self, input, output_width = 256, gen_width = 64):
        super(Generator, self).__init__()
        self.input = input
        self.output_width = output_width
        self.gen_width = gen_width

        # define generator
        s = self.output_width
        s2, s4, s8, s16, s32 = int(s / 2), int(s / 4), int(s / 8), int(s / 16), int(s / 32)
        model = [
            nn.ReLU(self.input, inplace=False),
            nn.ConvTranspose2d(775, 512, 5, 2, 3, bias=False),
            nn.ConvTranspose2d(512, 512, 5, 2, 3, bias=False),
            nn.ConvTranspose2d(775, 512, 5, 2, 3, bias=False),
            nn.ConvTranspose2d(775, 512, 5, 2, 3, bias=False),
        ]
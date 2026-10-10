#ADDED: new file (FMGAI assignment) - renders the Phase 1 declaration PDF.
"""Build assignment_results/declaration/Phase1_Declaration.pdf.

The numbers quoted for the schedule are read from
assignment_results/schedule_analysis/schedule_summary.json, so run
assignment/schedule_analysis.py first.
"""
import json
import os

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT_DIR = 'assignment_results/declaration'
ANALYSIS = 'assignment_results/schedule_analysis'
COMMIT = '01d27a7cbfa8502be1d8dbd4ee02fcbd5e44389d'
UPSTREAM = 'https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement'
BLOB = UPSTREAM + '/blob/' + COMMIT
FORK = 'https://github.com/rikin-patel/sr3_diffusion'

# DejaVu ships with matplotlib and has the Greek and subscript glyphs we need.
font_dir = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf')
pdfmetrics.registerFont(TTFont('Body', os.path.join(font_dir, 'DejaVuSerif.ttf')))
pdfmetrics.registerFont(TTFont('Body-Bold', os.path.join(font_dir, 'DejaVuSerif-Bold.ttf')))
pdfmetrics.registerFont(TTFont('Body-Italic', os.path.join(font_dir, 'DejaVuSerif-Italic.ttf')))
pdfmetrics.registerFont(TTFont('Mono', os.path.join(font_dir, 'DejaVuSansMono.ttf')))
pdfmetrics.registerFontFamily('Body', normal='Body', bold='Body-Bold', italic='Body-Italic')

BODY = ParagraphStyle('body', fontName='Body', fontSize=9.6, leading=14, alignment=TA_JUSTIFY, spaceAfter=6)
SMALL = ParagraphStyle('small', parent=BODY, fontSize=8.6, leading=12)
CELL = ParagraphStyle('cell', parent=BODY, fontSize=8.6, leading=11.5, alignment=0, spaceAfter=0)
H1 = ParagraphStyle('h1', fontName='Body-Bold', fontSize=15, leading=19, spaceAfter=4)
H2 = ParagraphStyle('h2', fontName='Body-Bold', fontSize=11.2, leading=15, spaceBefore=9, spaceAfter=4,
                    textColor=colors.HexColor('#1f4e79'))
CAPTION = ParagraphStyle('cap', parent=SMALL, fontName='Body-Italic', alignment=1)

AB = 'ᾱ<sub>t</sub>'          # alpha-bar_t
TH = 'θ<sub>t</sub>'          # theta_t


def link(url, text=None):
    return '<link href="{}" color="#1f4e79"><u>{}</u></link>'.format(url, text or url)


def code(text):
    return '<font name="Mono" size="8.4">{}</font>'.format(text)


def table(rows, widths, header=True):
    data = [[Paragraph(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#9aa5b1')),
             ('VALIGN', (0, 0), (-1, -1), 'TOP'),
             ('LEFTPADDING', (0, 0), (-1, -1), 5), ('RIGHTPADDING', (0, 0), (-1, -1), 5),
             ('TOPPADDING', (0, 0), (-1, -1), 3.5), ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5)]
    if header:
        style.append(('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8eef4')))
    t.setStyle(TableStyle(style))
    return t


def sci(v):
    mant, exp = '{:.3e}'.format(v).split('e')
    return '{} × 10<super>{}</super>'.format(mant, int(exp))


def main():
    with open(os.path.join(ANALYSIS, 'schedule_summary.json')) as f:
        s = json.load(f)
    nat, pw = s['theta_native_float64'], s['theta_power_float64']
    assert nat['strictly_in_open_unit_interval'] and pw['strictly_in_open_unit_interval']
    x = '{:.2f}'.format(s['x'])
    T = s['T']

    os.makedirs(OUT_DIR, exist_ok=True)
    doc = SimpleDocTemplate(os.path.join(OUT_DIR, 'Phase1_Declaration.pdf'), pagesize=A4,
                            leftMargin=2.1 * cm, rightMargin=2.1 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title='Phase 1 Declaration - Noise schedule study on SR3',
                            author='[Group members]')
    W = A4[0] - 4.2 * cm
    e = []

    e.append(Paragraph('Phase 1 Declaration', H1))
    e.append(Paragraph('Foundation Models and Generative AI: fine-tuning a pretrained diffusion model '
                       'under a power-transformed noise schedule', BODY))
    e.append(table([
        ['Group', '[GROUP NUMBER / NAME - to be filled in]'],
        ['Members', '[MEMBER 1 NAME, ROLL NO.]; [MEMBER 2 NAME, ROLL NO.]; [MEMBER 3 NAME, ROLL NO.]'],
        ['Code repository', link(FORK) + ' (our fork of the public SR3 implementation; '
                            'one feature branch per run)'],
    ], [3.2 * cm, W - 3.2 * cm], header=False))

    # 1 ---------------------------------------------------------------
    e.append(Paragraph('1. Image-to-image task', H2))
    e.append(Paragraph(
        'We study <b>8× single-image super-resolution of gastrointestinal endoscopy images, from '
        '16×16 pixels to 128×128 pixels</b>. The input is a 16×16 RGB image obtained by bicubic '
        'downsampling of an endoscopy frame. Following the model\'s own convention, this low-resolution image '
        'is first upsampled back to 128×128 with bicubic interpolation and given to the network as the '
        'conditioning image; the network then generates the 128×128 high-resolution image by iterative '
        'denoising. The target is the original frame resized and centre-cropped to 128×128.', BODY))

    # 2 ---------------------------------------------------------------
    e.append(Paragraph('2. Pretrained model', H2))
    e.append(Paragraph(
        'The model is <b>SR3</b> (Saharia et al., "Image Super-Resolution via Iterative Refinement"). '
        'Paper: ' + link('https://arxiv.org/abs/2104.07636') + '. Google did not release weights for SR3, so we '
        'use the widely used open PyTorch implementation by Janspiry and the checkpoint its authors released '
        'for the 16×16 → 128×128 setting.', BODY))
    e.append(table([
        ['Item', 'Reference'],
        ['Codebase', link(UPSTREAM) + ' (pinned at commit ' + code(COMMIT[:7]) + ')'],
        ['Checkpoint', 'Row "16×16 -> 128×128 on FFHQ-CelebaHQ" of the pretrained-model table, '
                       + link(BLOB + '/README.md#L84', 'README.md, line 84') + '. Files: '
                       + code('I640000_E37_gen.pth') + ' (generator, 97.8 M parameters, saved at iteration 640,000).'],
        ['Model config', link(BLOB + '/config/sr_sr3_16_128.json', 'config/sr_sr3_16_128.json')],
    ], [3.2 * cm, W - 3.2 * cm]))
    e.append(Spacer(1, 4))
    e.append(Paragraph('The same checkpoint is the starting point for Run 0, Run A and Run B.', BODY))

    # 3 ---------------------------------------------------------------
    e.append(Paragraph('3. Dataset', H2))
    e.append(Paragraph(
        '<b>Kvasir-SEG</b> (Jha et al., "Kvasir-SEG: A Segmented Polyp Dataset", MultiMedia Modeling 2020; '
        + link('https://arxiv.org/abs/1911.07069', 'arXiv:1911.07069') + '). Source URL: '
        + link('https://datasets.simula.no/kvasir-seg/') + '. The dataset contains 1,000 colonoscopy images of '
        'polyps together with segmentation masks. We use only the 1,000 images; the masks play no part in a '
        'super-resolution task. The images are split once into 800 training, 100 validation and 100 test '
        'images with a fixed random seed (42), and the three file lists are stored in the repository so that '
        'every run sees exactly the same split.', BODY))

    # 4 ---------------------------------------------------------------
    e.append(Paragraph('4. Evidence that the dataset was not used in pretraining', H2))
    e.append(Paragraph(
        'We follow option (b): we point to where the pretraining data is described and then argue that '
        'Kvasir-SEG lies in a different domain. Because the weights come from the open implementation rather '
        'than from the paper\'s authors, the codebase is the authoritative source for what this particular '
        'checkpoint was trained on. All line numbers refer to the pinned commit ' + code(COMMIT[:7]) + '.', BODY))
    e.append(table([
        ['Source', 'Location', 'What it says'],
        ['Codebase, ' + link(BLOB + '/README.md#L24', 'README.md'), 'line 24',
         'Lists the conditional model as "16×16 -> 128×128 on FFHQ-CelebaHQ".'],
        ['Codebase, ' + link(BLOB + '/README.md#L84', 'README.md'), 'line 84',
         'The released checkpoint we use is labelled "16×16 -> 128×128 on FFHQ-CelebaHQ".'],
        ['Codebase, ' + link(BLOB + '/config/sr_sr3_16_128.json#L16-L19', 'config/sr_sr3_16_128.json'),
         'lines 16-19', 'Training set: ' + code('"name": "FFHQ"') + ', ' + code('"dataroot": "dataset/ffhq_16_128"') + '.'],
        ['Codebase, ' + link(BLOB + '/config/sr_sr3_16_128.json#L28-L31', 'config/sr_sr3_16_128.json'),
         'lines 28-31',
         'Validation set: ' + code('"name": "CelebaHQ"') + ', ' + code('"dataroot": "dataset/celebahq_16_128"') + '.'],
        ['Paper, ' + link('https://arxiv.org/abs/2104.07636', 'arXiv:2104.07636'),
         'page 5, Section 4 (Experiments), right column, lines 11-13 (paragraph beginning "Datasets:")',
         'Face models are described as "training face super-resolution models on Flickr-Faces-HQ (FFHQ) and '
         'evaluating on CelebA-HQ".'],
    ], [4.4 * cm, 4.0 * cm, W - 8.4 * cm]))
    e.append(Spacer(1, 5))
    e.append(Paragraph(
        'FFHQ and CelebA-HQ are both collections of aligned, centred photographs of human faces taken with '
        'ordinary cameras in visible light: portraits from Flickr in one case and celebrity photographs in '
        'the other. Kvasir-SEG has nothing in common with that domain. Its images are frames from colonoscopy '
        'examinations, recorded inside the large bowel with a medical endoscope. They show mucosa, folds, '
        'polyps, fluid and strong specular highlights from the endoscope light, usually inside a dark '
        'border, sometimes with an overlaid position marker. There are no faces, no '
        'people and no everyday objects in any of them, and the colour distribution (dominated by reds and '
        'pinks), the textures and the scene geometry are all different from portrait photography. The '
        'pretraining pipeline reads only the two face folders named in the config above, so Kvasir-SEG could '
        'not have entered pretraining, and a model that has only ever seen faces has had no exposure to '
        'endoscopic imagery. This is exactly why we chose it: any improvement between Run 0 and the '
        'fine-tuned runs has to come from the fine-tuning.', BODY))

    # 5 ---------------------------------------------------------------
    e.append(Paragraph('5. Native noise parameterization', H2))
    e.append(Paragraph(
        'We take <b>' + TH + ' = ' + AB + '</b>, the cumulative product of the per-step signal retention '
        'factors, ' + AB + ' = ∏<sub>s=1..t</sub> (1 − β<sub>s</sub>). The SR3 paper calls this quantity '
        'γ<sub>t</sub> (Section 2.1, page 3: "γ<sub>t</sub> = ∏<sub>i=1..t</sub> α<sub>i</sub>") and states '
        '"Our model conditions on γ directly (vs t as in [17])" (Section 2.4, page 4). The implementation matches this: the forward process is written purely in terms '
        'of ' + AB + ', and the network receives √' + AB + ' as its noise-level input instead of the timestep '
        'index. In ' + link(BLOB + '/model/sr3_modules/diffusion.py#L95-L106', 'model/sr3_modules/diffusion.py') + ':', BODY))
    e.append(table([
        ['Lines', 'Role'],
        ['95-103', 'Builds the linear β<sub>t</sub> grid from the config and derives '
                   + code('alphas_cumprod = np.cumprod(1 - betas)') + ', i.e. ' + AB + '.'],
        ['105-106', 'Stores √' + AB + ' (with ᾱ<sub>0</sub> = 1 prepended) as the table of noise levels.'],
        ['212-219', 'Forward process: x<sub>t</sub> = √ᾱ · x<sub>0</sub> + √(1 − ᾱ) · ε.'],
        ['224-231', 'Training draws t, then a continuous noise level uniformly between √ᾱ<sub>t−1</sub> and √ᾱ<sub>t</sub>.'],
        ['153-157', 'At sampling time the network is called with √' + AB + ' as its conditioning input.'],
    ], [2.2 * cm, W - 2.2 * cm]))
    e.append(Spacer(1, 5))
    e.append(Paragraph(
        'The config file specifies the schedule through β<sub>t</sub> (linear from 10<super>−6</super> to '
        '10<super>−2</super>, T = ' + '{:,}'.format(T) + ', '
        + link(BLOB + '/config/sr_sr3_16_128.json#L58-L71', 'config lines 58-71') + '), and ' + AB + ' follows from it '
        'deterministically. We chose ' + AB + ' rather than β<sub>t</sub> as ' + TH + ' because it is the quantity '
        'the forward process and the network actually use, and it is the noise parameter the SR3 paper itself '
        'works with.', BODY))
    e.append(Paragraph(
        '<b>Bound check.</b> Every β<sub>s</sub> lies in [10<super>−6</super>, 10<super>−2</super>], so every '
        'factor 1 − β<sub>s</sub> lies strictly between 0 and 1, and a finite product of such factors is again '
        'strictly between 0 and 1. Numerically (float64), ' + AB + ' decreases monotonically from '
        '<b>{:.6f}</b> at t = 1 to <b>{}</b> at t = {:,}. Hence <b>{} ∈ (0, 1) for all t = 1, …, {:,}</b>. '
        'We also confirmed that the {} buffer stored inside the pretrained checkpoint is identical to this '
        'schedule.'.format(nat['max'], sci(nat['min']), T, TH, T, code('alphas_cumprod')), BODY))

    # 6 ---------------------------------------------------------------
    e.append(Paragraph('6. Chosen exponent', H2))
    e.append(Paragraph(
        '<b>x = ' + x + '</b>, which lies in (1.5, 3.0). Run B uses the schedule {' + TH + '<super>x</super>} = {'
        + AB + '<super>' + x + '</super>}, applied elementwise. Since 0 &lt; ' + AB + ' &lt; 1 and x &gt; 1, the '
        'transformed values are smaller than the originals and still inside (0, 1): they run from '
        '<b>{:.7f}</b> at t = 1 to <b>{}</b> at t = {:,}, and the smallest value is still representable in '
        'single precision. In words, Run B keeps less of the clean image and adds more noise at every '
        'timestep, while both schedules still end in essentially pure noise. Raising the cumulative product '
        'to the power x is the same as raising every per-step factor to the power x, so the modified process '
        'is a valid Gaussian diffusion with per-step variances β′<sub>t</sub> = 1 − (1 − β<sub>t</sub>)<super>x</super>, '
        'which range from {} to {:.4f}.'.format(pw['max'], sci(pw['min']), T,
                                                sci(s['implied_beta_power']['min']),
                                                s['implied_beta_power']['max']), BODY))
    fig = os.path.join(ANALYSIS, 'schedule_comparison.png')
    if os.path.isfile(fig):
        e.append(Image(fig, width=W, height=W * 3.8 / 13.5))
        e.append(Paragraph('Figure 1. Native schedule (blue) and power-transformed schedule with x = ' + x
                           + ' (red): linear scale, log scale, and signal-to-noise ratio.', CAPTION))

    # 7 ---------------------------------------------------------------
    e.append(Paragraph('7. Planned experiments', H2))
    e.append(table([
        ['Run', 'Weights', 'Schedule', 'What is done'],
        ['Run 0', 'Pretrained checkpoint, untouched', AB + ' (native)',
         'The pretrained model is applied directly to the 100 Kvasir-SEG test images. Outputs are saved as the '
         'pre-fine-tuning baseline.'],
        ['Run A', 'Pretrained checkpoint, fine-tuned', AB + ' (native)',
         'Fine-tuned on the 800 training images, then evaluated on the same 100 test images.'],
        ['Run B', 'Pretrained checkpoint, fine-tuned', AB + '<super>' + x + '</super>',
         'Same as Run A in every respect except the schedule, which is used for both training and sampling.'],
    ], [1.5 * cm, 3.6 * cm, 2.6 * cm, W - 7.7 * cm]))
    e.append(Spacer(1, 5))
    e.append(Paragraph('Settings shared by Run A and Run B (the schedule is the only difference):', BODY))
    e.append(table([
        ['Setting', 'Value'],
        ['Data', 'Kvasir-SEG, 800 / 100 / 100 train / validation / test, seed 42; shorter side resized with '
                 'bicubic interpolation and centre-cropped to a square; random horizontal flip during training '
                 '(the codebase default)'],
        ['Starting point', 'Generator weights of ' + code('I640000_E37') + '; new Adam optimizer, iteration counter starts at 0'],
        ['Optimizer', 'Adam, learning rate 1 × 10<super>−4</super> (constant), batch size 4'],
        ['Length', '20,000 iterations, i.e. 100 epochs over the 800 training images'],
        ['Loss and network', 'L1 loss on the predicted noise; U-Net and dropout (0.2) exactly as in the pretrained config'],
        ['Diffusion steps', 'T = {:,} for training and for sampling'.format(T)],
        ['Reproducibility', 'Random seed 42 for both runs; validation on 20 validation images every 2,000 '
                            'iterations; checkpoint every 5,000 iterations; the final (20,000-iteration) checkpoint is the one evaluated'],
        ['Test evaluation', 'All 100 test images, full {:,}-step sampling from the same seeded starting noise in '
                            'all three runs; PSNR, SSIM and LPIPS against the 128×128 ground truth'.format(T)],
    ], [3.4 * cm, W - 3.4 * cm]))

    # 8 ---------------------------------------------------------------
    e.append(Paragraph('8. Decisions and assumptions', H2))
    e.append(Paragraph(
        'The assignment leaves several points open. The choices below were discussed and agreed within the '
        'group before any run was started, and we list them so the setup can be judged as a whole.', BODY))
    items = [
        'The 16×16 → 128×128 checkpoint is used instead of the 64×64 → 512×512 one, so that fine-tuning and '
        'full 2,000-step sampling of the test set fit on a single 12 GB laptop GPU.',
        TH + ' is taken to be ' + AB + ' (the paper\'s γ<sub>t</sub>) and not β<sub>t</sub>, for the reasons given in Section 5.',
        'In Run B the code raises ' + AB + ' to the power x and then recomputes β<sub>t</sub> and all dependent '
        'quantities (posterior mean and variance) from the transformed ' + AB + ', so the reverse process stays '
        'consistent with the modified forward process.',
        'Run B is sampled with the modified schedule it was trained on. Run 0 is sampled with the native schedule.',
        'Fine-tuning loads the pretrained weights only. The Adam state saved with the checkpoint is discarded '
        'in both runs, because it was accumulated under the native schedule on face images and would '
        'otherwise favour Run A.',
        'Only the images of Kvasir-SEG are used; the segmentation masks are ignored.',
        'PSNR and SSIM are computed with the functions shipped in the codebase; LPIPS (AlexNet backbone) is '
        'added as a perceptual measure.',
        'Training statistics for each run are logged to Weights &amp; Biases (project FMGAI-Assn) in addition to the local log files.',
    ]
    for i, it in enumerate(items, 1):
        e.append(Paragraph('{}. {}'.format(i, it), ParagraphStyle('li', parent=BODY, leftIndent=14, firstLineIndent=-14, spaceAfter=3)))

    doc.build(e)
    print('wrote', os.path.join(OUT_DIR, 'Phase1_Declaration.pdf'))


if __name__ == '__main__':
    main()
